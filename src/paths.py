"""
paths.py - distributions and conditional probability paths for flow matching.

SHARED by GameDiT and StrokeDiT. Nothing in this file knows what the data is:
  - data only enters through a LabeledSampleable (defined per project in data.py)
  - conditioning is an opaque dict that is passed straight through

Time convention (MIT 6.S184 notes, Algorithm 3):
    t = 0  ->  pure noise (p_simple)
    t = 1  ->  data       (p_data)
    x_t = alpha_t * z + beta_t * eps
NOTE: the official SiT repo uses the OPPOSITE convention (t=0 is data).
      Do not copy its transport code without flipping t.
"""
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.func import vmap, jacrev


# Conditioning that travels alongside each data sample.
#   GameDiT:        {"y": relief}           y: b
#   StrokeDiT:      {"x_cond": healthy_ct}  x_cond: b c h w
#   Unconditional:  {}
# The path never touches it; only the model and trainer do.
Condition = Dict[str, torch.Tensor]


def expand_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape a per-sample scalar so it broadcasts against x.
    Args:
        - t: b  (or anything with b elements, e.g. b 1 1 1)
        - x: b ...
    Returns:
        - t: b 1 1 ... (same number of dims as x)
    """
    return t.reshape(-1, *([1] * (x.dim() - 1)))


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: b ...
        """
        pass


class LabeledSampleable(ABC):
    """
    Joint distribution over (data, conditioning).
    Each project implements exactly one of these in its data.py.
    """
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Condition]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - z: b ...           (data)
            - cond: Condition    (dict of tensors with leading dim b, may be empty)
        """
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        Args:
            - shape: shape of one sample, e.g. [1, 256, 256]
            - std: standard deviation
        """
        super().__init__()
        self.shape = shape
        self.std = std
        # Moves with .to(device), so samples land on the right device
        self.register_buffer("dummy", torch.zeros(1), persistent=False)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape, device=self.dummy.device)


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------
class Alpha(ABC):
    def __init__(self):
        # Check alpha_0 = 0
        assert torch.allclose(self(torch.zeros(1)), torch.zeros(1), atol=1e-6)
        # Check alpha_1 = 1
        assert torch.allclose(self(torch.ones(1)), torch.ones(1), atol=1e-6)

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: b
        Returns:
            - alpha_t: b
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t (autodiff fallback; subclasses override with closed form).
        Args:
            - t: b
        Returns:
            - d/dt alpha_t: b
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)


class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(self(torch.zeros(1)), torch.ones(1), atol=1e-6)
        # Check beta_1 = 0
        assert torch.allclose(self(torch.ones(1)), torch.zeros(1), atol=1e-6)

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates beta_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: b
        Returns:
            - beta_t: b
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t (autodiff fallback; subclasses override with closed form).
        Args:
            - t: b
        Returns:
            - d/dt beta_t: b
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)


class LinearAlpha(Alpha):
    """
    alpha_t = t   (CondOT / SiT "Linear" interpolant - the default)
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class LinearBeta(Beta):
    """
    beta_t = 1 - t
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


class CosineAlpha(Alpha):
    """
    alpha_t = sin(pi/2 * t)   (SiT "GVP" interpolant - optional alternative)
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(0.5 * math.pi * t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * math.pi * torch.cos(0.5 * math.pi * t)


class CosineBeta(Beta):
    """
    beta_t = cos(pi/2 * t)
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(0.5 * math.pi * t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * math.pi * torch.sin(0.5 * math.pi * t)


# ---------------------------------------------------------------------------
# Probability paths
# ---------------------------------------------------------------------------
class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: LabeledSampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: b
        Returns:
            - x: b ...
        """
        num_samples = t.shape[0]
        z, _ = self.sample_conditioning_variable(num_samples)  # b ...
        return self.sample_conditional_path(z, t)              # b ...

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Condition]:
        """
        Samples the data point z and its conditioning
        Args:
            - num_samples: the number of samples
        Returns:
            - z: b ...
            - cond: Condition
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from p_t(x|z)
        Args:
            - z: b ...
            - t: b
        Returns:
            - x: b ...
        """
        pass

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates u_t(x|z)
        Args:
            - x: b ...
            - z: b ...
            - t: b
        Returns:
            - u_t(x|z): b ...
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates grad log p_t(x|z)
        Args:
            - x: b ...
            - z: b ...
            - t: b
        Returns:
            - score: b ...
        """
        pass


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """
    p_t(x|z) = N(alpha_t z, beta_t^2 I)

    Works for any data shape (b c h w for heightmaps or CT slices, b d for toy data).
    """
    def __init__(self, p_data: LabeledSampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Condition]:
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x, _ = self.sample_conditional_path_with_noise(z, t)
        return x

    def sample_conditional_path_with_noise(self, z: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Same as sample_conditional_path, but also returns the noise used.
        Args:
            - z: b ...
            - t: b
        Returns:
            - x: b ...
            - eps: b ...
        """
        t = t.reshape(-1)
        alpha_t = expand_like(self.alpha(t), z)  # b 1 1 1
        beta_t = expand_like(self.beta(t), z)    # b 1 1 1
        eps = torch.randn_like(z)
        return alpha_t * z + beta_t * eps, eps

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        u_t(x|z) written in terms of x (notes, eq. 29).
        Divides by beta_t, so it is undefined at t = 1.
        """
        t = t.reshape(-1)
        alpha_t = expand_like(self.alpha(t), x)
        beta_t = expand_like(self.beta(t), x)
        dt_alpha_t = expand_like(self.alpha.dt(t), x)
        dt_beta_t = expand_like(self.beta.dt(t), x)
        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_vector_field_from_noise(self, z: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        u_t(x|z) written in terms of the noise (notes, eq. 31):
            u = alpha'_t z + beta'_t eps      (linear path: z - eps)
        Mathematically identical to conditional_vector_field, but never divides
        by beta_t, so it stays finite all the way to t = 1. Use this for training.
        Args:
            - z: b ...
            - eps: b ...
            - t: b
        Returns:
            - u_t(x|z): b ...
        """
        t = t.reshape(-1)
        dt_alpha_t = expand_like(self.alpha.dt(t), z)
        dt_beta_t = expand_like(self.beta.dt(t), z)
        return dt_alpha_t * z + dt_beta_t * eps

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1)
        alpha_t = expand_like(self.alpha(t), x)
        beta_t = expand_like(self.beta(t), x)
        return (z * alpha_t - x) / beta_t ** 2


# ---------------------------------------------------------------------------
# Self-test:  python paths.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    class _ToyData(LabeledSampleable):
        """Stand-in for a real dataset: random images plus a scalar condition."""
        def __init__(self, shape):
            self.shape = shape

        def sample(self, n):
            return torch.randn(n, *self.shape), {"y": torch.rand(n)}

    for name, (alpha, beta) in {
        "linear": (LinearAlpha(), LinearBeta()),
        "cosine": (CosineAlpha(), CosineBeta()),
    }.items():
        # 1-channel (GameDiT-like) and 2-channel (multi-channel data) shapes
        for shape in ([1, 32, 32], [2, 32, 32]):
            path = GaussianConditionalProbabilityPath(_ToyData(shape), shape, alpha, beta)
            z, cond = path.sample_conditioning_variable(8)
            assert z.shape == (8, *shape) and cond["y"].shape == (8,)

            # Endpoints: t=1 gives the data exactly, t=0 gives pure noise
            x1 = path.sample_conditional_path(z, torch.ones(8))
            assert torch.allclose(x1, z, atol=1e-5), "t=1 should return the data"
            x0, eps0 = path.sample_conditional_path_with_noise(z, torch.zeros(8))
            assert torch.allclose(x0, eps0, atol=1e-5), "t=0 should return the noise"

            # The two vector-field formulas must agree away from t=1
            t = torch.rand(8) * 0.98
            x, eps = path.sample_conditional_path_with_noise(z, t)
            u_x = path.conditional_vector_field(x, z, t)
            u_eps = path.conditional_vector_field_from_noise(z, eps, t)
            assert torch.allclose(u_x, u_eps, atol=1e-3), f"{name}: vector fields disagree"

        # Closed-form derivatives match autodiff
        t = torch.rand(16)
        assert torch.allclose(alpha.dt(t), Alpha.dt(alpha, t), atol=1e-5)
        assert torch.allclose(beta.dt(t), Beta.dt(beta, t), atol=1e-5)
        print(f"{name} path OK")

    # Linear path: target velocity is exactly z - eps (Algorithm 3)
    path = GaussianConditionalProbabilityPath(_ToyData([1, 8, 8]), [1, 8, 8], LinearAlpha(), LinearBeta())
    z, _ = path.sample_conditioning_variable(4)
    t = torch.rand(4)
    _, eps = path.sample_conditional_path_with_noise(z, t)
    assert torch.allclose(path.conditional_vector_field_from_noise(z, eps, t), z - eps)
    print("paths.py OK")