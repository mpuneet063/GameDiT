"""
paths.py - distributions and conditional probability paths for flow matching

Time convention:
t = 0 -> pure noise (p_simple)
t = 1 -> data distribution (p_data)
x_t = alpha_t * z + beta_t * x_0 

Note: The time convention is reversed from the original flow matching paper, where t = 0 corresponds to the data distribution and t = 1 corresponds to pure noise. 
This is done to be consistent with the MIT 6.S184 literature.   
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
from torch.func import vmap, jacrev  # vmap batches a function; jacrev computes its Jacobian via reverse-mode autodiff.

# Conditioning that travels alongside each data sample
Condition = Dict[str, torch.Tensor]

def expand_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape a per-sample scalar so it broadcasts against x.

    Args:
        t: b (or anything with b elements, e.g. b 1 1 1)
        x: b ...

    Returns:
        t: b 1 1 ... (same number of dimensions as x)
    """
    return t.reshape(-1, *([1] * (x.dim() - 1)))

# =========================
# Distributions
# =========================

class Sampleable(ABC):
    """
    Abstract base class for distributions that can be sampled from.
    """

    @abstractmethod
    def sample(self,num_samples: int) -> torch.Tensor:
        """
        Sample from the distribution.

        Args:
            num_samples: Number of samples to draw.

        Returns:
            x: b ...
        """
        pass

class LabeledSampleable(Sampleable):
    """
    Abstract base class for distributions that can be sampled from and have labels.
    """
    """
    Abstract base class for distributions that can be sampled from and have labels.
    """

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Condition]:
        """
        Sample from the distribution.

        Args:
            num_samples: Number of samples to draw.

        Returns:
            z: b ...    (data)
            cond: Condition (dict of tensors with leading dim b, may be empty)
        """
        pass

class IsotropicGaussian(nn.Module, Sampleable):
    """
    Isotropic Gaussian distribution with mean 0 and variance 1.
    """

    def __init__(self, shape: List[int], std: float = 1.0):
        """
        Args:
            shape: Shape of the samples to draw (excluding batch dimension).
            std: Standard deviation of the Gaussian.
        """
        super().__init__()
        self.shape = shape
        self.dim = math.prod(shape)
        self.std = std
        # moves with .to(device), so samples land on the right device
        self.register_buffer("dummy", torch.zeros(1), persistent=False)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample from the isotropic Gaussian distribution.

        Args:
            num_samples: Number of samples to draw.
        Returns:
            x: b ...
        """
        return self.std * torch.randn((num_samples, *self.shape), device=self.dummy.device)

# ========================
# Noise schedulers
# ========================

class Alpha(ABC):
    def __init__(self):
        # check that alpha(0) = 0
        assert torch.allclose(self(torch.zeros(1)), torch.zeros(1), atol=1e-6)
        # check that alpha(1) = 1
        assert torch.allclose(self(torch.ones(1)), torch.ones(1), atol=1e-6)

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha(t) for a given time t. Should satisfy alpha(0) = 0 and alpha(1) = 1.

        Args:
            t: Tensor of shape (b,) representing time values in [0, 1].
        Returns:
            alpha: Tensor of shape (b,) representing the alpha values.
        """
        pass
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of alpha(t) with respect to t.

        Args:
            t: Tensor of shape (b,) representing time values in [0, 1].
        Returns:
            d_alpha_dt: Tensor of shape (b,) representing the derivative of alpha with respect to t.
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)  # shape (b,)

class Beta(ABC):
    def __init__(self):
        # check that beta(0) = 1
        assert torch.allclose(self(torch.zeros(1)), torch.ones(1), atol=1e-6)
        # check that beta(1) = 0
        assert torch.allclose(self(torch.ones(1)), torch.zeros(1), atol=1e-6)

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute beta(t) for a given time t. Should satisfy beta(0) = 1 and beta(1) = 0.

        Args:
            t: Tensor of shape (b,) representing time values in [0, 1].
        Returns:
            beta: Tensor of shape (b,) representing the beta values.
        """
        pass
    
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of beta(t) with respect to t.

        Args:
            t: Tensor of shape (b,) representing time values in [0, 1].
        Returns:
            d_beta_dt: Tensor of shape (b,) representing the derivative of beta with respect to t.
        """
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1)  # shape (b,)

class LinearAlpha(Alpha):
    """
    Linear alpha(t) = t.
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

class LinearBeta(Beta):
    """
    Linear beta(t) = 1 - t.
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

class CosineAlpha(Alpha):
    """
    Cosine alpha(t) = sin(pi/2 * t).
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(0.5 * math.pi * t)
        
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * math.pi * torch.cos(0.5 * math.pi * t)

class CosineBeta(Beta):
    """
    Cosine beta(t) = cos(pi/2 * t).
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(0.5 * math.pi * t)
        
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * math.pi * torch.sin(0.5 * math.pi * t)

# ========================
# Probability paths
# ========================

class CondtionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths.
    """

    @abstractmethod
    def __init__(self, p_simple: Sampleable, p_data: LabeledSampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t:torch.Tensor) -> torch.Tensor:
        """
        Sample from the marginal distribution at time t. p_t(x) = p_t(x | z) p(z)

        Args:
            t: Tensor of shape (b,) representing time values in [0, 1].
        Returns:
            x: b ...
        """
        num_samples = t.shape[0]
        z, _ = self.sample_conditioning_variable(num_samples) # b ...
        return self.sample_conditional_path(z, t) # b ...

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Condition]:
        """
        Sample the data point z and its conditioning from the data distribution p_data(z).
        Args:
            num_samples: Number of samples to draw.
        Returns:
            z: b ...    (data)
            cond: Condition (dict of tensors with leading dim b, may be empty)
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from the conditional distribution p_t(x | z) at time t.

        Args:
            z: b ...    (data)
            t: Tensor of shape (b,) representing time values in [0, 1].
        Returns:
            x: b ...
        """
        pass
    
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional vector field v_t(x | z) at time t.

        Args:
            x: b ...    (data)
            z: b ...    (data)
            t: Tensor of shape (b,) representing time values in [0, 1].
        Returns:
            u_t(x|z): b ...    (vector field which will be the Neural Network)
        """
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional score function s_t(x | z) at time t.

        Args:
            x: b ...    (data)
            z: b ...    (data)
            t: Tensor of shape (b,) representing time values in [0, 1].
        Returns:
            s_t(x|z): b ...    (score function)
        """
        pass

class GaussianConditionalProbabilityPath(CondtionalProbabilityPath):
    """
    p_t(x | z) = N(x; alpha(t) * z, beta(t)^2 * I)
    """

    def __init__(self, p_data: LabeledSampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        """
        Args:
            p_data: Data distribution (LabeledSampleable).
            p_simple_shape: Shape of the simple distribution (excluding batch dimension).
            alpha: Alpha function.
            beta: Beta function.
        """
        p_simple = IsotropicGaussian(shape=p_simple_shape, std=1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta

    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, Condition]:
        return self.p_data.sample(num_samples)  # z, cond

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x, _ = self.sample_conditional_path_with_noise(z, t)
        return x

    def sample_conditional_path_with_noise(self, z: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the conditional distribution p_t(x | z) at time t with added noise.

        Args:
            z: b ...    (data)
            eps: b ...  (noise)
            t: Tensor of shape (b,) representing time values in [0, 1].
        Returns:
            u_t(x|z): b ... 
        """
        t = t.reshape(-1)
        alpha_t = expand_like(self.alpha(t), z)
        beta_t = expand_like(self.beta(t), z)
        eps = torch.randn_like(z) 
        return alpha_t * z + beta_t * eps, eps

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1)
        alpha_t = expand_like(self.alpha(t), x)
        beta_t = expand_like(self.beta(t), x)
        dt_alpha_t = expand_like(self.alpha.dt(t), x)
        dt_beta_t = expand_like(self.beta.dt(t), x)
        return (dt_alpha_t - dt_beta_t/beta_t * alpha_t)* z + dt_beta_t/beta_t * x

    def conditional_vector_field_from_noise(self, z: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1)
        dt_alpha_t = expand_like(self.alpha.dt(t), z)
        dt_beta_t = expand_like(self.beta.dt(t), z)
        return dt_alpha_t * z + dt_beta_t * eps

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1)
        alpha_t = expand_like(self.alpha(t), x)
        beta_t = expand_like(self.beta(t), x)
        return (z*alpha_t - x) / (beta_t ** 2)

# ========================
# sanity checks
# ========================

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
