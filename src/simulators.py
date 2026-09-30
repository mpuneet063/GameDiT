"""
simulators.py - turning a learned vector field into samples.

SHARED by GameDiT and StrokeDiT.

Contents
  - ConditionalVectorField : the interface every model implements (model.py subclasses it)
  - ODE / SDE              : what to simulate
  - Simulator              : how to step through time (Euler, Heun, Euler-Maruyama)
  - CFGVectorFieldODE      : classifier-free guidance, generalised to a condition dict

Time convention (same as paths.py): t = 0 noise  ->  t = 1 data.
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from paths import Condition, expand_like


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------
class ConditionalVectorField(nn.Module, ABC):
    """
    Conditional vector field u_t^theta(x | cond)

    `drop_cond` replaces the notebook's null label:
      True for a sample  ->  the model ignores that sample's droppable conditioning
                             (uses its learned null embedding instead).
    Which conditions are droppable is decided by the model, not the simulator,
    so this interface is identical for GameDiT and StrokeDiT.
    """
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Condition,
        drop_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            - x: b ...
            - t: b
            - cond: Condition (dict of tensors with leading dim b)
            - drop_cond: b (bool) or None (= drop nothing)
        Returns:
            - u_t^theta(x|cond): b ...
        """
        pass


# ---------------------------------------------------------------------------
# ODE / SDE
# ---------------------------------------------------------------------------
class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            - xt: b ...
            - t: b
        Returns:
            - drift_coefficient: b ...
        """
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            - xt: b ...
            - t: b
        Returns:
            - drift_coefficient: b ...
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            - xt: b ...
            - t: b
        Returns:
            - diffusion_coefficient: b ...
        """
        pass


class CFGVectorFieldODE(ODE):
    """
    Classifier-free guidance:
        u = (1 - w) * u(x | null) + w * u(x | cond)

    Differences from the notebook version:
      - works with a condition dict instead of an integer label
      - the guided and unguided passes run as ONE batched forward pass (about 2x faster)
      - w = 1 skips the unguided pass entirely
    """
    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, cond: Condition) -> torch.Tensor:
        """
        Args:
            - x: b ...
            - t: b
            - cond: Condition
        Returns:
            - guided vector field: b ...
        """
        b = x.shape[0]
        if self.guidance_scale == 1.0:
            return self.net(x, t, cond)

        # Stack [conditioned batch ; unconditioned batch] and run once
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        cond2 = {k: torch.cat([v, v], dim=0) for k, v in cond.items()}
        drop = torch.cat([
            torch.zeros(b, dtype=torch.bool, device=x.device),
            torch.ones(b, dtype=torch.bool, device=x.device),
        ])
        u_cond, u_null = self.net(x2, t2, cond2, drop_cond=drop).chunk(2, dim=0)
        return (1 - self.guidance_scale) * u_null + self.guidance_scale * u_cond


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------
class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Takes one simulation step
        Args:
            - xt: b ...
            - t: b
            - h: b (step size)
        Returns:
            - nxt: b ...
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs) -> torch.Tensor:
        """
        Simulates using the discretization given by ts
        Args:
            - x: b ...     (initial state, i.e. noise)
            - ts: b nt
        Returns:
            - x_final: b ...
        """
        nts = ts.shape[1]
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs) -> torch.Tensor:
        """
        Simulates and keeps every intermediate state (useful for denoising GIFs)
        Args:
            - x: b ...
            - ts: b nt
        Returns:
            - x_traj: b nt ...
        """
        x_traj = [x.clone()]
        nts = ts.shape[1]
        pbar = tqdm(range(nts - 1)) if use_tqdm else range(nts - 1)
        for t_idx in pbar:
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            x_traj.append(x.clone())
        return torch.stack(x_traj, dim=1)


class EulerSimulator(Simulator):
    """
    First order: x_{t+h} = x_t + h * u(x_t, t)
    """
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        h = expand_like(h, xt)
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h


class HeunSimulator(Simulator):
    """
    Second order (predictor-corrector): takes an Euler step, re-evaluates the
    vector field at the landing point, and averages the two slopes.
    Two model calls per step, but usually needs far fewer steps than Euler
    for the same quality.
    """
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        h_ = expand_like(h, xt)
        u1 = self.ode.drift_coefficient(xt, t, **kwargs)
        x_pred = xt + h_ * u1
        u2 = self.ode.drift_coefficient(x_pred, t + h, **kwargs)
        return xt + 0.5 * h_ * (u1 + u2)


class EulerMaruyamaSimulator(Simulator):
    """
    Kept from the notebook for completeness (SDE sampling).
    """
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs) -> torch.Tensor:
        h = expand_like(h, xt)
        return (
            xt
            + self.sde.drift_coefficient(xt, t, **kwargs) * h
            + self.sde.diffusion_coefficient(xt, t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_time_grid(num_samples: int, num_steps: int, device: torch.device) -> torch.Tensor:
    """
    Uniform grid from noise (t=0) to data (t=1).
    Returns:
        - ts: b nt   with nt = num_steps + 1
    """
    return torch.linspace(0.0, 1.0, num_steps + 1, device=device).expand(num_samples, -1)


def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat([
        torch.arange(0, num_timesteps - 1, record_every),
        torch.tensor([num_timesteps - 1]),
    ])


# ---------------------------------------------------------------------------
# Self-test:  python simulators.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    b, shape = 16, (1, 8, 8)

    class _ExactField(ConditionalVectorField):
        """
        If every data point is the same image mu, the true linear-path velocity is
            u(x, t) = (mu - x) / (1 - t)
        cond["y"] scales mu; dropping the condition makes the target all zeros.
        """
        def __init__(self, mu):
            super().__init__()
            self.mu = mu

        def forward(self, x, t, cond, drop_cond=None):
            target = self.mu * expand_like(cond["y"], x)
            if drop_cond is not None:
                target = torch.where(expand_like(drop_cond, x), torch.zeros_like(target), target)
            return (target - x) / expand_like(1 - t, x)

    mu = torch.randn(1, *shape)
    net = _ExactField(mu)
    cond = {"y": torch.ones(b)}
    x0 = torch.randn(b, *shape)
    ts = make_time_grid(b, 20, x0.device)

    # 1. Euler and Heun both land on mu (straight-line flow is solved exactly)
    for Sim in (EulerSimulator, HeunSimulator):
        # End at t=0.999, not 1: the exact field divides by (1 - t), and Heun
        # evaluates it at the end of each step
        x1 = Sim(CFGVectorFieldODE(net, 1.0)).simulate(x0, ts * 0.999, use_tqdm=False, cond=cond)
        assert torch.allclose(x1, mu.expand_as(x1), atol=1e-2), f"{Sim.__name__} did not reach the target"
    print("Euler / Heun OK")

    # 2. Batched CFG equals the formula computed with two separate passes
    w = 3.0
    xt, t = torch.randn(b, *shape), torch.rand(b) * 0.9
    batched = CFGVectorFieldODE(net, w).drift_coefficient(xt, t, cond)
    u_c = net(xt, t, cond)
    u_n = net(xt, t, cond, drop_cond=torch.ones(b, dtype=torch.bool))
    assert torch.allclose(batched, (1 - w) * u_n + w * u_c, atol=1e-5)
    print("CFG OK")

    # 3. Trajectory shape
    traj = EulerSimulator(CFGVectorFieldODE(net, 1.0)).simulate_with_trajectory(
        x0, ts * 0.999, use_tqdm=False, cond=cond)
    assert traj.shape == (b, 21, *shape)
    print("simulators.py OK")