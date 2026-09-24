"""
simulators.py - turning a learned vector field into samples.


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

# ==============================
# Model interface
# =============================

class ConditionalVectorField(nn.Module, ABC):
    """
    A conditional vector field is a function f(x, t, c) that takes in a state x, a time t, and a condition c,
    and returns a vector field f(x, t, c) of the same shape as x.
    """

    @abstractmethod
    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: Condition,
            drop_cond: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Compute the vector field at state x, time t, and condition c.
            
            Args:
                x: b ...
                t: b
                cond: Condition (dict of tensors with leading batch dimension b)
                drop_cond: b (bool) or None (=drop nothing)
            Returns:
                ut_theta(x|cond): b ...
            """
            pass

# =============================
# ode / sde
# =============================

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the drift coefficient of the ODE at state xt and time t.
        
        Args:
            xt: b ...
            t: b
            kwargs: additional arguments
        Returns:
            drift coefficient = h*ut(x)dt : b ...
        """
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the drift coefficient of the SDE at state xt and time t.
        
        Args:
            xt: b ...
            t: b
            kwargs: additional arguments
        Returns:
            drift coefficient = h*ut(x)dt : b ...
        """
        pass
    
    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute the diffusion coefficient of the SDE at state xt and time t.

        Args:
            xt: b ...
            t: b
            kwargs: additional arguments
        Returns:
            diffusion coefficient = sigma(t)*dWt : b ...

        """
        pass


class CFGVectorFieldODE(ODE):
    """
    Classifier-free guidance for an ODE.
    u = (1 - w) * u(x | null) + w * u(x | cond)
    """

    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, cond: Condition) -> torch.Tensor:
        """
        Compute the drift coefficient of the ODE at state x and time t, with classifier-free guidance.
        
        Args:
            x: b ...
            t: b
            cond: Condition (dict of tensors with leading batch dimension b)
        Returns:
            drift coefficient = h*ut(x|cond)dt : b ...
        """
        b = x.shape[0]
        if self.guidance_scale == 1.0:
            return self.net(x, t, cond)

        # Stack [conditioned batch ; unconditioned batch] and compute vector field in one forward pass
        x2 = torch.cat([x, x], dim=0)
        t2 = torch.cat([t, t], dim=0)
        cond2 = {k: torch.cat([v, v], dim=0) for k, v in cond.items()}
        drop = torch.cat([
            torch.zeros(b, dtype=torch.bool, device=x.device),
            torch.ones(b, dtype=torch.bool, device=x.device)
        ])
        u_cond, u_null = self.net(x2, t2, cond2, drop_cond=drop).chunk(2, dim=0)
        return (1-self.guidance_scale) * u_null + self.guidance_scale * u_cond
        # return Unguided Vector Field + Guided Vector Field

# =============================
# Simulators
# =============================

class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.tensor, **kwargs) -> torch.Tensor:
        """
        Take one simulation step
        Args:
            xt: b ...
            t: b
            h: b (step size)
        Returns:
            nxt: b ...
        """
        pass
    
    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs) -> torch.Tensor:
        """
        Simulates using the discretization given by ts
        Args:
            x: b ...
            ts: b nt
        Returns:
            x_traj: b nt ...
        """
        x_traj = [x.clone()]
        nts = ts.shape[1]
        pbar = tqdm(range(nts-1)) if use_tqdm else range(nts-1)
        for t_idx in pbar:
            t = ts[:, t_idx]
            h = ts[:, t_idx+1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            x_traj.append(x.clone())
        return torch.stack(x_traj, dim=1)

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, use_tqdm: bool = True, **kwargs) -> torch.Tensor:
        """
        Simulates and keeps every intermediate step (useful for debugging)
        Args:
            x: b ...
            ts: b nt
        Returns:
            x_traj: b nt ...
        """
        x_traj = [x.clone()]
        nts = ts.shape[1]
        pbar = tqdm(range(nts-1)) if use_tqdm else range(nts-1)
        for t_idx in pbar:
            t = ts[:, t_idx]
            h = ts[:, t_idx+1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            x_traj.append(x.clone())
        return torch.stack(x_traj, dim=1)

class EulerSimulator(Simulator):
    """
    first order: x_{t+h} = x_t + h * u(x_t, t)
    """
    def __init__(self, ode: ODE):
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.tensor, **kwargs) -> torch.Tensor:
        h = expand_like(h, xt)
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h

class HeunSimulator(Simulator):
    """
    second order (predictor-corrector): Takes an Euler step, re-evaluates the vector field,
    and averages the two slopes.
    Two model calls per step, but usually needs far fewer steps than Euler for the same quality.
    """
    def __init__(self, ode: ODE):
        self.ode = ode
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.tensor, **kwargs) -> torch.Tensor:
        h_expanded = expand_like(h, xt)
        k1 = self.ode.drift_coefficient(xt, t, **kwargs)
        x_pred = xt + k1 * h_expanded
        k2 = self.ode.drift_coefficient(x_pred, t + h, **kwargs)
        return xt + 0.5 * (k1 + k2) * h_expanded

class EulerMaruyamaSimulator(Simulator):
    """
    Optional SDE sampling
    """
    def __init__(self, sde: SDE):
        self.sde = sde
    
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.tensor, **kwargs) -> torch.Tensor:
        h = expand_like(h, xt)
        return (
            xt
            + self.sde.drift_coefficient(xt, t, **kwargs) * h
            + self.sde.diffusion_coefficient(xt, t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)
        )

# =============================
# Helpers
# ===========================

def make_time_grid(num_samples: int, num_steps: int, device: torch.device) -> torch.Tensor:
    """
    Make a time grid for simulation.
    Args:
        num_samples: number of samples in the batch
        num_steps: number of steps in the simulation
        device: device to place the tensor on
    Returns:
        ts: b nt
    """
    return torch.linspace(0.0, 1.0, num_steps, device=device).expand(num_samples, -1)

def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Make a time grid for simulation, but only record every `record_every` steps.
    Args:
        num_timesteps: number of timesteps in the simulation
        record_every: record every `record_every` steps
    Returns:
        ts: nt
    """
    if record_every == 1:
        return torch.arange(num_timesteps, dtype=torch.float32)
    return torch.cat([
        torch.arange(0, num_timesteps - 1, record_every),
        torch.tensor([num_timesteps - 1], dtype=torch.float32)
    ])

# ===========================
# sanity checks
# ========================

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
        assert torch.allclose(x1[:, -1], mu.expand_as(x1[:, -1]), atol=1e-2), f"{Sim.__name__} did not reach the target"
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
    assert traj.shape == (b, ts.shape[1], *shape)
    print("simulators.py OK")