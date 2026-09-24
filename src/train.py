"""
train.py - training loop for the shared SiT model.

SHARED by GameDiT and StrokeDiT. Same structure as the 6.S184 lab
(Trainer -> CFGTrainer, get_train_loss, warmup LR, checkpoint callback), plus:
  - EMA weights (samples come from the EMA copy)
  - bf16 mixed precision on GPU, gradient clipping
  - checkpoint + automatic resume (a crashed / preempted run just restarts)
  - sample grids saved at every checkpoint
  - optional `aws s3 sync` of the run folder

Usage
  python train.py --config configs/gamedit.yaml
  python train.py --selftest              (tiny CPU run, checks everything works)

The only project-specific piece is the data module named in the config:
it must expose  build_sampler(**kwargs) -> LabeledSampleable
and may expose  visualize(x: Tensor[b, c, h, w]) -> ndarray[b, h, w(, 3)] in [0, 1]
"""
import argparse
import contextlib
import copy
import csv
import glob
import importlib
import math
import os
import random
import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from paths import (
    CosineAlpha, CosineBeta, GaussianConditionalProbabilityPath,
    LabeledSampleable, LinearAlpha, LinearBeta,
)
from simulators import CFGVectorFieldODE, EulerSimulator, HeunSimulator, make_time_grid
from model import SIT_PRESETS, DiffusionTransformerFlowModel, count_params

MiB = 1024 ** 2


def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    """
    Exponential moving average of the model weights:
        ema = decay * ema + (1 - decay) * current
    Individual training steps are noisy; the average is a much smoother model
    and is what we sample from.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1.0 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def _default_visualize(x: torch.Tensor):
    """First channel of each sample, min-max scaled to [0, 1]. Returns b h w."""
    x = x[:, 0]
    lo = x.amin(dim=(1, 2), keepdim=True)
    hi = x.amax(dim=(1, 2), keepdim=True)
    return ((x - lo) / (hi - lo + 1e-8)).numpy()


def save_grid(x: torch.Tensor, path: str, visualize: Optional[Callable] = None, nrow: int = 4):
    """
    Args:
        - x: b c h w
        - path: .png file
        - visualize: optional project-specific renderer (e.g. hillshade for heightmaps)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    imgs = (visualize or _default_visualize)(x.detach().float().cpu())
    imgs = np.asarray(imgs)
    b, h, w = imgs.shape[:3]
    ncol = min(nrow, b)
    nrows = math.ceil(b / ncol)
    pad = 2
    shape = (nrows * (h + pad) - pad, ncol * (w + pad) - pad) + imgs.shape[3:]
    grid = np.ones(shape, dtype=np.float32)
    for i in range(b):
        r, c = divmod(i, ncol)
        grid[r * (h + pad): r * (h + pad) + h, c * (w + pad): c * (w + pad) + w] = imgs[i]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, np.clip(grid, 0, 1), cmap="gray" if grid.ndim == 2 else None)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer(ABC):
    def __init__(
        self,
        ema_decay: float = 0.9999,
        grad_clip: Optional[float] = 1.0,
        weight_decay: float = 1e-4,
        use_bf16: bool = True,
        keep_last: int = 3,
        runs_root: str = "runs",
        s3_uri: Optional[str] = None,
        config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.model = None       # what the loop calls (possibly torch.compile'd)
        self.raw_model = None   # the underlying nn.Module (for saving / EMA)
        self.ema = None
        self.opt = None
        self.output_dir = None
        self.device = None
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.use_bf16 = use_bf16
        self.keep_last = keep_last
        self.runs_root = runs_root
        self.s3_uri = s3_uri
        self.config = config or {}
        self._warned_s3 = False

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def on_checkpoint(self, step: int):
        """Hook for subclasses (e.g. save sample grids). Runs after each checkpoint."""
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.AdamW(self.raw_model.parameters(), lr=lr, weight_decay=self.weight_decay)

    def autocast(self):
        if self.use_bf16 and self.device is not None and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def random_name(self) -> str:
        adjectives = ["autumn", "hidden", "bitter", "misty", "silent", "empty", "dry", "dark", "summer", "icy"]
        foods = ["apple", "banana", "pear", "plum", "orange", "persimmon", "tangerine", "durian", "jackfruit", "peach"]
        return f"{random.choice(adjectives)}-{random.choice(foods)}-{str(uuid.uuid4())[:8]}"

    # --- checkpointing -----------------------------------------------------
    def _ckpt_dir(self) -> str:
        return os.path.join(self.output_dir, "checkpoints")

    def checkpoint(self, step: int):
        os.makedirs(self._ckpt_dir(), exist_ok=True)
        state = {
            "step": step,
            "model": self.raw_model.state_dict(),
            "ema": self.ema.model.state_dict(),
            "opt": self.opt.state_dict(),
            "config": self.config,
        }
        path = os.path.join(self._ckpt_dir(), f"step_{step:07d}.pt")
        torch.save(state, path)
        shutil.copyfile(path, os.path.join(self._ckpt_dir(), "latest.pt"))

        # Keep only the newest `keep_last` numbered checkpoints (disk space)
        old = sorted(glob.glob(os.path.join(self._ckpt_dir(), "step_*.pt")))[:-self.keep_last]
        for f in old:
            os.remove(f)

        self.model.eval()
        self.on_checkpoint(step)
        self.model.train()
        self.sync()

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.device)
        self.raw_model.load_state_dict(state["model"])
        self.ema.model.load_state_dict(state["ema"])
        self.opt.load_state_dict(state["opt"])
        return int(state["step"])

    def sync(self):
        """Copy the run folder to S3 in the background (needs the aws CLI)."""
        if not self.s3_uri:
            return
        if shutil.which("aws") is None:
            if not self._warned_s3:
                print("[sync] s3_uri set but `aws` CLI not found - skipping S3 sync")
                self._warned_s3 = True
            return
        dest = f"{self.s3_uri.rstrip('/')}/{os.path.basename(self.output_dir)}"
        subprocess.Popen(["aws", "s3", "sync", self.output_dir, dest, "--only-show-errors"])

    # --- main loop -----------------------------------------------------------
    def train(
        self,
        model: nn.Module,
        num_steps: int,
        lr: float = 1e-4,
        warmup_steps: int = 1000,
        ckpt_every: Optional[int] = 5000,
        log_every: int = 100,
        run_name: Optional[str] = None,
        resume: bool = True,
        compile: bool = False,
        **kwargs,
    ) -> Tuple[List[float], List[int]]:
        """
        Linear warmup from 0 -> lr over `warmup_steps`, then constant lr.
        If runs/<run_name>/checkpoints/latest.pt exists and resume=True, training
        continues from there.
        """
        run_name = run_name or self.random_name()
        self.output_dir = os.path.join(self.runs_root, run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print("Output directory: " + self.output_dir)

        self.raw_model = model
        self.device = next(model.parameters()).device
        print(f"Training model with {count_params(model):.1f}M params ({model_size_b(model) / MiB:.1f} MiB)")

        self.ema = EMA(model, self.ema_decay)
        self.opt = self.get_optimizer(lr)

        start_step = 0
        latest = os.path.join(self._ckpt_dir(), "latest.pt")
        if resume and os.path.exists(latest):
            start_step = self.load_checkpoint(latest)
            print(f"Resumed from step {start_step}")

        self.model = torch.compile(model) if compile else model
        self.model.train()

        log_path = os.path.join(self.output_dir, "loss.csv")
        new_log = not os.path.exists(log_path)
        log_file = open(log_path, "a", newline="")
        log = csv.writer(log_file)
        if new_log:
            log.writerow(["step", "loss", "lr", "grad_norm"])

        losses: List[float] = []
        steps: List[int] = []
        last_saved = start_step

        pbar = tqdm(range(start_step, num_steps), initial=start_step, total=num_steps)
        for step in pbar:
            # Update LR
            if warmup_steps > 0 and step < warmup_steps:
                cur_lr = lr * float(step + 1) / float(warmup_steps)
            else:
                cur_lr = lr
            for pg in self.opt.param_groups:
                pg["lr"] = cur_lr

            # Forward + backward
            self.opt.zero_grad(set_to_none=True)
            with self.autocast():
                loss = self.get_train_loss(**kwargs)
            loss.backward()

            grad_norm = float("nan")
            if self.grad_clip:
                grad_norm = float(nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip))

            # Gradient step + EMA
            self.opt.step()
            self.ema.update(self.raw_model)

            loss_val = float(loss.detach().item())
            losses.append(loss_val)
            steps.append(step)
            pbar.set_description(f"loss={loss_val:.4f} lr={cur_lr:.1e}")

            done = step + 1  # number of completed steps
            if done % log_every == 0:
                log.writerow([done, loss_val, cur_lr, grad_norm])
                log_file.flush()
            if ckpt_every is not None and done % ckpt_every == 0:
                self.checkpoint(done)
                last_saved = done

        if last_saved != num_steps and num_steps > start_step:
            self.checkpoint(num_steps)
        log_file.close()
        self.model.eval()
        return losses, steps


class CFGTrainer(Trainer):
    """
    Flow-matching loss with classifier-free guidance dropout (notes, Algorithm 3):
        z, cond ~ p_data;   t ~ U[0, 1];   eps ~ N(0, I)
        x = alpha_t z + beta_t eps
        loss = || u_theta(x, t, cond) - (alpha'_t z + beta'_t eps) ||^2
    Each sample's condition is dropped with probability eta.
    """
    def __init__(
        self,
        path: GaussianConditionalProbabilityPath,
        eta: float = 0.1,
        num_samples: int = 16,
        sampler_steps: int = 50,
        simulator: str = "heun",
        guidance_scale: float = 2.0,
        visualize: Optional[Callable] = None,
        **kwargs,
    ):
        assert 0 <= eta < 1
        super().__init__(**kwargs)
        self.path = path
        self.eta = eta
        self.num_samples = num_samples
        self.sampler_steps = sampler_steps
        self.simulator = simulator
        self.guidance_scale = guidance_scale
        self.visualize = visualize
        self._fixed = None  # fixed noise + conditions, so sample grids are comparable across steps

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: sample z, cond from p_data
        z, cond = self.path.p_data.sample(batch_size)
        z = z.to(self.device)
        cond = {k: v.to(self.device) for k, v in cond.items()}

        # Step 2: drop each sample's condition with probability eta
        drop = None
        if self.eta > 0 and self.raw_model.has_droppable_cond:
            drop = torch.rand(batch_size, device=self.device) < self.eta

        # Step 3: sample t and x (and keep the noise for the target)
        t = torch.rand(batch_size, device=self.device)
        x, eps = self.path.sample_conditional_path_with_noise(z, t)

        # Step 4: regress onto the conditional vector field
        ut_target = self.path.conditional_vector_field_from_noise(z, eps, t)
        ut_theta = self.model(x, t, cond, drop_cond=drop)
        return torch.square(ut_theta.float() - ut_target.float()).mean()

    @torch.no_grad()
    def sample(self, x0: torch.Tensor, cond, model: Optional[nn.Module] = None) -> torch.Tensor:
        """Noise (t=0) -> data (t=1) with the EMA model by default."""
        model = model or self.ema.model
        w = self.guidance_scale if model.has_droppable_cond else 1.0
        Sim = HeunSimulator if self.simulator == "heun" else EulerSimulator
        ts = make_time_grid(x0.shape[0], self.sampler_steps, x0.device)
        with self.autocast():
            x1 = Sim(CFGVectorFieldODE(model, w)).simulate(x0, ts, use_tqdm=False, cond=cond)
        return x1.float()

    @torch.no_grad()
    def on_checkpoint(self, step: int):
        if self.num_samples <= 0:
            return
        sample_dir = os.path.join(self.output_dir, "samples")
        if self._fixed is None:
            z, cond = self.path.p_data.sample(self.num_samples)
            g = torch.Generator().manual_seed(1234)
            noise = torch.randn(z.shape, generator=g).to(self.device)
            cond = {k: v.to(self.device) for k, v in cond.items()}
            self._fixed = (noise, cond)
            save_grid(z, os.path.join(sample_dir, "real.png"), self.visualize)
        noise, cond = self._fixed
        x1 = self.sample(noise, cond)[:, -1]
        save_grid(x1, os.path.join(sample_dir, f"step_{step:07d}.png"), self.visualize)


# ---------------------------------------------------------------------------
# Config entry point
# ---------------------------------------------------------------------------
SCHEDULES = {
    "linear": (LinearAlpha, LinearBeta),
    "cosine": (CosineAlpha, CosineBeta),
}


def build_from_config(cfg: dict, device: torch.device):
    # Data (the only project-specific part)
    data_cfg = cfg["data"]
    data_mod = importlib.import_module(data_cfg.get("module", "data"))
    p_data: LabeledSampleable = data_mod.build_sampler(**data_cfg.get("kwargs", {}))
    visualize = getattr(data_mod, "visualize", None)

    # Model
    mcfg = dict(cfg["model"])
    preset = mcfg.pop("preset", None)
    if preset is not None:
        layers, dim, heads = SIT_PRESETS[preset]
        mcfg.setdefault("num_layers", layers)
        mcfg.setdefault("dim", dim)
        mcfg.setdefault("heads", heads)
    model = DiffusionTransformerFlowModel(**mcfg).to(device)

    # Path
    img_size = mcfg.get("img_size", 256)
    H, W = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
    alpha_cls, beta_cls = SCHEDULES[cfg.get("path", {}).get("schedule", "linear")]
    path = GaussianConditionalProbabilityPath(p_data, [mcfg.get("c", 1), H, W], alpha_cls(), beta_cls()).to(device)

    # Trainer
    tcfg, scfg = cfg.get("train", {}), cfg.get("sample", {})
    trainer = CFGTrainer(
        path,
        eta=tcfg.get("eta", 0.1),
        num_samples=scfg.get("num_samples", 16),
        sampler_steps=scfg.get("steps", 50),
        simulator=scfg.get("simulator", "heun"),
        guidance_scale=scfg.get("guidance_scale", 2.0),
        visualize=visualize,
        ema_decay=tcfg.get("ema_decay", 0.9999),
        grad_clip=tcfg.get("grad_clip", 1.0),
        weight_decay=tcfg.get("weight_decay", 1e-4),
        use_bf16=tcfg.get("bf16", True),
        keep_last=tcfg.get("keep_last", 3),
        runs_root=tcfg.get("runs_root", "runs"),
        s3_uri=tcfg.get("s3_uri"),
        config=cfg,
    )
    return model, trainer


def main(config_path: str):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, trainer = build_from_config(cfg, device)
    tcfg = cfg.get("train", {})
    trainer.train(
        model,
        num_steps=tcfg["num_steps"],
        lr=tcfg.get("lr", 1e-4),
        warmup_steps=tcfg.get("warmup_steps", 1000),
        ckpt_every=tcfg.get("ckpt_every", 5000),
        log_every=tcfg.get("log_every", 100),
        run_name=cfg.get("run_name"),
        resume=tcfg.get("resume", True),
        compile=tcfg.get("compile", False),
        batch_size=tcfg.get("batch_size", 128),
    )


# ---------------------------------------------------------------------------
# Self-test: tiny model, toy data, CPU, ~1 minute
# ---------------------------------------------------------------------------
def _selftest():
    import tempfile

    torch.manual_seed(0)

    class _ToyData(LabeledSampleable):
        """A bright square on a dark background; its brightness is the condition y."""
        def sample(self, n):
            y = torch.rand(n)
            z = -torch.ones(n, 1, 16, 16)
            z[:, :, 4:12, 4:12] = (2 * y - 1).view(-1, 1, 1, 1)
            return z, {"y": y}

    def _make():
        model = DiffusionTransformerFlowModel(img_size=16, patch_size=4, num_layers=2, dim=64,
                                              heads=4, c=1, scalar_conds=["y"])
        path = GaussianConditionalProbabilityPath(_ToyData(), [1, 16, 16], LinearAlpha(), LinearBeta())
        trainer = CFGTrainer(path, eta=0.1, num_samples=4, sampler_steps=10,
                             ema_decay=0.99, runs_root=tmp)
        return model, trainer

    tmp = tempfile.mkdtemp()
    try:
        # 1. Loss goes down
        model, trainer = _make()
        losses, _ = trainer.train(model, num_steps=300, lr=1e-3, warmup_steps=20,
                                  ckpt_every=150, log_every=50, run_name="toy", batch_size=64)
        first, last = sum(losses[:20]) / 20, sum(losses[-50:]) / 50
        print(f"loss: first 20 steps {first:.3f} -> last 50 steps {last:.3f}")
        assert last < 0.8 * first, "loss did not decrease"

        # 2. Files were written
        run = os.path.join(tmp, "toy")
        for f in ["checkpoints/step_0000150.pt", "checkpoints/step_0000300.pt",
                  "checkpoints/latest.pt", "samples/real.png",
                  "samples/step_0000150.png", "samples/step_0000300.png", "loss.csv"]:
            assert os.path.exists(os.path.join(run, f)), f"missing {f}"
        print("checkpoints, sample grids and loss log OK")

        # 3. Resume continues from step 300 instead of starting over
        model2, trainer2 = _make()
        losses2, steps2 = trainer2.train(model2, num_steps=320, lr=1e-3, warmup_steps=20,
                                         ckpt_every=150, log_every=50, run_name="toy", batch_size=64)
        assert steps2[0] == 300 and len(losses2) == 20, "resume did not pick up at step 300"
        print("resume OK")
        print("train.py OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to a YAML config")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    elif args.config:
        main(args.config)
    else:
        parser.print_help()