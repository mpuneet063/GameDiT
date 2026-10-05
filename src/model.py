"""
model.py - SiT (Scalable Interpolant Transformer) backbone.
"""
import math
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from paths import Condition
from simulators import ConditionalVectorField

ImgSize = Union[int, Tuple[int, int]]


def _hw(img_size: ImgSize) -> Tuple[int, int]:
    return (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)


# ---------------------------------------------------------------------------
# Building blocks (from the lab)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: Type[nn.Module] = nn.SiLU, final_init: bool = False):
        super().__init__()
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        self.net = nn.Sequential(*mlp)

        if final_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: b ... d_in
        Returns:
            - x: b ... d_out
        """
        return self.net(x)


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: b
        Returns:
            - embeddings: b d
        """
        f_i = 2 * torch.pi * self.weights * t.unsqueeze(-1)  # b d/2
        emb = torch.cat([torch.sin(f_i), torch.cos(f_i)], dim=-1)  # b d
        return emb * math.sqrt(2)


class Patchifier(nn.Module):
    def __init__(self, img_size: ImgSize, patch_size: int, c_in: int, dim: int):
        super().__init__()
        H, W = _hw(img_size)
        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.dim = dim
        self.net = nn.Sequential(
            # convolution: cut into patches + linear projection in one op
            nn.Conv2d(c_in, dim, kernel_size=patch_size, stride=patch_size),
            # patchify
            Rearrange("b d h w -> b (h w) d"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: b c_in h w
        Returns:
            - x: b n d      (n = h/p * w/p)
        """
        return self.net(x)


class MHA(nn.Module):
    """
    Multi-headed self-attention with QK-norm.

    Without QK-norm, queries and keys can slowly grow during training, so the
    attention scores q.k grow too, softmax saturates and gradients explode
    (our first A100 run collapsed at ~4.3k steps this way).
    QK-norm RMS-normalises q and k per head before the dot product, so every
    score is bounded: |q.k| / sqrt(d_head) <= sqrt(d_head) * gain_q * gain_k.
    """
    def __init__(self, dim: int, heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        # learnable per-channel gain lets the model tune its attention "temperature"
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: b n d
        Returns:
            - x: b n d
        """
        qkv = rearrange(self.qkv(x), "b n (three h dh) -> three b h n dh", three=3, h=self.heads)
        q, k, v = qkv.unbind(0)                   # each b h n dh
        # normalise q and k; cast back so all three share a dtype under bf16 autocast
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)
        out = F.scaled_dot_product_attention(q, k, v)  # fused (flash) kernel on GPU
        out = rearrange(out, "b h n dh -> b n (h dh)")
        return self.proj(out)


def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Args:
        - x: b n d
        - scale: b 1 d
        - bias: b 1 d
    Returns:
        - x: b n d
    """
    return x * (1 + scale) + bias


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        """
        Args:
            - dim: dimension of hidden layers
            - heads: number of attention heads
        """
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, dim * 6),
        )
        # adaLN-Zero: every layer starts as the identity - stabilises the residual stream
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

        self.attn = MHA(dim, heads)
        self.ff = MLP([dim, 4 * dim, dim])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: b n d
            - c: b d
        Returns:
            - x: b n d
        """
        c = rearrange(self.ada_ln(c), "b d -> b 1 d")  # b 1 6d
        attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim=-1)

        x = x + attn_gate * self.attn(modulate(self.norm1(x), attn_scale, attn_bias))
        x = x + ff_gate * self.ff(modulate(self.norm2(x), ff_scale, ff_bias))
        return x


# ---------------------------------------------------------------------------
# Tweak 1: fixed 2D sin-cos positional encodings
# ---------------------------------------------------------------------------
def get_2d_sincos_pos_embed(dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    Half the channels encode the row, half the column; each half is sin/cos
    at geometrically spaced frequencies (slow = coarse position, fast = fine).
    Returns:
        - pos: (grid_h * grid_w) d
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin-cos"
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )

    def _1d(pos: torch.Tensor, d: int) -> torch.Tensor:
        omega = 1.0 / (10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2.0)))  # d/2
        angles = pos.reshape(-1, 1) * omega.reshape(1, -1)  # n d/2
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # n d

    return torch.cat([_1d(yy, dim // 2), _1d(xx, dim // 2)], dim=1)  # n d


class DiffusionTransformer(nn.Module):
    def __init__(self, depth: int, grid_size: Tuple[int, int], dim: int, **layer_kwargs):
        """
        Args:
            - depth: number of layers
            - grid_size: (h/p, w/p) - patch grid, for the positional encodings
            - dim: dimension of hidden layers
            - layer_kwargs: e.g. heads
        """
        super().__init__()
        self.layers = nn.ModuleList([DiffusionTransformerLayer(dim, **layer_kwargs) for _ in range(depth)])
        # Fixed, not learned: no parameters, and each resolution gets its own grid
        pos = get_2d_sincos_pos_embed(dim, *grid_size)
        self.register_buffer("positional_encodings", pos, persistent=False)  # n d

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: b n d
            - c: b d
        Returns:
            - x: b n d
        """
        x = x + self.positional_encodings.unsqueeze(0).to(x.dtype)
        for layer in self.layers:
            x = layer(x, c)
        return x


class Depatchifier(nn.Module):
    def __init__(self, img_size: ImgSize, patch_size: int, dim: int, final_dim: int, c_out: int):
        super().__init__()
        H, W = _hw(img_size)
        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"
        self.net = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            MLP([dim, final_dim * patch_size ** 2, final_dim * patch_size ** 2]),
            Rearrange(
                "b (h w) (f ph pw) -> b f (h ph) (w pw)",
                h=H // patch_size,
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Conv2d(final_dim, c_out, 1, bias=False),
        )
        # Tweak 3: model starts by predicting zero velocity (as in DiT/SiT)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: b n d
        Returns:
            - x: b c_out h w
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Tweak 2: pluggable condition embedder
# ---------------------------------------------------------------------------
class GlobalConditionEmbedder(nn.Module):
    """
    Turns 'whole-image' conditions into one vector that is added to the time
    embedding (the notebook's t_emb + y_emb).

      scalar_conds: continuous values, e.g. ["y"] for GameDiT relief in [0, 1]
      class_conds:  discrete labels,  e.g. {"region": 7}

    Each condition has its own learned null (the notebook's label 10), used
    wherever drop_cond is True - this is what classifier-free guidance needs.
    """
    def __init__(self, dim: int, scalar_conds: List[str], class_conds: Dict[str, int]):
        super().__init__()
        self.dim = dim
        self.scalar_embedders = nn.ModuleDict({k: MLP([1, dim, dim]) for k in scalar_conds})
        self.scalar_nulls = nn.ParameterDict({k: nn.Parameter(torch.zeros(dim)) for k in scalar_conds})
        # index n_classes is reserved as the null label
        self.class_embedders = nn.ModuleDict({k: nn.Embedding(n + 1, dim) for k, n in class_conds.items()})
        self.class_null_idx = dict(class_conds)

    @property
    def keys(self) -> List[str]:
        return list(self.scalar_embedders.keys()) + list(self.class_embedders.keys())

    def forward(self, cond: Condition, drop_cond: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - cond: Condition
            - drop_cond: b (bool) or None
            - like: b d   (the time embedding; sets batch size, device and dtype)
        Returns:
            - emb: b d
        """
        emb = torch.zeros_like(like)
        for k, mlp in self.scalar_embedders.items():
            e = mlp(cond[k].float().reshape(-1, 1))  # b d
            if drop_cond is not None:
                e = torch.where(drop_cond.reshape(-1, 1), self.scalar_nulls[k].expand_as(e), e)
            emb = emb + e
        for k, table in self.class_embedders.items():
            labels = cond[k].long().reshape(-1)
            if drop_cond is not None:
                labels = torch.where(drop_cond, torch.full_like(labels, self.class_null_idx[k]), labels)
            emb = emb + table(labels)
        return emb


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------
SIT_PRESETS = {
    # name: (num_layers, dim, heads)   - same sizes as the DiT/SiT papers
    "S": (12, 384, 6),
    "B": (12, 768, 12),
    "L": (24, 1024, 16),
}


class DiffusionTransformerFlowModel(ConditionalVectorField):
    def __init__(
        self,
        img_size: ImgSize = 256,
        patch_size: int = 8,
        num_layers: int = 12,
        c: int = 1,
        dim: int = 384,
        heads: int = 6,
        final_dim: int = 16,
        scalar_conds: Optional[List[str]] = None,
        class_conds: Optional[Dict[str, int]] = None,
        spatial_conds: Optional[Dict[str, int]] = None,
        drop_spatial: bool = False,
    ):
        """
        Args:
            - img_size: int or (h, w)
            - patch_size: p (tokens = h/p * w/p)
            - num_layers, dim, heads: transformer size (see SIT_PRESETS)
            - c: channels of the data being generated
            - final_dim: hidden channels inside the Depatchifier
            - scalar_conds: continuous whole-image conditions, e.g. ["y"]
            - class_conds: discrete whole-image conditions, e.g. {"region": 7}
            - spatial_conds: image conditions concatenated as extra input channels,
                             {key: channels}, e.g. {"x_cond": 1} for a healthy CT slice
            - drop_spatial: if True, spatial conditions are zeroed when drop_cond is True
                            (False = always kept, e.g. StrokeDiT's anatomy)
        """
        super().__init__()
        self.c = c
        self.spatial_conds = dict(spatial_conds or {})
        self.drop_spatial = drop_spatial
        H, W = _hw(img_size)

        # 0. Embedders: time + global conditions
        self.time_embedder = FourierEncoder(dim=dim)
        self.cond_embedder = GlobalConditionEmbedder(dim, list(scalar_conds or []), dict(class_conds or {}))

        # 1. Patchifier: data channels + spatial-condition channels
        c_in = c + sum(self.spatial_conds.values())
        self.patchifier = Patchifier(img_size=(H, W), patch_size=patch_size, c_in=c_in, dim=dim)

        # 2. DiT
        grid_size = (H // patch_size, W // patch_size)
        self.dit = DiffusionTransformer(depth=num_layers, grid_size=grid_size, dim=dim, heads=heads)

        # 3. Depatchifier: outputs only the data channels (the velocity)
        self.depatchifier = Depatchifier(img_size=(H, W), patch_size=patch_size, dim=dim,
                                         final_dim=final_dim, c_out=c)

    @property
    def cond_keys(self) -> List[str]:
        """Every key the model expects in `cond`."""
        return self.cond_embedder.keys + list(self.spatial_conds.keys())

    @property
    def has_droppable_cond(self) -> bool:
        """Whether classifier-free guidance does anything for this model."""
        return len(self.cond_embedder.keys) > 0 or (self.drop_spatial and len(self.spatial_conds) > 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Condition,
        drop_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            - x: b c h w
            - t: b   (or b 1 1 1)
            - cond: Condition
            - drop_cond: b (bool) or None
        Returns:
            - u_t^theta(x|cond): b c h w
        """
        missing = [k for k in self.cond_keys if k not in cond]
        assert not missing, f"cond is missing keys {missing}; model expects {self.cond_keys}"

        # 1. Embed time and global conditions
        t_emb = self.time_embedder(t.reshape(-1))                  # b d
        c_emb = t_emb + self.cond_embedder(cond, drop_cond, t_emb)  # b d

        # 2. Stack spatial conditions onto the input as extra channels
        if self.spatial_conds:
            extras = []
            for k in self.spatial_conds:
                s = cond[k].to(x.dtype)
                if self.drop_spatial and drop_cond is not None:
                    s = torch.where(drop_cond.reshape(-1, 1, 1, 1), torch.zeros_like(s), s)
                extras.append(s)
            x = torch.cat([x] + extras, dim=1)                      # b c_in h w

        # 3. Patchify -> DiT -> Depatchify
        x = self.patchifier(x)        # b n d
        x = self.dit(x, c_emb)        # b n d
        return self.depatchifier(x)   # b c h w


def count_params(model: nn.Module) -> float:
    """Millions of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# ---------------------------------------------------------------------------
# Self-test:  python model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from simulators import CFGVectorFieldODE, EulerSimulator, make_time_grid

    torch.manual_seed(0)
    b = 2

    # 1. GameDiT at full size: SiT-S/8 on 256x256 heightmaps, relief condition
    layers, dim, heads = SIT_PRESETS["S"]
    game = DiffusionTransformerFlowModel(img_size=256, patch_size=8, num_layers=layers,
                                         dim=dim, heads=heads, c=1, scalar_conds=["y"])
    x, t, cond = torch.randn(b, 1, 256, 256), torch.rand(b), {"y": torch.rand(b)}
    out = game(x, t, cond)
    assert out.shape == x.shape
    # zero-init Depatchifier => exactly zero velocity before training
    assert torch.all(out == 0), "fresh model should predict zero velocity"
    print(f"GameDiT SiT-S/8: {count_params(game):.1f}M params, output {tuple(out.shape)}, zero at init")

    # Small models from here on (fast on CPU)
    small = dict(img_size=64, patch_size=8, num_layers=2, dim=64, heads=4)

    def _unzero(m):
        """Give the final conv random weights so conditioning effects are visible."""
        nn.init.normal_(m.depatchifier.net[-1].weight, std=0.1)
        for layer in m.dit.layers:
            nn.init.normal_(layer.ada_ln[1].weight, std=0.1)
        return m

    # 2. Dropping the condition: output stops depending on y
    g = _unzero(DiffusionTransformerFlowModel(**small, c=1, scalar_conds=["y"]))
    x, t = torch.randn(b, 1, 64, 64), torch.rand(b)
    y_a, y_b = {"y": torch.zeros(b)}, {"y": torch.ones(b)}
    assert not torch.allclose(g(x, t, y_a), g(x, t, y_b)), "y should change the output"
    drop = torch.ones(b, dtype=torch.bool)
    assert torch.allclose(g(x, t, y_a, drop), g(x, t, y_b, drop)), "dropped y should be ignored"
    print("scalar condition + drop OK")

    # 3. StrokeDiT: healthy slice as a spatial condition (always kept by default)
    s = _unzero(DiffusionTransformerFlowModel(**small, c=1, spatial_conds={"x_cond": 1}))
    healthy_a, healthy_b = torch.randn(b, 1, 64, 64), torch.randn(b, 1, 64, 64)
    o1 = s(x, t, {"x_cond": healthy_a})
    o2 = s(x, t, {"x_cond": healthy_b})
    assert o1.shape == x.shape and not torch.allclose(o1, o2), "spatial condition should matter"
    assert not s.has_droppable_cond
    print(f"StrokeDiT spatial condition OK ({count_params(s):.2f}M params)")

    # 4. Unconditional, non-square, multi-channel
    u = DiffusionTransformerFlowModel(img_size=(64, 32), patch_size=8, num_layers=2, dim=64, heads=4, c=2)
    assert u(torch.randn(b, 2, 64, 32), t, {}).shape == (b, 2, 64, 32)
    print("unconditional / non-square / 2-channel OK")

    # 5. Missing condition gives a clear error
    try:
        g(x, t, {})
        raise RuntimeError("should have failed")
    except AssertionError as e:
        print("missing-key check OK:", e)

    # 6. Gradients flow, and the model plugs into the sampler
    loss = g(x, t, y_b).pow(2).mean()
    loss.backward()
    assert g.patchifier.net[0].weight.grad is not None, "no gradient reached the patchifier"
    assert g.cond_embedder.scalar_embedders["y"].net[0].weight.grad is not None, "no gradient reached the condition embedder"
    ts = make_time_grid(b, 5, x.device)
    sample = EulerSimulator(CFGVectorFieldODE(g, guidance_scale=3.0)).simulate(
        torch.randn(b, 1, 64, 64), ts, use_tqdm=False, cond=y_b)
    assert sample.shape == (b, 1, 64, 64) and torch.isfinite(sample).all()
    print("backward + CFG sampling OK")

    # 7. QK-norm bounds the attention scores, however large the inputs get
    attn = MHA(dim=64, heads=4)
    for scale in (1.0, 1e3):
        xx = torch.randn(2, 16, 64) * scale
        q, k, _ = rearrange(attn.qkv(xx), "b n (three h dh) -> three b h n dh", three=3, h=4).unbind(0)
        logits = attn.q_norm(q) @ attn.k_norm(k).transpose(-1, -2) / math.sqrt(attn.head_dim)
        assert logits.abs().max() <= math.sqrt(attn.head_dim) + 1e-3, "QK-norm should bound the scores"
    raw = (q @ k.transpose(-1, -2) / math.sqrt(attn.head_dim)).abs().max().item()
    print(f"QK-norm OK: scores <= {math.sqrt(attn.head_dim):.1f} (without it, this input gives {raw:.0f})")
    assert attn(torch.randn(2, 16, 64)).shape == (2, 16, 64)
    print("model.py OK")
