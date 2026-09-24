"""
Diffusion Transformer (DiT) (well, more precisely it is a SiT Model) for the GameDiT application.
"""
import math
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from paths import Condition
from simulators import ConditionalVectorField

ImgSize = Union[int, Tuple[int, int]]

def _hw(img_size: ImgSize) -> Tuple[int, int]:
    return (img_size, img_size) if isinstance(img_size, int) else img_size

# ==========================
# Building Blocks
# ==========================

class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: Type[nn.Module] = nn.SiLU, final_init: bool=False):
        super().__init__()
        mlp = []
        for i in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mlp.append(activation())
        self.net = nn.Sequential(*mlp)

        if final_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FourierEncoder(nn.Module):    # to encode Time
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "FourierEncoder dimension must be even"
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_i = 2 * math.pi * self.weights * x.unsqueeze(-1)  # (B, half_dim)
        emb = torch.cat([torch.sin(f_i), torch.cos(f_i)], dim=-1)  # (B, dim)
        return emb * math.sqrt(2.0)

class Patchifier(nn.Module):    # to patchify the image into tokens: kinda like tokenizer
    def __init__(self, img_size: ImgSize, patch_size: int, c_in: int, dim:int):
        super().__init__()
        H, W = _hw(img_size)
        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.dim = dim
        self.net = nn.Sequential(
            # convolution: to cut into patches + linear projection
            nn.Conv2d(c_in, dim, kernel_size=patch_size, stride=patch_size),
            # patchify
            Rearrange('b c h w -> b (h w) c')
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MHA(nn.Module):
    """
    The MULTI-HEADED ATTENTION.
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=0.0, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        return out
    
def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor x with scale and bias.
    """
    return x * (1 + scale) + bias

class DiffusionTransformerLayer(nn.Module):
    """
    A single layer of the Diffusion Transformer.
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, dim * 6)
        )
        # init Adaptive Layer Norm parameters to zero
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

        self.attn = MHA(dim, num_heads)
        self.ff = MLP([dim, dim * 4, dim])
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = rearrange(self.ada_ln(c), "b d -> b 1 d")
        attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim =-1)
        x = x + attn_gate * self.attn(modulate(self.norm1(x), attn_scale, attn_bias))
        x = x + ff_gate * self.ff(modulate(self.norm2(x), ff_scale, ff_bias))
        return x

# =========================
# sin-cos position embedding
# =========================

def get_2d_sincos_pos_embed(dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    Half the channels encode the row, half the column.
    returns:
        pos_embed: (grid_h * grid_w) d
    """
    assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D sin-cos embedding"
    yy, xx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij"
    )

    def _1d(pos: torch.Tensor, d:int) -> torch.Tensor:
        omega = 1.0 / (10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d // 2)))
        angles = pos.reshape(-1,1) * omega.reshape(1,-1)
        return torch.cat([
            torch.sin(angles), 
            torch.cos(angles)
        ], dim=1)
    return torch.cat([
        _1d(yy, dim // 2),
        _1d(xx, dim // 2)
    ], dim=1)

class DiffusionTransformer(nn.Module):
    """
    The Diffusion Transformer model.
    """
    def __init__(
        self,
        depth: int,
        grid_size: Tuple[int, int],
        dim: int,
        **layer_kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([DiffusionTransformerLayer(dim, **layer_kwargs) for _ in range(depth)])
        # fixed, not learned sin-cos positional embedding
        pos = get_2d_sincos_pos_embed(dim, *grid_size)
        self.register_buffer("pos_embed", pos, persistent=False)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x, c)
        return x

class Depatchifier(nn.Module):
    """
    Depatchifier to reconstruct the image from patches.
    """
    def __init__(self, img_size: ImgSize, patch_size: int, c_out: int, dim:int):
        super().__init__()
        H, W = _hw(img_size)
        assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"
        assert dim % (patch_size ** 2) == 0, "Dimension must be divisible by patch area"
        self.patch_size = patch_size
        self.dim = dim
        patch_channels = dim // (patch_size ** 2)
        self.net = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            Rearrange(
                'b (h w) (f ph pw) -> b f (h ph) (w pw)',
                h = H // patch_size,
                ph = patch_size,
                pw = patch_size,
            ),
            nn.Conv2d(patch_channels, c_out, kernel_size=1, bias=False)
        )
        # model starts by predicting zero velocity
        nn.init.zeros_(self.net[-1].weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =================================
# Pluggable Conditional Embedder
# =================================

class GlobalConditionEmbedder(nn.Module):
    """
    Turns the whole-image condition into one vector that is added to the time
    embedding.
    Each condition has its own learned null, used wherever the condition is dropped.
    """
    def __init__(self, dim: int, scalar_conds: List[str], class_conds: Dict[str, int]):
        super().__init__()
        self.dim = dim
        self.scalar_embedders = nn.ModuleDict({k: MLP([1, dim, dim]) for k in scalar_conds})
        self.scalar_nulls = nn.ParameterDict({k: nn.Parameter(torch.zeros(dim)) for k in scalar_conds})
        
        self.class_embedders = nn.ModuleDict({k: nn.Embedding(n+1, dim) for k, n in class_conds.items()})
        self.class_nulls_idx = dict(class_conds)
    
    @property
    def keys(self) -> List[str]:
        return list(self.scalar_embedders.keys()) + list(self.class_embedders.keys())
    
    def forward(self, cond: Condition, drop_cond: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
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

# ===============================================================
# THE FULL MODEL: Diffusion Transformer (DiT)
# ================================================================
SIT_PRESETS = {
    'S' : (12, 384, 6),
    'M' : (12, 768, 12),
    'L' : (24, 1024, 16)
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
        drop_spatial: bool = False
    ):
        super().__init__()
        self.c = c
        self.spatial_conds = dict(spatial_conds or {})
        self.drop_spatial = drop_spatial
        H, W = _hw(img_size)

        # 0. Embedders: Time + Global Condition
        self.time_embedder = FourierEncoder(dim)
        self.cond_embedder = GlobalConditionEmbedder(dim, list(scalar_conds or []), dict(class_conds or {}))

        # 1. Patchifier: Image -> Patches
        c_in = c + sum(self.spatial_conds.values())
        self.patchifier = Patchifier(img_size=(H, W), patch_size=patch_size, c_in=c_in, dim=dim)

        # 2. Diffusion Transformer: The main model
        grid_size = (H // patch_size, W // patch_size)
        self.transformer = DiffusionTransformer(num_layers, grid_size, dim, num_heads=heads)

        # 3. Depatchifier: Patches -> Image
        self.depatchifier = Depatchifier(img_size=(H, W), patch_size=patch_size, c_out=c, dim=dim)

    @property
    def cond_keys(self) -> List[str]:
        return self.cond_embedder.keys + list(self.spatial_conds.keys())
    
    @property
    def has_droppable_cond(self):
        return len(self.cond_embedder.keys) > 0 or (self.drop_spatial and len(self.spatial_conds) > 0)

    def forward(
        self, 
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Condition,
        drop_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Input image tensor of shape (B, C, H, W)
            t: Time tensor of shape (B,)
            cond: Condition dictionary
            drop_cond: Optional tensor indicating which conditions to drop
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        missing = [k for k in self.cond_keys if k not in cond]
        assert not missing, f"cond is missing keys: {missing}; model expects {self.cond_keys}"
        # 0. Embed time and global condition
        t_emb = self.time_embedder(t)  # (B, dim)
        c_emb = t_emb + self.cond_embedder(cond, drop_cond, like=t_emb)  # (B, dim)

        # 1. Stack spatial conditions as extra channels
        if self.spatial_conds:
            extras = []
            for k in self.spatial_conds:
                s = cond[k].to(x.dtype)
                if self.drop_spatial and drop_cond is not None:
                    s = torch.where(drop_cond.reshape(-1, 1, 1, 1), torch.zeros_like(s), s)
                extras.append(s)
            x = torch.cat([x] + extras, dim=1)  

        # 2. Patchify -> DiT -> Depatchify
        x = self.patchifier(x)  # (B, N, dim)
        x = self.transformer(x, c_emb)  # (B, N, dim)
        x = self.depatchifier(x)  # (B, C, H, W)
        return x

def count_params(model: nn.Module) -> float:
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e-6  # in millions

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
        for layer in m.transformer.layers:
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
    assert sample[:, -1].shape == (b, 1, 64, 64) and torch.isfinite(sample).all()
    print("backward + CFG sampling OK")
    print("model.py OK")
