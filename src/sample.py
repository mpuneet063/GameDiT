"""
this script is to sample images/heightmap from the Diffusion model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
import contextlib

from model import DiffusionTransformerFlowModel, SIT_PRESETS
from simulators import CFGVectorFieldODE, EulerSimulator, HeunSimulator, make_time_grid
from train import save_grid
from data import visualize, y_to_relief, random_dihedral
from PIL import Image

def load_model(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location='cpu')
    cfg = state['config']
    mcfg = cfg['model'].copy()

    if "preset" in mcfg:
        preset = mcfg.pop('preset')
        num_layers, dim, heads = SIT_PRESETS[preset]
        mcfg.setdefault("num_layers", num_layers)
        mcfg.setdefault("dim", dim)
        mcfg.setdefault("heads", heads)     
    model = DiffusionTransformerFlowModel(**mcfg)
    model.load_state_dict(state['ema'])
    model.to(device).eval()
    img_size = mcfg.get("img_size", 256)     # (H = W = 256)
    return model, cfg, state['step'], img_size

def generate(model, n,y,w,steps, seed, device, img_size):
    gen = torch.Generator().manual_seed(seed)
    H = W = img_size
    x0 = torch.randn(n,1,H,W, generator=gen).to(device)
    cond = {'y': torch.full((n,), float(y), device=device)}
    ode = CFGVectorFieldODE(model, guidance_scale=w)
    sim = HeunSimulator(ode)
    ts = make_time_grid(n, steps, device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
    with torch.no_grad(), amp:
        x1 = sim.simulate(x0, ts, cond=cond)
    return x1.float()

def cmd_grid(model, device, seed, out_dir, img_size, y=0.7, n=8, ws = [1.0, 1.5, 2.0, 3.0]):
    for steps in [100]:
        rows = [generate(model, n, y, w, steps, seed, device, img_size) for w in ws]
        path = os.path.join(out_dir, f"grid_steps{steps}.png")
        save_grid(torch.cat(rows), path, visualize, nrow=n)
        print(f"{path}: rows top->bottom are w= {list(ws)}")

def cmd_relief(model, device, seed, out_dir, img_size, w=2.0, steps=50, n=8, ys=[0.1, 0.3, 0.5, 0.7, 0.9]):
    rows = [generate(model, n, y, w, steps, seed, device, img_size) for y in ys]
    path = os.path.join(out_dir, "relief_sweep.png")
    save_grid(torch.cat(rows), path, visualize, nrow=n)
    print(f"{path}: rows top-> bottom are y = {list(ys)}")

def cmd_export(model, device, seed, out_dir, img_size, y, w, steps, n=16, size=1009, smooth=0.0):
    export_dir = os.path.join(out_dir, 'export')
    os.makedirs(export_dir, exist_ok=True)
    x = generate(model, n, y, w, steps, seed, device, img_size)
    save_grid(x, os.path.join(export_dir, "candidates.png"), visualize)
    relief_m = float(y_to_relief(y))

    for i in range(n):
        h = x[i:i+1]
        h = (h+1)/2*relief_m
        h = F.interpolate(h, (size, size), mode='bicubic', align_corners=True)
        if smooth > 0:
            h = gaussian_blur(h, sigma=smooth)  # gaussian_blur is a helper

        h = h[0,0].cpu().numpy()
        lo, hi = float(h.min()), float(h.max())

        u16 = np.round((h-lo)/(hi-lo) * 65535).astype(np.uint16)

        Image.fromarray(u16).save(os.path.join(export_dir, f"terrain_{i:02d}.png"))
        
        extent_m = img_size * 60
        mpp = extent_m / (size - 1)
        meta = {
            'file': f'terrain_{i:02d}.png',
            "y": y,
            "w": w,
            "steps": steps,
            "seed": seed,
            "height_range_m": float(hi - lo),
            "size_px" : size,
            "metres_per_pixel": mpp,
            "ue_scale_xy": mpp*100,
            "ue_scale_z": (hi-lo)*100 / 512
        }
        with open(os.path.join(export_dir, f'terrain_{i:02d}.json'), 'w') as f:
            json.dump(meta, f, indent=2)
    print(f"Saved {n} heightmaps to {export_dir}; pick favourites from candidates.png by index")

def gaussian_blur(h, sigma):
    r = math.ceil(3 * sigma)                                                # fix 5
    xs = torch.arange(-r, r + 1, dtype=h.dtype, device=h.device)            # fix 6
    k = torch.exp(-xs ** 2 / (2 * sigma ** 2))
    k = k / k.sum()

    h = F.pad(h, (r, r, 0, 0), mode="reflect")                              # fix 8: pad left/right
    h = F.conv2d(h, k.view(1, 1, 1, -1))                                    # fix 7, 9: horizontal
    h = F.pad(h, (0, 0, r, r), mode="reflect")                              # pad top/bottom
    h = F.conv2d(h, k.view(1, 1, -1, 1))                                    # vertical
    return h

def nearest(queries, bank, device, chunk=1024):
    """
    For each query, the closest crop in `bank`, checking all 8 flips/rotations
    of the query (training used dihedral augmentation, so a rotated copy is still a copy).
    Args:
        - queries: q h w
        - bank:    n h w
    Returns:
        - rmse: q   (distance to the nearest crop, in [-1, 1] heightmap units)
        - idx:  q   (index of that crop in bank)
    """
    variants = []
    for flip in (False, True):
        qf = queries.flip(-1) if flip else queries
        for k in range(4):
            variants.append(torch.rot90(qf, k, dims=(-2, -1)))
    v = torch.stack(variants, dim=1)                                  # q 8 h w
    q, nv = v.shape[:2]
    A = v.reshape(q * nv, -1).float().to(device)                      # (q*8) d
    D = A.shape[1]
    a2 = (A * A).sum(dim=1)

    best_d = torch.full((q * nv,), float("inf"), device=device)
    best_i = torch.zeros(q * nv, dtype=torch.long, device=device)
    for start in range(0, bank.shape[0], chunk):
        B = bank[start:start + chunk].reshape(-1, D).float().to(device)  # c d
        b2 = (B * B).sum(dim=1)
        d2 = a2[:, None] + b2[None, :] - 2.0 * (A @ B.T)               # squared distances
        d2_min, j = d2.min(dim=1)
        better = d2_min < best_d
        best_d = torch.where(better, d2_min, best_d)
        best_i = torch.where(better, j + start, best_i)

    best_d = best_d.reshape(q, nv)
    best_i = best_i.reshape(q, nv)
    v_best = best_d.argmin(dim=1)                                     # best orientation per query
    rows = torch.arange(q, device=device)
    rmse = torch.sqrt(best_d[rows, v_best].clamp(min=0) / D)
    return rmse.cpu(), best_i[rows, v_best].cpu()


def cmd_nn(model, device, seed, out_dir, img_size, data_dir, w, steps, n=32):
    train = torch.from_numpy(np.load(os.path.join(data_dir, "train.npy")))   # N h w float16
    val = torch.from_numpy(np.load(os.path.join(data_dir, "val.npy")))
    meta = np.load(os.path.join(data_dir, "train_meta.npz"))

    rng = np.random.default_rng(seed)
    ys = rng.choice(meta["y"], size=n)                                # realistic mix of reliefs

    gen = torch.cat([
        generate(model, 1, float(yv), w, steps, seed + i, device, img_size)
        for i, yv in enumerate(ys)
    ])[:, 0].cpu()                                                    # n h w

    # same per-crop normalisation as the training data
    lo = gen.amin(dim=(1, 2), keepdim=True)
    hi = gen.amax(dim=(1, 2), keepdim=True)
    gen = (gen - lo) / (hi - lo + 1e-8) * 2 - 1

    val_idx = rng.choice(len(val), size=min(n, len(val)), replace=False)
    val_q = val[torch.from_numpy(val_idx)].float()

    d_gen, idx = nearest(gen, train, device)
    d_val, _ = nearest(val_q, train, device)

    med_gen, med_val = d_gen.median().item(), d_val.median().item()
    ratio = med_gen / med_val
    print(f"median nearest-neighbour RMSE  generated: {med_gen:.4f}   real-unseen (val): {med_val:.4f}")
    print(f"ratio generated / val = {ratio:.2f}  (about 1 or more = new terrain; < 0.5 = possible copying)")
    if ratio < 0.5:
        print("WARNING: possible memorisation - inspect nn_check.png")

    pairs = torch.stack([gen, train[idx].float()], dim=1).reshape(-1, 1, *gen.shape[1:])  # g0, nn0, g1, nn1, ...
    path = os.path.join(out_dir, "nn_check.png")
    save_grid(pairs, path, visualize, nrow=8)
    print(f"{path}: each row = 4 pairs of (generated | nearest training crop)")

    with open(os.path.join(out_dir, "nn_check.json"), "w") as f:
        json.dump({
            "median_rmse_generated": med_gen,
            "median_rmse_val": med_val,
            "ratio": ratio,
            "per_sample": [{"y": float(yv), "rmse": float(d), "nearest_train_idx": int(j)}
                           for yv, d, j in zip(ys, d_gen, idx)],
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Sample terrain from a trained GameDiT checkpoint")
    parser.add_argument("command", choices=["grid", "relief", "export", "nn"])
    parser.add_argument("--ckpt", required=True, help="path to latest.pt")
    parser.add_argument("--out", default="samples_out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--y", type=float, default=0.7, help="relief condition in [0, 1]")
    parser.add_argument("--w", type=float, default=2.0, help="guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="sampler steps")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--size", type=int, default=505, help="export size (UE: 505, 1009, ...)")
    parser.add_argument("--smooth", type=float, default=0.0, help="export blur sigma in pixels")
    parser.add_argument("--data_dir", default="data/processed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step, img_size = load_model(args.ckpt, device)
    print(f"Loaded EMA weights from step {step} on {device}")
    os.makedirs(args.out, exist_ok=True)

    if args.command == "grid":
        cmd_grid(model, device, args.seed, args.out, img_size, y=args.y)
    elif args.command == "relief":
        cmd_relief(model, device, args.seed, args.out, img_size, w=args.w, steps=args.steps)
    elif args.command == "export":
        cmd_export(model, device, args.seed, args.out, img_size,
                   args.y, args.w, args.steps, n=args.n, size=args.size, smooth=args.smooth)
    elif args.command == "nn":
        cmd_nn(model, device, args.seed, args.out, img_size,
               args.data_dir, args.w, args.steps, n=args.n)


if __name__ == "__main__":
    main()