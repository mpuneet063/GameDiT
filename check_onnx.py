"""
Sanity check for export_onnx.py
"""
import torch
import numpy as np
import onnxruntime as ort
import os
import time
import argparse

from sample import load_model, generate
from export_onnx import HeunStepCFG
from train import save_grid
from data import visualize

def main():
    parser = argparse.ArgumentParser(description='Generating terrain using GameDiT')
    parser.add_argument("--ckpt", required=True, help="path to latest.pt")
    parser.add_argument("--out", default="onnx_out")
    parser.add_argument("--onnx", default="onnx_out/gamedit_step.onnx")
    parser.add_argument("--seed", type = int, default=16)
    parser.add_argument("--y", type = float, default=0.7)
    parser.add_argument("--w", type = float, default=2.0)
    parser.add_argument("--steps", type = int, default = 50)
    args = parser.parse_args()

    model, cfg, step, img_size = load_model(args.ckpt, device='cpu')
    os.makedirs(args.out, exist_ok=True)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    def run_ort(x,t,h,y,w):
        return sess.run(["x_next", "x1_hat"], {"x":x, "t": t, "h": h, "y": y, "w": w})

    # --- Check 1: one step ---
    x = torch.randn(1,1,img_size, img_size)
    t, h, yt, wt = (torch.tensor([v], dtype=torch.float32) for v in (0.3, 0.02, args.y, args.w))

    with torch.no_grad():
        pt_next, pt_hat = HeunStepCFG(model).eval()(x,t,h,yt,wt)
    
    ort_next, ort_hat = run_ort(x.numpy(), t.numpy(), h.numpy(), yt.numpy(), wt.numpy())
    print("x_next max diff:", np.abs(pt_next.numpy() - ort_next).max())
    print("x1_hat max diff:", np.abs(pt_hat.numpy() - ort_hat).max())

    #  --- Check 2: full generation, same noise ---
    ref = generate(model, 1, args.y, args.w, args.steps, args.seed, device='cpu', img_size=img_size)
    x0 = torch.randn(1,1,img_size, img_size, generator=torch.Generator().manual_seed(args.seed))
    ts = torch.linspace(0,1,args.steps+1)
    x = x0.numpy()
    previews = []
    y_np = np.array([args.y], dtype=np.float32)
    w_np = np.array([args.w], dtype=np.float32)
    start = time.time()
    for i in range(args.steps):
        t = ts[i:i+1].numpy()
        h = (ts[i+1:i+2] - ts[i:i+1]).numpy()
        x, x1_hat = run_ort(x, t, h, y_np, w_np)
        if i % 10 == 0 or i == args.steps - 1:
            previews.append(x1_hat)
    end = time.time()
    print(f"Total time taken: {end-start}")
    print(f"Time taken per step: {(end-start)/args.steps}")

    print("maximum absolute difference between ref and final x", np.abs(ref.numpy() - x).max())

    #  Visuals
    save_grid(torch.cat([ref, torch.tensor(x)]), os.path.join(args.out, "parity.png"), visualize, nrow=2) # should look identical
    save_grid(torch.cat([torch.from_numpy(p) for p in previews]), os.path.join(args.out, "x1_hat_strip.png"), visualize, nrow = len(previews))   # what the player will see

if __name__ == "__main__":
    main()