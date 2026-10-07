"""
This script is to make the trained GameDiT model reproducible in other DL frameworks using 
ONNX (Open Neural Network Exchange)
"""
import torch
import torch.nn as nn
import json
import os
import argparse
import onnx

from sample import load_model
from data import RELIEF_MAX_M, RELIEF_MIN_M

class HeunStepCFG(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.register_buffer('drop', torch.tensor([False,True]))

    def velocity(self, x, t, y, w):
        x2, t2, y2 = torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, y])
        u_cond, u_null = self.model(x2, t2, {"y": y2}, drop_cond=self.drop).chunk(2)

        w4 = w.view(1,1,1,1)
        return (1-w4) * u_null + w4 * u_cond

    def forward(self, x, t, h, y, w):
        h4, t4 = h.view(1,1,1,1), t.view(1,1,1,1)
        u1 = self.velocity(x,t,y,w)
        x_pred = x + h4 * u1
        u2 = self.velocity(x_pred, t+h, y, w)
        x_next = x + 0.5 * h4 * (u1+u2)
        x1_hat = x + (1 - t4) * u1      # model's current guess of the final terrain
        return x_next, x1_hat

class ManualRMSNorm(nn.Module):
    def __init__(self, rms: nn.RMSNorm) -> None:
        super().__init__()
        self.eps = rms.eps if rms.eps is not None else torch.finfo(torch.float32).eps
        self.weight = rms.weight

    def forward(self, x):
        out = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return out * self.weight if self.weight is not None else out
    
def swap_rmsnorm(module: nn.Module):
    for (name, child) in module.named_children():
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, ManualRMSNorm(child))
        else:
            swap_rmsnorm(child)

def main():
    parser = argparse.ArgumentParser(description='Generating terrain using GameDiT')
    parser.add_argument("--ckpt", required=True, help="path to latest.pt")
    parser.add_argument("--out", default="onnx_out")
    parser.add_argument("--opset", type = int, default=17)
    args = parser.parse_args()

    model, cfg, step, img_size = load_model(args.ckpt, device='cpu')
    swap_rmsnorm(model)
    wrapper = HeunStepCFG(model).eval()

    # dummy inputs
    x = torch.randn(1, 1, img_size, img_size)
    t, h, y, w = (torch.tensor([v], dtype=torch.float32) for v in (0.3, 0.02, 0.7, 2.0))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'gamedit_step.onnx')

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x,t,h,y,w),
            path,
            input_names=['x', 't', 'h', 'y', 'w'],
            output_names = ['x_next', 'x1_hat'],
            opset_version=args.opset,
            dynamo = False,
            do_constant_folding=True
        )

    
    onnx.checker.check_model(path)
    print(f"{os.path.getsize(path) / 1e6:.0f} MB")

    meta = {
        "ckpt_step": step, "img_size": img_size, "opset": args.opset,
        "inputs": {"x": [1, 1, img_size, img_size], "t": [1], "h": [1], "y": [1], "w": [1]},
        "outputs": ["x_next", "x1_hat"],
        "default_steps": 50, "default_w": 2.0,
        "relief_min_m": RELIEF_MIN_M, "relief_max_m": RELIEF_MAX_M,
        "metres_per_pixel": 60.0,
    }
    with open(os.path.join(args.out, "gamedit_step.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == '__main__':
    main()