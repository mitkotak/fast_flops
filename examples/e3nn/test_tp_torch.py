import argparse
import torch
# Borrowed from https://github.com/pytorch-labs/gpt-fast/blob/db7b273ab86b75358bd3b014f1f022a19aba4797/generate.py#L16-L18
torch.set_float32_matmul_precision('high')
import torch._dynamo.config
import torch._inductor.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
                
import time
from e3nn import o3
import numpy as np
from fantastic_flops import flops_counter

parser = argparse.ArgumentParser(description='Tensor product calculation')
parser.add_argument('--lmax', type=int, default=2, help='max_ell')
parser.add_argument('--batch', type=int, default=1, help='batch size')

args = parser.parse_args()
irreps = o3.Irreps.spherical_harmonics(args.lmax)

x = irreps.randn(args.batch, -1).to(device='cuda')
y = irreps.randn(args.batch, -1).to(device='cuda')

tp = o3.experimental.FullTensorProductv2(irreps, irreps).to(device='cuda')


@flops_counter
def func_flops(func, x, y):
    return func(x, y)

tp = torch.compile(tp, mode="max-autotune", fullgraph=True).to(device='cuda')
func_flops(tp, x, y)