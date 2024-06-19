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

SIZE = 500
LMAX = 4

irreps = o3.Irreps.spherical_harmonics(LMAX)
x = irreps.randn(SIZE, -1).to(device='cuda')
y = irreps.randn(SIZE, -1).to(device='cuda')

tp = o3.experimental.FullTensorProductv2(irreps, irreps).to(device='cuda')

def func(x, y):
    return tp(x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)

func = torch.compile(func, mode="max-autotune", fullgraph=True)
func_flops(func, x, y)