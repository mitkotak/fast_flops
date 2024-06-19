import torch
import time
import numpy as np
from fantastic_flops import flops_counter

# Borrowed from https://github.com/pytorch-labs/gpt-fast/blob/db7b273ab86b75358bd3b014f1f022a19aba4797/generate.py#L16-L18
torch.set_float32_matmul_precision('high')
import torch._dynamo.config
import torch._inductor.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch.backends.cudnn.enabled = True
    

SIZE = 500

x = torch.randn(SIZE).to(device='cuda')
y = torch.randn(SIZE).to(device='cuda')

def func(x, y):
    return torch.einsum("i,j->", x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)


func = torch.compile(func, fullgraph=True, mode='max-autotune')
func_flops(func, x, y)
