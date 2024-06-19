import torch
import time
import numpy as np
from main import flops_counter

# Borrowed from https://github.com/pytorch-labs/gpt-fast/blob/db7b273ab86b75358bd3b014f1f022a19aba4797/generate.py#L16-L18
torch.set_float32_matmul_precision('high')
import torch._dynamo.config
import torch._inductor.config
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch.backends.cudnn.enabled = True
    

SIZE = 500

x = torch.randn(SIZE, SIZE).to(device='cuda')
y = torch.randn(SIZE, SIZE).to(device='cuda')

def func(x, y):
    return torch.einsum("ij,jk->ik", x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)


func = torch.compile(func, fullgraph=True, mode='max-autotune')
func_flops(func, x, y)

# compiled = jax.jit(func).lower(x,y).compile()
# flops = compiled.cost_analysis()[0]['flops']
# bytes = compiled.cost_analysis()[0]['bytes accessed']
# wall_time = benchmark(func, x, y)
# print("Empirical GFLOPS/s", flops/wall_time/1024/1024/1024)
# print("Empirical FLOPS", 2*100*100*100)
# print("Empirical Bytes", bytes)


