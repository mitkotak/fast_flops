import argparse
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

parser = argparse.ArgumentParser(description='Matrix Multiplication')
parser.add_argument('--size', type=int, default=1000, help='matrix size')

args = parser.parse_args()

SIZE = args.size

x = torch.randn(SIZE, SIZE).to(device='cuda')
y = torch.randn(SIZE, SIZE).to(device='cuda')

def func(x, y):
    return torch.einsum("ij,jk->ik", x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)


func = torch.compile(func, fullgraph=True)
func_flops(func, x, y)

print("Analytical GFLOPS:", 2*SIZE*SIZE*SIZE/1024/1024/1024)
BYTES = SIZE * 4
print("Analytical GB:", 3*BYTES*BYTES/1024/1024/1024)


