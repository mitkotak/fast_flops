import argparse
import jax
import jax.numpy as jnp
import time
import numpy as np
from fantastic_flops import flops_counter

parser = argparse.ArgumentParser(description='Matrix Multiplication')
parser.add_argument('--size', type=int, default=1000, help='matrix size')

args = parser.parse_args()

SIZE = args.size

x = jax.random.normal(jax.random.PRNGKey(0), (SIZE, SIZE), dtype=jnp.float32)
y = jax.random.normal(jax.random.PRNGKey(1), (SIZE, SIZE), dtype=jnp.float32)

def func(x, y):
    return jnp.einsum("ij,jk->ik", x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)

func = jax.jit(func)
func_flops(func, x, y)

print("Analytical GFLOPS:", 2*SIZE*SIZE*SIZE/1024/1024/1024)
BYTES = SIZE * 4
print("Analytical GB:", 3*BYTES*BYTES/1024/1024/1024)
