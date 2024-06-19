import argparse
import jax
import jax.numpy as jnp
import time
import e3nn_jax as e3nn
import numpy as np
from fantastic_flops import flops_counter

parser = argparse.ArgumentParser(description='Tensor product calculation')
parser.add_argument('--lmax', type=int, default=2, help='max_ell')
parser.add_argument('--batch', type=int, default=1, help='batch size')

args = parser.parse_args()

x = e3nn.normal(e3nn.s2_irreps(args.lmax), jax.random.PRNGKey(0), (args.batch, ))
y = e3nn.normal(e3nn.s2_irreps(args.lmax), jax.random.PRNGKey(1), (args.batch, ))

def func(x, y):
    return e3nn.tensor_product(x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)

func = jax.jit(func)
func_flops(func, x, y)
