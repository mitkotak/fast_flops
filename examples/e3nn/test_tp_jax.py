import jax
import jax.numpy as jnp
import time
import e3nn_jax as e3nn
import numpy as np
from fantastic_flops import flops_counter

SIZE = 500
LMAX = 4

x = e3nn.normal(e3nn.s2_irreps(LMAX), jax.random.PRNGKey(0), (SIZE, ))
y = e3nn.normal(e3nn.s2_irreps(LMAX), jax.random.PRNGKey(1), (SIZE, ))

def func(x, y):
    return e3nn.tensor_product(x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)

func = jax.jit(func)
func_flops(func, x, y)
