import jax
import jax.numpy as jnp
import time
import numpy as np
from fast_flops import flops_counter

SIZE = 500

x = jax.random.normal(jax.random.PRNGKey(0), (SIZE, SIZE))
y = jax.random.normal(jax.random.PRNGKey(1), (SIZE, SIZE))

def func(x, y):
    return jnp.einsum("ij,jk->ik", x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)

func = jax.jit(func)
func_flops(func, x, y)
