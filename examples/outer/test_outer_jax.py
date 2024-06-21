import time
import jax
import jax.numpy as jnp
import numpy as np
from fantastic_flops import flops_counter

SIZE = 500

x = jax.random.normal(jax.random.PRNGKey(0), (SIZE,))
y = jax.random.normal(jax.random.PRNGKey(1), (SIZE,))

def func(x, y):
    return jnp.einsum("i,j->", x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)


func = jax.jit(func)
func_flops(func, x, y)
