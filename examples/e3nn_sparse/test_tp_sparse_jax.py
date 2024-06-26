import argparse
import jax
import jax.numpy as jnp
import time
import e3nn_jax as e3nn
from e3nn_jax._src.tensor_products import _validate_filter_ir_out, _prepare_inputs
import numpy as np
from fantastic_flops import flops_counter

parser = argparse.ArgumentParser(description='Tensor product calculation')
parser.add_argument('--lmax', type=int, default=2, help='max_ell')
parser.add_argument('--batch', type=int, default=1, help='batch size')

args = parser.parse_args()

def tensor_product_sparse(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    filter_ir_out=None,
) -> e3nn.IrrepsArray:

    input1, input2, leading_shape = _prepare_inputs(input1, input2)
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    irreps_out = []
    chunks = []
    for (mul_1, ir_1), x1 in zip(input1.irreps, input1.chunks):
        for (mul_2, ir_2), x2 in zip(input2.irreps, input2.chunks):
            if x1 is None or x2 is None:
                continue

            x1_t = jnp.transpose(x1)
            x2_t = jnp.transpose(x2)

            for ir_out in ir_1 * ir_2:
                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                irreps_out.append((mul_1 * mul_2, ir_out))

                l1, l2, l3 = ir_1.l, ir_2.l, ir_out.l
                cg = e3nn.clebsch_gordan_basislib(l1, l2, l3)
                chunk = jnp.zeros((2 * l3 + 1, x1.shape[-2], x2.shape[-2]) + leading_shape)
                for m3 in range(-l3, l3 + 1):
                    sum = 0
                    for m1 in range(-l1, l1 + 1):
                        for m2 in set([m3 - m1, m3 + m1, -m3 + m1, -m3 - m1]):
                            if m2 < -l2 or m2 > l2:
                                continue

                            path = jnp.einsum(
                                "u...,v... -> uv...",
                                x1_t[l1 + m1, ...],
                                x2_t[l2 + m2, ...],
                            )
                            cg_coeff = cg[l1 + m1, l2 + m2, l3 + m3]
                            cg_coeff *= jnp.sqrt(ir_1.dim * ir_2.dim)
                            path *= cg_coeff
                            sum += path
                    chunk = chunk.at[l3 + m3].set(sum)

                chunk = jnp.transpose(chunk)
                chunk = jnp.reshape(
                    chunk, chunk.shape[:-3] + (mul_1 * mul_2, ir_out.dim)
                )
                chunks.append(chunk)

    output = e3nn.from_chunks(irreps_out, chunks, leading_shape, input1.dtype)
    output = output.sort()
    return output

input_irreps = e3nn.Irreps([f"{l}e" for l in range(args.lmax + 1)])
x = e3nn.normal(input_irreps, jax.random.PRNGKey(0), (args.batch, ))
y = e3nn.normal(input_irreps, jax.random.PRNGKey(1), (args.batch, ))

def func(x, y):
    return tensor_product_sparse(x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)

func = jax.jit(func)
func_flops(func, x, y)
