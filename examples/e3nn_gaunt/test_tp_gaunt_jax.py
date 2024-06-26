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

def gaunt_tensor_product_grid(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    *,
    res_beta=90,
    res_alpha=89,
    filter_ir_out=None,
) -> e3nn.IrrepsArray:

    return gaunt_tensor_product_fixed_parity(
        input1,
        input2,
        p_val1=1,
        p_val2=1,
        res_beta=res_beta,
        res_alpha=res_alpha,
        filter_ir_out=filter_ir_out,
    )


def gaunt_tensor_product_fixed_parity(
    input1: e3nn.IrrepsArray,
    input2: e3nn.IrrepsArray,
    p_val1: int,
    p_val2: int,
    *,
    res_beta=90,
    res_alpha=89,
    quadrature="gausslegendre",
    filter_ir_out=None,
):
    input1, input2, leading_shape = _prepare_inputs(input1, input2)
    if filter_ir_out is None:
        filter_ir_out = e3nn.Irreps([f"{l}e" for l in range(input1.irreps.lmax + input2.irreps.lmax)])
        #filter_ir_out = e3nn.s2_irreps(
        #    input1.irreps.lmax + input2.irreps.lmax, p_val=p_val1 * p_val2
        #)
    filter_ir_out = _validate_filter_ir_out(filter_ir_out)

    return e3nn.from_s2grid(
        e3nn.to_s2grid(
            input1,
            res_beta=res_beta,
            res_alpha=res_alpha,
            quadrature="gausslegendre",
            fft=False,
            p_val=1,
            p_arg=1,
        )
        * e3nn.to_s2grid(
            input2,
            res_beta=res_beta,
            res_alpha=res_alpha,
            quadrature="gausslegendre",
            fft=False,
            p_val=1,
            p_arg=1,
        ),
        irreps=filter_ir_out,
        fft=False,
    )
input_irreps = e3nn.Irreps([f"{l}e" for l in range(args.lmax + 1)])
x = e3nn.normal(input_irreps, jax.random.PRNGKey(0), (args.batch, ))
y = e3nn.normal(input_irreps, jax.random.PRNGKey(1), (args.batch, ))

def func(x, y):
    return gaunt_tensor_product_grid(x, y)

@flops_counter
def func_flops(func, x, y):
    return func(x, y)

func = jax.jit(func)
func_flops(func, x, y)
