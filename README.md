# Fast FLOPS

Ever needed to report FLOPs for your Torch/JAX code. We got you covered ! (More importantly, here's a survey on [why bunnies flop](https://rabbit.org/behavior/reading-your-rabbits-behavior/))

Borrowed from https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tree/roofline-hackathon-2020

<img src="./logo/fast_flops_tuong.jpeg" alt="bunny" width="300"/> 
(Image Credit: Tuong Phung)

## Workflow (Warning: Extremely clunky right now)

- Define your function that you wanna profile

    - JAX
        ```python
        SIZE = 500
        x = jax.random.normal(jax.random.PRNGKey(0), (SIZE, SIZE))
        y = jax.random.normal(jax.random.PRNGKey(1), (SIZE, SIZE))

        def func(x, y):
            return jnp.einsum("ij,jk->ik", x, y)
        ```

    - Torch
        ```python
        SIZE = 500
        x = torch.randn(SIZE, SIZE).to(device='cuda')
        y = torch.randn(SIZE, SIZE).to(device='cuda')

        def func(x, y):
            return torch.einsum("ij,jk->ik", x, y)
        ```

- Wrap the function in the `flops_counter` decorator
    
    ```python
    from fast_flops import func_flops

    @flops_counter
    def func_flops(func, x, y):
        return func(x, y)
    ```

- Let it run through JIT (Don't worry we have warmups to cleanse the JIT overhead) and execute !
    - JAX
        ```python
        func = jax.jit(func)
        func_flops(func, x, y)
        ```

    - Torch
        ```python
        func = torch.compile(func, fullgraph=True, mode='max-autotune')
        func_flops(func, x, y)
        ```

- The pipeline can be executed using
    ```bash
    python run_profiler.py examples/matmul/test_matmul_torch.py output_matmul.csv && python postprocess.py output_matmul.csv
    ```
  with the output looking something like

  ```bash
  Measured Time: 3.196823494576654e-05
  Measured GFLOP/s: 2353.1630879907675
  Measured FLOPS: 67108864.0
  ```

