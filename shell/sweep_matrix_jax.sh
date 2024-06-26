for lmax in 1000 5000 10000 50000 100000 500000 1000000
    do
    echo $size
    python run_profiler.py examples/matmul/test_matmul_jax.py output_mamtul_$size.csv --size $size && python postprocess.py output_matmul_$size.csv
    python examples/matmul/test_matmul_jax.py
    done
