for lmax in {6..10}
    do
    python run_profiler.py examples/e3nn_sparse/test_tp_sparse_jax.py output_tp_sparse_$lmax.csv --lmax $lmax && python postprocess.py output_tp_sparse_$lmax.csv
    #python examples/e3nn/test_tp_jax.py --lmax $lmax
    done
