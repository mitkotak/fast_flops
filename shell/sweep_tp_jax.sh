for lmax in {1..10}
    do
    python run_profiler.py examples/e3nn/test_tp_jax.py output_$lmax.csv --lmax $lmax && python postprocess.py output_$lmax.csv
    #python examples/e3nn/test_tp_jax.py --lmax $lmax
    done
