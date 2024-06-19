import subprocess
import sys

# Time
metrics = "sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,"

# DP
metrics += "sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"

# SP
metrics += "sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"

# HP
metrics += "sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"

# Tensor Core
metrics += "sm__inst_executed_pipe_tensor.sum,"

# DRAM, L2 and L1
metrics += "dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum"

# Check if an input file is provided
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = "output.csv"
profile_str = f"ncu --nvtx --nvtx-include \"profile\" --metrics {metrics} --csv --print-units base"

try:
    subprocess.run(f"{profile_str} python {input_file} > {output_file}", shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    sys.exit(1)