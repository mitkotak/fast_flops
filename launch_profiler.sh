#!/bin/bash
# Input file
input_file=$1

# Create a temporary file for output
#temp_output=$(mktemp --suffix=.csv)
temp_output="test.csv"

# Run the profiler
python run_profiler.py "$input_file" "$temp_output"

# Run the postprocessing
python postprocess.py $temp_output

# Clean up the temporary file
rm -f "$temp_output"