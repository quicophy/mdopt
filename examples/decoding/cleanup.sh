#!/bin/bash

# Delete all files ending with .out
find . -type f -name "*.out" -exec rm -f {} \;

# Move all files ending with .npy to the "data" folder
mkdir -p data
find . -type f -name "*.npy" -exec mv {} data/ \;

echo "All .out files deleted and .npy files moved to the 'data' folder."
