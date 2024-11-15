#!/bin/bash

# Delete all files ending with .out
find . -type f -name "*.out" -exec rm -f {} \;

# Move all files containing "bonddim" in the file name to the "data" folder
mkdir -p data
find . -type f -name "*bonddim*" -exec mv {} data/ \;

echo "All .out files deleted and files containing 'bonddim' moved to the 'data' folder."
