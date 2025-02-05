#!/bin/bash

# Define arrays of lattice sizes, bond dimensions, error rates, and seeds
lattice_sizes=(3 5)
bond_dims=(10)
seeds=(
    123 # 10 random seeds 124 125 126 127 128 129 130 131 132
)
num_experiments=10000 # Per each random seed
error_model="Bitflip" # Error model used in the experiments
bias_prob=1e-1 # The decoder bias probability
tolerance=1e-6 # The numerical tolerance for the MPS within the decoder.
num_processes=16 # Number of processes to use in parallel
silent=true # Whether to suppress the output of the Python script

error_rates=()
start=0.01
end=0.21
step=0.03
current=$start
while (( $(echo "$current <= $end" | bc -l) ))
do
    error_rates+=($current)
    current=$(echo "$current + $step" | bc -l)
done

# Iterate over combinations of the arguments and run the Python script
for seed in "${seeds[@]}"; do
    for lattice_size in "${lattice_sizes[@]}"; do
        for bond_dim in "${bond_dims[@]}"; do
            for error_rate in "${error_rates[@]}"; do
                # Print current configuration
                echo "Running for lattice size ${lattice_size}, bond dimension ${bond_dim}, error rate ${error_rate}, error model ${error_model}, and seed ${seed}."
                # Run the Python script
                poetry run python examples/decoding/quantum_surface.py --lattice_size ${lattice_size} --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance}
            done
        done
    done
done

echo "All calculations are complete."
