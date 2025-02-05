#!/bin/bash

lattice_sizes=(3)                               # Array of lattice sizes
bond_dims=(10)                                  # Array of bond dimensions
seeds=(123 124 125 126 127 128 129 130 131 132) # Array of (10) random seeds
num_experiments=10000                           # Runs per each random seed
error_model="Bitflip"                           # The error model
bias_probs=(1e-8 1e-4 1e-3 1e-2 1e-1)           # Array of decoder bias probabilities
tolerances=(1e-17 1e-12 1e-8 1e-4 1e-2 1e-1)    # Array of numerical tolerances for the MPS within the decoder
cuts=(1e-17 1e-12 1e-8 1e-6 1e-4 1e-3 1e-2)     # Array of SVD cut-offs for the MPS within the decoder
num_processes=16                                # The number of processes to use in parallel
silent=false                                    # Whether to suppress the output of the Python script

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
error_rates=(0.1)
# Iterate over combinations of the arguments and run the Python script
for seed in "${seeds[@]}"; do
    for lattice_size in "${lattice_sizes[@]}"; do
        for bond_dim in "${bond_dims[@]}"; do
            for error_rate in "${error_rates[@]}"; do
                for bias_prob in "${bias_probs[@]}"; do
                    for tolerance in "${tolerances[@]}"; do
                        for cut in "${cuts[@]}"; do
                            # Print current configuration
                            echo "Running for lattice size ${lattice_size}, bond dimension ${bond_dim}, error rate ${error_rate}, error model ${error_model}, and seed ${seed}."
                            # Run the Python script
                            poetry run python examples/decoding/quantum_surface.py --lattice_size ${lattice_size} --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance} --cut ${cut}
                        done
                    done
                done
            done
        done
    done
done

echo "All calculations are complete."
