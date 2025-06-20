#!/bin/bash

order_x=6                                       # Orders for polynomials to create BB codes
order_y=6
poly_a="1 + x + y"                              # The polynomials used to create BB codes
poly_b="1 + x**2 + y**2"
bond_dims=(30)                                  # Array of bond dimensions
seeds=(100 101 102 103 104 105 106 107 108 109) # Array of (10) random seeds
num_experiments=100                             # Runs per each random seed
error_model="Bitflip"                           # The error model
bias_probs=(1e-1)                               # Array of decoder bias probabilities
tolerances=(0)                                  # Array of numerical tolerances for the MPS within the decoder
cuts=(0)                                        # Array of SVD cut-offs for the MPS within the decoder
num_processes=16                                # The number of processes to use in parallel
silent=false                                    # Whether to suppress the output of the Python script

error_rates=()
start=0.01
end=0.07
step=0.01
current=$start
while (( $(echo "$current <= $end" | bc -l) ))
do
    error_rates+=($current)
    current=$(echo "$current + $step" | bc -l)
done
error_rates=(0.001 0.002 0.004 0.008 0.01)
# Iterate over combinations of the arguments and run the Python script
for seed in "${seeds[@]}"; do
    for bond_dim in "${bond_dims[@]}"; do
        for error_rate in "${error_rates[@]}"; do
            for bias_prob in "${bias_probs[@]}"; do
                for tolerance in "${tolerances[@]}"; do
                    for cut in "${cuts[@]}"; do
                        # Print current configuration
                        echo "Running for orders ${order_x} ${order_y}, bond dimension ${bond_dim}, error rate ${error_rate}, error model ${error_model}, and seed ${seed}."
                        # Run the Python script
                        poetry run python examples/decoding/quantum_bivariate_bicycle.py --order_x ${order_x} --order_y ${order_y} --poly_a "${poly_a}" --poly_b "${poly_b}" --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance} --cut ${cut}
                    done
                done
            done
        done
    done
done

echo "All calculations are complete."
