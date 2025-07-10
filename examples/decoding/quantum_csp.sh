#!/bin/bash

batches=({1..14})                                # Array of batches
nums_qubits=(30)                                 # Array of numbers of qubits
code_ids=({0..99})                               # Array of code IDs
bond_dims=(150)                                  # Array of bond dimensions
seeds=(123)                                      # Array of random seeds
num_experiments=5000                             # Runs per each random seed
error_model="Bitflip"                            # The error model
bias_probs=(1e-3)                                # Array of decoder bias probabilities
tolerances=(0)                                   # Array of numerical tolerances for the MPS within the decoder
cuts=(0)                                         # Array of SVD cut-offs for the MPS within the decoder
num_processes=16                                 # The number of processes to use in parallel
silent=false                                     # Whether to suppress the output of the Python script

error_rates=()
start=0.01
end=0.21
step=0.02
current=$start
while (( $(echo "$current <= $end" | bc -l) ))
do
    error_rates+=($current)
    current=$(echo "$current + $step" | bc -l)
done
error_rates=(0.0001 0.001 0.002 0.004 0.008 0.01)
# Iterate over combinations of the arguments and run the Python script
for seed in "${seeds[@]}"; do
    for num_qubits in "${nums_qubits[@]}"; do
        for batch in "${batches[@]}"; do
            for code_id in "${code_ids[@]}"; do
                for bond_dim in "${bond_dims[@]}"; do
                    for error_rate in "${error_rates[@]}"; do
                        for bias_prob in "${bias_probs[@]}"; do
                            for tolerance in "${tolerances[@]}"; do
                                for cut in "${cuts[@]}"; do
                                    # Print current configuration
                                    echo "Running for ${num_qubits} qubits, batch ${batch}, code id ${code_id}, bond dimension ${bond_dim}, error rate ${error_rate}, error model ${error_model}, and seed ${seed}."
                                    # Run the Python script
                                    poetry run python examples/decoding/quantum_csp.py --num_qubits ${num_qubits} --batch ${batch} --code_id ${code_id} --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance} --cut ${cut}
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All calculations are complete."
