#!/bin/bash

# Load the Python module
module load python/3.11.5

# Check if the virtual environment exists, if not, create and activate it
if [ ! -d "$HOME/envs/myenv" ]; then
    virtualenv --no-download "$HOME/envs/myenv"
fi
source "$HOME/envs/myenv/bin/activate"

# Install required Python packages if they are not already installed
pip install --no-index --upgrade pip
pip install --no-index numpy scipy opt_einsum tqdm qecstruct more_itertools networkx
pip install git+ssh://git@github.com/quicophy/matrex.git

#batches=(9)                                        # Array of batches
#nums_qubits=(30)                                   # Array of numbers of qubits
#code_ids=(5 8 9 17 24 30 35 39 44 77 84 87 88 91)  # Array of code IDs (max 99)
#nums_qubits=(40)
#code_ids=(0 1 2 3 4 6 10 11 13 14 19 20 21 22 24 25 28 29 30 31 33 35 36 37 39 41 43 44 45 46 47 48 49 50 51 52 54 55 56 57 58 59 60 61 63 65 66 67 68 69 70 71 72 75 76 77 78 79 80 81 84 85 86 89 90 91 94 95 96 97 98 99)
#nums_qubits=(50)
#code_ids=({0..99})
#nums_qubits=(60)
#code_ids=(0 1 2 3 4 5 6 7 8 9 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 47 49 50 51 52 54 55 56 57 58 59 60 64 66 67 69 70 71 72 73 74 76 77 78 79 80 81 85 86 87 88 89 90 91 93 94 96 97 99)
#nums_qubits=(70)
#code_ids=(16 20 22 25 29 30 32 33 39 41 45 46 47 48 54 59 61 62 64 67 72 73 78 80 85 87 95 98)
#nums_qubits=(80)
#code_ids=(1 3 7 25 35 38 42 47 50 53 74 79)
#nums_qubits=(90)
#code_ids=(17 24 30)

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
                                    # Define job script filename
                                    job_script="submit-job-${num_qubits}-${bond_dim}-${error_rate}-${error_model}-${batch}-${code_id}-${seed}.sh"
                                    # Create the job submission script
                                    cat > "$job_script" <<EOS
#!/bin/bash
#SBATCH --time=12:00:00                                                                                         # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=${num_processes}                                                                        # Number of CPU cores per task
#SBATCH --mem=32000                                                                                             # Memory per node
#SBATCH --job-name=csp-${num_qubits}-${bond_dim}-${error_rate}-${error_model}-${batch}-${code_id}-${seed}       # Descriptive job name
#SBATCH --output=csp-${num_qubits}-${bond_dim}-${error_rate}-${error_model}-${batch}-${code_id}-${seed}-%j.out  # Standard output and error log

export OMP_NUM_THREADS=1
module load python/3.11.5
source "$HOME/envs/myenv/bin/activate"

# Run the Python script with the specified arguments
python examples/decoding/quantum_csp.py --num_qubits ${num_qubits} --batch ${batch} --code_id ${code_id} --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance} --cut ${cut}
EOS
                            echo "Submitting the job for ${num_qubits} qubits, bond dimension ${bond_dim}, error rate ${error_rate}, error model ${error_model}, bias probability ${bias_prob}, batch ${batch}, code id ${code_id} and seed ${seed}."
                            sbatch "$job_script"
                            rm "$job_script"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All jobs have been submitted. Check the queue with 'squeue -u \${USER}'"
