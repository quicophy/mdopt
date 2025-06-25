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

nums_qubits=(30)                                # Array of numbers of qubits
batches=(1)                                     # Array of batches
code_ids=(0)                                    # Array of code IDs
bond_dims=(20)                                  # Array of bond dimensions
seeds=(100) # Array of (10) random seeds
num_experiments=10                              # Runs per each random seed
error_model="Bitflip"                           # The error model
bias_probs=(1e-1)                               # Array of decoder bias probabilities
tolerances=(0)                                  # Array of numerical tolerances for the MPS within the decoder
cuts=(0)                                        # Array of SVD cut-offs for the MPS within the decoder
num_processes=16                                # The number of processes to use in parallel
silent=false                                    # Whether to suppress the output of the Python script

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
#SBATCH --time=09:00:00                                                                                          # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=2                                                                                        # Number of CPU cores per task
#SBATCH --mem=4000                                                                                               # Memory per node
#SBATCH --job-name=csp-${num_qubits}-${bond_dim}-${error_rate}-${error_model}-${batch}-${code_id}-${seed}        # Descriptive job name
#SBATCH --output=csp-${num_qubits}-${bond_dim}-${error_rate}-${error_model}-${batch}-${code_id}-${seed}-%j.out   # Standard output and error log

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

echo "All calculations are complete."
