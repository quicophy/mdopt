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
pip install --no-index numpy scipy opt_einsum tqdm qecstruct more_itertools networkx matrex@git+https://github.com/quicophy/matrex

# Define arrays of numbers of qubits, bond dimensions, error rates, and batches
nums_qubits=(30) # 40 50 60 70 80 90 100)     # The number of qubits
bond_dims=(8 16 32 64 128 256 512 1024)       # Bond dimensions to scan
seeds=(123 124 125 126)                       # 4 random seeds
num_experiments=25                            # Per each random seed
batches=(1 2 3 4 5) # 6 7 8 9 10 11 12 13 14) # Batches from 1 to 14

error_rates=()
start=0.01
end=0.5
num_points=10
step=$(echo "($end - $start) / ($num_points - 1)" | bc -l)
for ((i=0; i<$num_points; i++))
do
    value=$(echo "$start + $i * $step" | bc -l)
    error_rates+=($value)
done

# Create job submission scripts by iterating over
# each combination of seed, batch, num_qubits, bond_dim, and error_rate and submitting them
for seed in "${seeds[@]}"; do
    for num_qubits in "${nums_qubits[@]}"; do
        for bond_dim in "${bond_dims[@]}"; do  # Iterate over bond dimensions
            for error_rate in "${error_rates[@]}"; do
                for batch in "${batches[@]}"; do  # Iterate over batches
                    # Create a job submission script for each combination
                    cat > "submit-job-batch${batch}-qubits${num_qubits}-bond${bond_dim}-error${error_rate}-seed${seed}.sh" <<EOS
#!/bin/bash
#SBATCH --time=24:00:00                                                                                               # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                                                                                             # Number of CPU cores per task
#SBATCH --mem=16000                                                                                                   # Memory per node
#SBATCH --job-name=decoding-csp-batch${batch}-qubits${num_qubits}-bond${bond_dim}-error${error_rate}-seed${seed}      # Descriptive job name
#SBATCH --output=decoding-csp-batch${batch}-qubits${num_qubits}-bond${bond_dim}-error${error_rate}-seed${seed}-%j.out # Standard output and error log

module load python/3.11.5
source "$HOME/envs/myenv/bin/activate"

# Run the Python script with the specified parameters
python examples/decoding/quantum_csp.py --num_qubits ${num_qubits} --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --seed ${seed} --batch ${batch}
EOS
                    echo "Submitting the job for batch ${batch}, number of qubits ${num_qubits}, bond dimension ${bond_dim}, error rate ${error_rate}, and seed ${seed}."
                    sbatch "submit-job-batch${batch}-qubits${num_qubits}-bond${bond_dim}-error${error_rate}-seed${seed}.sh"
                    rm "submit-job-batch${batch}-qubits${num_qubits}-bond${bond_dim}-error${error_rate}-seed${seed}.sh"
                done
            done
        done
    done
done

echo "All jobs have been submitted. Check the queue with 'squeue -u \${USER}'"
