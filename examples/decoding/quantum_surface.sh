#!/bin/bash

# Load the Python module
module load python/3.11.5

# Check if the virtual environment exists, if not, create and activate it
if [ ! -d "~/envs/myenv" ]; then
    virtualenv --no-download ~/envs/myenv
fi
source ~/envs/myenv/bin/activate

# Install required Python packages if they are not already installed
pip install --no-index --upgrade pip
pip install --no-index numpy scipy opt_einsum tqdm qecstruct more_itertools networkx matrex@git+https://github.com/quicophy/matrex

# Define arrays of system sizes, bond dimensions, and error probabilities
lattice_sizes=(3)
bond_dims=(128)
seeds=(123 124 125 126 127 128 129 130 131 132 133) # 10 random seeds
num_experiments=10 # Per each random seed

error_probabilities=()
start=0.0001
end=0.1
num_points=10
step=$(echo "($end - $start) / ($num_points - 1)" | bc -l)
for ((i=0; i<$num_points; i++))
do
    value=$(echo "$start + $i * $step" | bc -l)
    error_probabilities+=($value)
done

# Create job submission scripts by iterating over
# each combination of seed, lattice_size, bond_dim, and error_prob and submitting them
for seed in "${seeds[@]}"; do
    for lattice_size in "${lattice_sizes[@]}"; do
        for bond_dim in "${bond_dims[@]}"; do
            for error_prob in "${error_probabilities[@]}"; do
                # Create a job submission script for each combination
                cat > "submit-job-${lattice_size}-${bond_dim}-${error_prob}-${seed}.sh" <<EOS
#!/bin/bash
#SBATCH --time=24:00:00                                                                             # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                                                                           # Number of CPU cores per task
#SBATCH --mem=16000                                                                                 # Memory per node
#SBATCH --job-name=decoding-quantum-surface-${lattice_size}-${bond_dim}-${error_prob}-${seed}       # Descriptive job name
#SBATCH --output=decoding-quantum-surface-${lattice_size}-${bond_dim}-${error_prob}-${seed}-%j.out  # Standard output and error log

module load python/3.11.5
source ~/envs/myenv/bin/activate

# Run the Python script with the specified lattice size, bond dimension, and error probability
python examples/decoding/quantum_surface.py --lattice_size $lattice_size --bond_dim $bond_dim \
    --error_prob $error_prob --num_experiments $num_experiments --seed $seed
EOS
            echo "Submitting the job for lattice size ${lattice_size}, bond dimension ${bond_dim}, error probability ${error_prob} and seed ${seed}."
            sbatch "submit-job-${lattice_size}-${bond_dim}-${error_prob}-${seed}.sh"
            rm "submit-job-${lattice_size}-${bond_dim}-${error_prob}-${seed}.sh"
            done
        done
    done
done

echo "All jobs have been submitted. Check the queue with 'squeue -u \${USER}'"