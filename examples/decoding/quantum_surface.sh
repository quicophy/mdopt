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

# Define arrays of lattice sizes, bond dimensions, error rates, and seeds
lattice_sizes=(3 5 7 9 11)
bond_dims=(64 128 256)
seeds=(123 124 125 126) # 4 random seeds
num_experiments=25 # Per each random seed
error_model="Bit Flip" # Error model used in the experiments

error_rates=()
start=0.05
end=0.15
num_points=11
step=$(echo "($end - $start) / ($num_points - 1)" | bc -l)
for ((i=0; i<$num_points; i++))
do
    value=$(echo "$start + $i * $step" | bc -l)
    error_rates+=($value)
done

# Create job submission scripts by iterating over combinations of the arguments
for seed in "${seeds[@]}"; do
    for lattice_size in "${lattice_sizes[@]}"; do
        for bond_dim in "${bond_dims[@]}"; do
            for error_rate in "${error_rates[@]}"; do
                # Sanitize the error model name by replacing spaces with underscores
                sanitized_error_model=$(echo "${error_model}" | sed 's/ /_/g')
                # Define job script filename
                job_script="submit-job-${lattice_size}-${bond_dim}-${error_rate}-${sanitized_error_model}-${seed}.sh"
                # Create the job submission script
                cat > "$job_script" <<EOS
#!/bin/bash
#SBATCH --time=24:00:00                                                                                           # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                                                                                         # Number of CPU cores per task
#SBATCH --mem=32000                                                                                               # Memory per node
#SBATCH --job-name=decoding-${lattice_size}-${bond_dim}-${error_rate}-${sanitized_error_model}-${seed}            # Descriptive job name
#SBATCH --output=decoding-${lattice_size}-${bond_dim}-${error_rate}-${sanitized_error_model}-${seed}-%j.out       # Standard output and error log

module load python/3.11.5
source "$HOME/envs/myenv/bin/activate"

# Run the Python script with the specified arguments
python examples/decoding/quantum_surface.py --lattice_size ${lattice_size} --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --error_model "${error_model}" --seed ${seed}
EOS
                echo "Submitting the job for lattice size ${lattice_size}, bond dimension ${bond_dim}, error rate ${error_rate}, error model ${error_model}, and seed ${seed}."
                sbatch "$job_script"
                rm "$job_script"
            done
        done
    done
done

echo "All jobs have been submitted. Check the queue with 'squeue -u \${USER}'"
