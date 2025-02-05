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

# Define arrays of lattice sizes, bond dimensions, error rates, and seeds
lattice_sizes=(3 5 7)
bond_dims=(20)
seeds=(
    123 124 125 126 127 128 129 130 131 132
) # 10 random seeds
num_experiments=100 # Per each random seed
error_model="Bitflip" # Error model used in the experiments
bias_prob=0.1 # The decoder bias probability
tolerance=1e-6 # The numerical tolerance for the MPS within the decoder.
num_processes=16 # Number of processes to use in parallel
silent=true # Whether to suppress the output of the Python script

error_rates=()
start=0.105
end=0.115
step=0.001
current=$start
while (( $(echo "$current <= $end" | bc -l) ))
do
    error_rates+=($current)
    current=$(echo "$current + $step" | bc -l)
done

# Create job submission scripts by iterating over combinations of the arguments
for seed in "${seeds[@]}"; do
    for lattice_size in "${lattice_sizes[@]}"; do
        for bond_dim in "${bond_dims[@]}"; do
            for error_rate in "${error_rates[@]}"; do
                # Define job script filename
                job_script="submit-job-${lattice_size}-${bond_dim}-${error_rate}-${error_model}-${seed}.sh"
                # Create the job submission script
                cat > "$job_script" <<EOS
#!/bin/bash
#SBATCH --time=96:00:00                                                                              # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=${num_processes}                                                             # Number of CPU cores per task
#SBATCH --mem=16000                                                                                  # Memory per node
#SBATCH --job-name=decoding-${lattice_size}-${bond_dim}-${error_rate}-${error_model}-${seed}         # Descriptive job name
#SBATCH --output=decoding-${lattice_size}-${bond_dim}-${error_rate}-${error_model}-${seed}-%j.out    # Standard output and error log

module load python/3.11.5
source "$HOME/envs/myenv/bin/activate"

# Run the Python script with the specified arguments
python examples/decoding/quantum_surface.py --lattice_size ${lattice_size} --bond_dim ${bond_dim} --error_rate ${error_rate} --bias_prob ${bias_prob} --num_experiments ${num_experiments} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance}
EOS
                echo "Submitting the job for lattice size ${lattice_size}, bond dimension ${bond_dim}, error rate ${error_rate}, error model ${error_model}, bias probability ${bias_prob} and seed ${seed}."
                sbatch "$job_script"
                rm "$job_script"
            done
        done
    done
done

echo "All jobs have been submitted. Check the queue with 'squeue -u \${USER}'"
