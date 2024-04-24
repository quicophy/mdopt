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
pip install --no-index numpy scipy opt_einsum tqdm qecstruct more_itertools

# Define arrays of system sizes and bond dimensions
system_sizes=(96 192 384)
bond_dims=(128 256 512 1024)

# Create job submission scripts by iterating
# over each combination of system_size and bond_dim and submitting them
for system_size in "${system_sizes[@]}"; do
    for bond_dim in "${bond_dims[@]}"; do
        # Create a job submission script for each combination
        cat > "submit-job-${system_size}-${bond_dim}.sh" <<EOS
#!/bin/bash
#SBATCH --time=48:00:00                                                     # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                                                   # Number of CPU cores per task
#SBATCH --mem=64000                                                         # Memory per node
#SBATCH --job-name=decoding-classical-ldpc-${system_size}-${bond_dim}       # Descriptive job name
#SBATCH --output=decoding-classical-ldpc-${system_size}-${bond_dim}-%j.out  # Standard output and error log

module load python/3.11.5
source ~/envs/myenv/bin/activate

# Run the Python script with the specified system size and bond dimension
python examples/decoding/classical.py $system_size $bond_dim
EOS

        # Submit the job
        echo "Submitting the job for system size ${system_size} and bond dimension ${bond_dim}"
        sbatch "submit-job-${system_size}-${bond_dim}.sh" --export=system_size=$system_size,bond_dim=$bond_dim
    done
done

echo "All jobs have been submitted. Check the queue with 'squeue -u \${USER}'"