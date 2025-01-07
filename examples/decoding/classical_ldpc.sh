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

# Define arrays of system sizes, bond dimensions, and error rates
system_sizes=(96 192 384)
bond_dims=(1024)
seeds=(
    123 124 125 126 127 128 129 130 131 132 133
    134 135 136 137 138 139 140 141 142 143 144
    145 146 147 148 149 150 151 152 153 154 155
    156 157 158 159 160 161 162 163 164 165 166
    167 168 169 170 171 172
) # 50 random seeds
num_experiments=2 # Per each random seed

error_rates=()
start=0.05
end=0.45
num_points=11
step=$(echo "($end - $start) / ($num_points - 1)" | bc -l)
for ((i=0; i<$num_points; i++))
do
    value=$(echo "$start + $i * $step" | bc -l)
    error_rates+=($value)
done

# Create job submission scripts by iterating over each combination of
# random seed, system_size, bond_dim, and error_rate and submitting them
for seed in "${seeds[@]}"; do
    for system_size in "${system_sizes[@]}"; do
        for bond_dim in "${bond_dims[@]}"; do
            for error_rate in "${error_rates[@]}"; do
                # Create a job submission script for each combination
                cat > "submit-job-${system_size}-${bond_dim}-${error_rate}-${seed}.sh" <<EOS
#!/bin/bash
#SBATCH --time=24:00:00                                                                           # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                                                                         # Number of CPU cores per task
#SBATCH --mem=32000                                                                               # Memory per node
#SBATCH --job-name=decoding-classical-ldpc-${system_size}-${bond_dim}-${error_rate}-${seed}       # Descriptive job name
#SBATCH --output=decoding-classical-ldpc-${system_size}-${bond_dim}-${error_rate}-${seed}-%j.out  # Standard output and error log

module load python/3.11.5
source "$HOME/envs/myenv/bin/activate"

# Run the Python script with the specified system size, bond dimension, and error rate
python examples/decoding/classical_ldpc.py --system_size $system_size --bond_dim $bond_dim --error_rate $error_rate --num_experiments $num_experiments --seed $seed
EOS
                echo "Submitting the job for system size ${system_size}, bond dimension ${bond_dim}, error rate ${error_rate} and seed ${seed}."
                sbatch "submit-job-${system_size}-${bond_dim}-${error_rate}-${seed}.sh"
                rm "submit-job-${system_size}-${bond_dim}-${error_rate}-${seed}.sh"
            done
        done
    done
done

echo "All jobs have been submitted. Check the queue with 'squeue -u \${USER}'"
