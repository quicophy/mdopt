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
lattice_sizes=(13)
bond_dims=(256)
seeds=(
    123 124 125 126 127 128 129 130 131 132
    133 134 135 136 137 138 139 140 141 142
    143 144 145 146 147 148 149 150 151 152
    153 154 155 156 157 158 159 160 161 162
    163 164 165 166 167 168 169 170 171 172
    173 174 175 176 177 178 179 180 181 182
    183 184 185 186 187 188 189 190 191 192
    193 194 195 196 197 198 199 200 201 202
    203 204 205 206 207 208 209 210 211 212
    213 214 215 216 217 218 219 220 221 222
) # 100 random seeds
num_experiments=10 # Per each random seed
error_model="Bit Flip" # Error model used in the experiments

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
                # Sanitize the error model name by replacing spaces with underscores
                sanitized_error_model=$(echo "${error_model}" | sed 's/ /_/g')
                # Define job script filename
                job_script="submit-job-${lattice_size}-${bond_dim}-${error_rate}-${sanitized_error_model}-${seed}.sh"
                # Create the job submission script
                cat > "$job_script" <<EOS
#!/bin/bash
#SBATCH --time=48:00:00                                                                                           # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=1                                                                                         # Number of CPU cores per task
#SBATCH --mem=64000                                                                                               # Memory per node
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
