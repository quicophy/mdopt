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
pip install --no-index numpy scipy opt_einsum tqdm qecstruct more_itertools networkx sympy qldpc
pip install git+ssh://git@github.com/quicophy/matrex.git

order_x=6                                       # Orders for polynomials to create BB codes
order_y=6
poly_a="1 + x + y"                              # The polynomials used to create BB codes
poly_b="1 + x**2 + y**2"
bond_dims=(30)                                  # Array of bond dimensions
seeds=(100 101 102 103 104 105 106 107 108 109) # Array of (10) random seeds
num_experiments=100                             # Runs per each random seed
error_model="Bitflip"                           # The error model
bias_probs=(1e-1)                               # Array of decoder bias probabilities
tolerances=(0)                                  # Array of numerical tolerances for the MPS within the decoder
cuts=(0)                                        # Array of SVD cut-offs for the MPS within the decoder
num_processes=16                                # The number of processes to use in parallel
silent=false                                    # Whether to suppress the output of the Python script

error_rates=()
start=0.01
end=0.07
step=0.01
current=$start
while (( $(echo "$current <= $end" | bc -l) ))
do
    error_rates+=($current)
    current=$(echo "$current + $step" | bc -l)
done

# Create job submission scripts by iterating over combinations of the arguments
for seed in "${seeds[@]}"; do
    for bond_dim in "${bond_dims[@]}"; do
        for error_rate in "${error_rates[@]}"; do
            for bias_prob in "${bias_probs[@]}"; do
                for tolerance in "${tolerances[@]}"; do
                    for cut in "${cuts[@]}"; do
                        # Define a descriptive job‐script filename
                        job_script="submit-job-${order_x}x${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}.sh"

                        # Write the SBATCH script
                        cat > "$job_script" <<EOS
#!/bin/bash
export OMP_NUM_THREADS=1
#SBATCH --time=08:00:00                                                               # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=${num_processes}                                              # CPU cores per task
#SBATCH --mem=4000                                                                    # Memory per node
#SBATCH --job-name=bivar-${order_x}x${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}
#SBATCH --output=bivar-${order_x}x${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}-%j.out

module load python/3.11.5
source "\$HOME/envs/myenv/bin/activate"

# Run the Python script with the specified arguments
poetry run python examples/decoding/quantum_bivariate_bicycle.py --order_x ${order_x} --order_y ${order_y} --poly_a "${poly_a}" --poly_b "${poly_b}" --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance} --cut ${cut}
EOS
                        # Submit and clean up
                        echo "Submitting job for orders ${order_x}x${order_y}, bond_dim=${bond_dim}, error_rate=${error_rate}, error_model=${error_model}, bias_prob=${bias_prob}, seed=${seed}."
                        sbatch "$job_script"
                        rm "$job_script"
                    done
                done
            done
        done
    done
done

echo "All calculations are complete."
