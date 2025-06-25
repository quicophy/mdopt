#!/bin/bash

# Load the Python module
module load python/3.11.5

# Directory for your virtual environment
ENV_DIR="$HOME/envs/myenv"

# Create and populate the venv on first run
if [ ! -d "$ENV_DIR" ]; then
    python -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
    pip install numpy scipy tqdm qecstruct more_itertools networkx sympy opt_einsum
    pip install git+ssh://git@github.com/quicophy/matrex.git

    # Build & install the C++ ldpc library from source
    if [ ! -d "$HOME/ldpc" ]; then
      git clone https://github.com/quantumgizmos/ldpc.git "$HOME/ldpc"
    fi
    cd "$HOME/ldpc"
    git checkout v2.3.6   # or whatever tag matches qLDPC’s ldpc>=2.1.8 requirement
    pip install .         # compiles rng.hpp, etc.

    # Install qLDPC (will see local ldpc and pull in the rest)
    if [ ! -d "$HOME/qLDPC" ]; then
      git clone https://github.com/qLDPCOrg/qLDPC.git "$HOME/qLDPC"
    fi
    cd "$HOME/qLDPC"
    pip install .         # installs qLDPC plus the dependencies

    # back to the working directory and uninstall matplotlib
    cd /home/bereza/projects/def-ko1/bereza/mdopt
else
    source "$ENV_DIR/bin/activate"
fi

# Create job submission scripts by iterating over combinations of the arguments
order_x=6                                       # Orders for polynomials to create BB codes
order_y=6
poly_a="x**3 + y + y**2"                        # The polynomials used to create BB codes
poly_b="y**3 + x + x**2"
bond_dims=(30)                                  # Array of bond dimensions
seeds=(100 101 102 103 104 105 106 107 108 109) # (10) random seeds
num_experiments=500                             # Runs per each random seed
error_model="Bitflip"                           # The error model
bias_probs=(1e-1)                               # Decoder bias probabilities
tolerances=(0)                                  # Numerical tolerances for the MPS
cuts=(0)                                        # SVD cut-offs for the MPS
num_processes=16                                # Parallel processes
silent=false                                    # Suppress script output

error_rates=()
start=0.01
end=0.07
step=0.01
current=$start
while (( $(echo "$current <= $end" | bc -l) )); do
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
                        # Define job script filename
                        job_script="submit-job-${order_x}x${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}.sh"
                        # Create the job submission script
                        cat > "$job_script" <<EOS
#!/bin/bash
#SBATCH --time=04:00:00                                                                                  # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=${num_processes}                                                                 # Number of CPU cores per task
#SBATCH --mem=4000                                                                                       # Memory per node
#SBATCH --job-name=decoding-${order_x}-${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}       # Descriptive job name
#SBATCH --output=decoding-${order_x}-${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}-%j.out  # Standard output and error log

export OMP_NUM_THREADS=1
module load python/3.11.5
source "\$HOME/envs/myenv/bin/activate"

# Run the Python script with the specified arguments
python examples/decoding/quantum_bivariate_bicycle.py --order_x ${order_x} --order_y ${order_y} --poly_a "${poly_a}" --poly_b "${poly_b}" --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance} --cut ${cut}
EOS
                        echo "Submitting the job for orders ${order_x}x${order_y}, bond dimension ${bond_dim}, error rate ${error_rate}, error model ${error_model}, bias probability ${bias_prob} and seed ${seed}."
                        sbatch "$job_script"
                        rm "$job_script"
                    done
                done
            done
        done
    done
done

echo "All jobs have been submitted. Check the queue with 'squeue -u \${USER}'"
