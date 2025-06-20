#!/bin/bash

# Load the Python module
module load python/3.11.5

# Directory for your virtual environment
ENV_DIR="$HOME/envs/myenv"

# Create and populate the venv on first run (login node, with internet)
if [ ! -d "$ENV_DIR" ]; then
    python -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
    pip install numpy scipy opt_einsum tqdm qecstruct more_itertools networkx sympy qldpc
    pip install git+https://github.com/quicophy/matrex.git
else
    source "$ENV_DIR/bin/activate"
fi

# Create job submission scripts by iterating over combinations of the arguments
order_x=6                                       # Orders for polynomials to create BB codes
order_y=6
poly_a="1 + x + y"                              # The polynomials used to create BB codes
poly_b="1 + x**2 + y**2"
bond_dims=(30)                                  # Array of bond dimensions
seeds=(100 101 102 103 104 105 106 107 108 109) # (10) random seeds
num_experiments=100                             # Runs per each random seed
error_model="Bitflip"                           # The error model
bias_probs=(1e-1)                               # Decoder bias probabilities
tolerances=(0)                                  # Numerical tolerances for the MPS
cuts=(0)                                        # SVD cut-offs for the MPS
num_processes=16                                # Parallel processes
silent=false                                    # Suppress script output?

error_rates=()
start=0.01
end=0.07
step=0.01
current=$start
while (( $(echo "$current <= $end" | bc -l) )); do
    error_rates+=($current)
    current=$(echo "$current + $step" | bc -l)
done

# Create & submit one SBATCH script per parameter combo
for seed in "${seeds[@]}"; do
  for bond_dim in "${bond_dims[@]}"; do
    for error_rate in "${error_rates[@]}"; do
      for bias_prob in "${bias_probs[@]}"; do
        for tolerance in "${tolerances[@]}"; do
          for cut in "${cuts[@]}"; do

            job_script="submit-job-${order_x}x${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}.sh"

            cat > "$job_script" <<EOS
#!/bin/bash
export OMP_NUM_THREADS=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=${num_processes}
#SBATCH --mem=4000
#SBATCH --job-name=bivar-${order_x}x${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}
#SBATCH --output=bivar-${order_x}x${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}-%j.out

module load python/3.11.5
source "\$HOME/envs/myenv/bin/activate"

python examples/decoding/quantum_bivariate_bicycle.py --order_x ${order_x} --order_y ${order_y} --poly_a "${poly_a}" --poly_b "${poly_b}" --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance} --cut ${cut}
EOS

            echo "Submitting: orders ${order_x}x${order_y}, bond_dim=${bond_dim}, error_rate=${error_rate}, seed=${seed}"
            sbatch "$job_script"
            rm "$job_script"

          done
        done
      done
    done
  done
done

echo "All job submissions dispatched."
