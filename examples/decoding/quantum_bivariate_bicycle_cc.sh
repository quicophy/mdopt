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
    git checkout v2.3.6
    pip install .

    # Install qLDPC
    if [ ! -d "$HOME/qLDPC" ]; then
      git clone https://github.com/qLDPCOrg/qLDPC.git "$HOME/qLDPC"
    fi
    cd "$HOME/qLDPC"
    pip install .

    cd /home/bereza/projects/def-ko1/bereza/mdopt
else
    source "$ENV_DIR/bin/activate"
fi

# ── Code: [[72, 12, 6]] — the critical gap in plotting_bb.ipynb ──────────────
order_x=6
order_y=6
poly_a="x**3 + y + y**2"
poly_b="y**3 + x + x**2"

bond_dims=(400)
seeds=(
    1100 1101 1102 1103 1104 1105 1106 1107 1108 1109
    1110 1111 1112 1113 1114 1115 1116 1117 1118 1119
    1120 1121 1122 1123 1124 1125 1126 1127 1128 1129
    1130 1131 1132 1133 1134 1135 1136 1137 1138 1139
    1140 1141 1142 1143 1144 1145 1146 1147 1148 1149
    1150 1151 1152 1153 1154 1155 1156 1157 1158 1159
    1160 1161 1162 1163 1164 1165 1166 1167 1168 1169
    1170 1171 1172 1173 1174 1175 1176 1177 1178 1179
    1180 1181 1182 1183 1184 1185 1186 1187 1188 1189
    1190 1191 1192 1193 1194 1195 1196 1197 1198 1199
)

error_model="Bitflip"
bias_probs=(1e-3)
tolerances=(0)
cuts=(0)
num_processes=16
silent=false

error_rates=(0.01 0.001 0.0001)

# Differential num_experiments — sized so total samples per (N, p) gives
# either ~10 expected failures (p=0.01, 0.001) or a tight Poisson upper bound (p=0.0001).
#   p=0.01   → 100 seeds × 50,000   = 5,000,000 samples   (expect ~5 failures if LER ~ 1e-6)
#   p=0.001  → 100 seeds × 100,000  = 10,000,000 samples  (likely upper bound LER < 3e-7)
#   p=0.0001 → 100 seeds × 200,000  = 20,000,000 samples  (upper bound LER < 1.5e-7)
get_num_experiments() {
    case $1 in
        0.01)   echo 50000  ;;
        0.001)  echo 100000 ;;
        0.0001) echo 200000 ;;
    esac
}

# Create job submission scripts by iterating over combinations of the arguments
for seed in "${seeds[@]}"; do
    for bond_dim in "${bond_dims[@]}"; do
        for error_rate in "${error_rates[@]}"; do
            num_experiments=$(get_num_experiments "$error_rate")
            for bias_prob in "${bias_probs[@]}"; do
                for tolerance in "${tolerances[@]}"; do
                    for cut in "${cuts[@]}"; do
                        # Define job script filename
                        job_script="submit-job-${order_x}x${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}.sh"
                        # Create the job submission script
                        cat > "$job_script" <<EOS
#!/bin/bash
#SBATCH --time=48:00:00                                                                            # 2x safety margin (worst case ~35h for 200k-exp jobs)
#SBATCH --cpus-per-task=${num_processes}                                                           # 16 parallel python processes
#SBATCH --mem-per-cpu=6000                                                                         # 6 GB/core × 16 = 96 GB total; 20% margin over 5 GB/core peak
#SBATCH --job-name=bb-${order_x}-${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}       # Descriptive job name
#SBATCH --output=bb-${order_x}-${order_y}-${bond_dim}-${error_rate}-${error_model}-${seed}-%j.out  # Standard output and error log

export OMP_NUM_THREADS=1
module load python/3.11.5
source "\$HOME/envs/myenv/bin/activate"

# Run the Python script with the specified arguments
python -m mdopt.examples.decoding.quantum_bivariate_bicycle --order_x ${order_x} --order_y ${order_y} --poly_a "${poly_a}" --poly_b "${poly_b}" --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance} --cut ${cut}
EOS
                        echo "Submitting the job for orders ${order_x}x${order_y}, bond dimension ${bond_dim}, error rate ${error_rate}, num_exp ${num_experiments}, error model ${error_model}, bias probability ${bias_prob} and seed ${seed}."
                        sbatch "$job_script"
                        rm "$job_script"
                    done
                done
            done
        done
    done
done

echo "All jobs have been submitted. Check the queue with 'squeue -u \${USER}'"
