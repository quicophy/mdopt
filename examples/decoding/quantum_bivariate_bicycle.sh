#!/bin/bash

# Orders and polynomials for the bivariate bicycle codes
# [[72, 12, 6]]
#order_x=6
#order_y=6
#poly_a="x**3 + y + y**2"
#poly_b="y**3 + x + x**2"

# [[90, 8, 10]]
#order_x=15
#order_y=3
#poly_a="x**9 + y + y**2"
#poly_b="1 + x**2 + x**7"

# [[108, 8, 10]]
#order_x=9
#order_y=6
#poly_a="x**3 + y + y**2"
#poly_b="y**3 + x + x**2"

# [[144, 12, 12]]
#order_x=12
#order_y=6
#poly_a="x**3 + y + y**2"
#poly_b="y**3 + x + x**2"

bond_dims=(400)                                 # Array of bond dimensions
seeds=(                                         # 100 random seeds
    0 1 2 3 4 5 6 7 8 9
    100 101 102 103 104 105 106 107 108 109
    200 201 202 203 204 205 206 207 208 209
    300 301 302 303 304 305 306 307 308 309
    400 401 402 403 404 405 406 407 408 409
    500 501 502 503 504 505 506 507 508 509
    600 601 602 603 604 605 606 607 608 609
    700 701 702 703 704 705 706 707 708 709
    800 801 802 803 804 805 806 807 808 809
    900 901 902 903 904 905 906 907 908 909
)
num_experiments=100                             # Runs per each random seed
error_model="Bitflip"                           # The error model
bias_probs=(1e-3)                               # Decoder bias probabilities
tolerances=(0)                                  # Numerical tolerances for the MPS
cuts=(0)                                        # SVD cut-offs for the MPS
num_processes=16                                # Parallel processes
silent=false                                    # Suppress script output

error_rates=(0.0001)

# Create job submission scripts by iterating over combinations of the arguments
for seed in "${seeds[@]}"; do
    for bond_dim in "${bond_dims[@]}"; do
        for error_rate in "${error_rates[@]}"; do
            for bias_prob in "${bias_probs[@]}"; do
                for tolerance in "${tolerances[@]}"; do
                    for cut in "${cuts[@]}"; do
                        # Print current configuration
                        echo "Running ${num_experiments} experiments for orders ${order_x}, ${order_y}, polynomials "${poly_a}", "${poly_b}", bond dimension ${bond_dim}, error rate ${error_rate}, error model ${error_model}, and seed ${seed}."
                        # Run the Python script with the specified arguments
                        poetry run python examples/decoding/quantum_bivariate_bicycle.py --order_x ${order_x} --order_y ${order_y} --poly_a "${poly_a}" --poly_b "${poly_b}" --bond_dim ${bond_dim} --error_rate ${error_rate} --num_experiments ${num_experiments} --bias_prob ${bias_prob} --error_model "${error_model}" --seed ${seed} --num_processes ${num_processes} --silent ${silent} --tolerance ${tolerance} --cut ${cut}
                    done
                done
            done
        done
    done
done
