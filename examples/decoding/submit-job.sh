#!/bin/bash
#SBATCH --time=12:00:00                     # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=16                  # Number of CPU cores per task
#SBATCH --mem=256000                        # Memory per node
#SBATCH --job-name=decoding-classical-ldpc  # Descriptive job name
#SBATCH --output=%x-%j.out                  # Standard output and error log

module load python/3.11.5
source ~/envs/myenv/bin/activate

python classical.py
