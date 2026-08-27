#!/bin/bash
#SBATCH --time=24:00:00            # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=256000               # Memory per node
#SBATCH --job-name=random-circuit  # Descriptive job name
#SBATCH --output=%x-%j.out         # Standard output and error log

module load python/3.11.5
source ~/envs/myenv/bin/activate

# Resolve the script inside the installed package: it no longer sits beside
# this file. The hyphen in the name rules out `python -m`.
python "$(python -c 'import pathlib, mdopt.examples.random_circuit as m; print(pathlib.Path(m.__file__).parent / "mps-rand-circ.py")')"
