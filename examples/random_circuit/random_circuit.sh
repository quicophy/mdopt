##!/bin/bash

# Load the Python module
module load python/3.11.5

# Check if the virtual environment exists, if not, create and activate it
if [ ! -d "~/envs/myenv" ]; then
    virtualenv --no-download ~/envs/myenv
fi
source ~/envs/myenv/bin/activate

# Install required Python packages if they are not already installed
pip install --no-index --upgrade pip
pip install --no-index numpy scipy opt_einsum tqdm

# Create a job submission script
cat > submit-job.sh << 'EOS'
#!/bin/bash
#SBATCH --time=24:00:00            # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=256000               # Memory per node
#SBATCH --job-name=random-circuit  # Descriptive job name
#SBATCH --output=%x-%j.out         # Standard output and error log

module load python/3.11.5
source ~/envs/myenv/bin/activate

# Resolve the script inside the installed package rather than by a
# checkout-relative path, so the batch job does not depend on its working
# directory. The file name has a hyphen, so it cannot be run with -m.
python "$(python -c 'import pathlib, mdopt.examples.random_circuit as m; print(pathlib.Path(m.__file__).parent / "mps-rand-circ.py")')"
EOS

# Submit the job
echo "Submitting the job..."
sbatch submit-job.sh

echo "Job submission script has been created and the job is submitted. Check the queue with 'squeue -u \${USER}'"
