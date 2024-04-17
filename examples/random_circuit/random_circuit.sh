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
#SBATCH --time=02:00:00            # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=32000                # Memory per node
#SBATCH --job-name=random-circuit  # Descriptive job name
#SBATCH --output=%x-%j.out         # Standard output and error log

module load python/3.11.5
source ~/envs/myenv/bin/activate

python mps-rand-circ.py
EOS

# Submit the job
echo "Submitting the job..."
sbatch submit-job.sh

echo "Job submission script has been created and the job is submitted. Check the queue with 'squeue -u \${USER}'"
