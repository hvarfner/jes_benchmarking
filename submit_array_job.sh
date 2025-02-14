#!/bin/bash

# Input arguments from command line
BENCHMARK=$1
DIMENSION=$2
NOISE=$3
ITERATIONS=$4
NUM_SEEDS=$5
ACQUISITION_FUNCTIONS=("jes" "pes" "logei")

# Acquisition functions to run
NUM_ACQ=${#ACQUISITION_FUNCTIONS[@]}

# Total number of jobs (acquisition functions * seeds)
TOTAL_JOBS=$((NUM_ACQ * NUM_SEEDS))

# Slurm script output
SBATCH_SCRIPT="submit_botorch_job.slurm"
# Slurm script output

cat <<EOF > $SBATCH_SCRIPT
#!/bin/bash
#SBATCH --job-name=jesteest
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-$(($NUM_SEEDS - 1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH -A naiss2024-22-1657

# Determine acquisition function and seed from job ID
ACQ_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_ACQ))
SEED=$((SLURM_ARRAY_TASK_ID / NUM_ACQ))
ACQ=\${ACQUISITION_FUNCTIONS[ACQ_INDEX]}

# Run the Python script for the specific acquisition function and seed
python botorch_bo_runner.py \
    --acq \${ACQ} \
    --dim ${DIMENSION} \
    --f ${BENCHMARK} \
    --seed \${SEED} \
    --iters ${ITERATIONS} \
    --noise ${NOISE}

echo "Completed \${ACQ} with seed \${SEED}"
EOF

# Submit the batch job
sbatch $SBATCH_SCRIPT
