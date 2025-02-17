#!/bin/bash



#SBATCH --job-name=jes_ablations
#SBATCH --output=slurm_logs/benchmark_%A_%a.err
#SBATCH --error=slurm_logs/benchmark_%A_%a.err
#SBATCH --array=1-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -A naiss2024-22-1657

# Load the required modules (adjust based on your cluster setup)

# Define benchmark parameters
BENCHMARK=$1
DIMENSION=$2
ITERATIONS=$3
NOISE=$4
ACQ=$5
RESULTS_DIR=$6

python main.py \
    --acq ${ACQ} \
    --dim ${DIMENSION} \
    --f ${BENCHMARK} \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --iters ${ITERATIONS} \
    --noise ${NOISE} \
    --directory ${RESULTS_DIR}

echo "All tasks completed for seed ${SLURM_ARRAY_TASK_ID}"
