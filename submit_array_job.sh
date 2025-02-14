#!/bin/bash
#SBATCH --job-name=${1}${2}_${6}
#SBATCH --output=slurm_logs/benchmark_%A_%a.err
#SBATCH --error=slurm_logs/benchmark_%A_%a.err
#SBATCH --array=1-${5}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -A naiss2024-22-1657

# Load the required modules (adjust based on your cluster setup)

# Define benchmark parameters
BENCHMARK=$1
DIMENSION=$2
ITERATIONS=$3
NOISE=$4
# SEEDS=$5
ACQ=$6

python main.py \
    --acq ${ACQ} \
    --dim ${DIMENSION} \
    --f ${BENCHMARK} \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --iters ${ITERATIONS} \
    --noise ${NOISE}

echo "All tasks completed for seed ${SLURM_ARRAY_TASK_ID}"
