# Slurm script to run 20 seeds for all acquisition functions as an array job
# Save this script as `run_benchmark.slurm`

#!/bin/bash
#SBATCH --job-name=botorch_benchmark
#SBATCH --output=results/slurm_logs/benchmark_%A_%a.out
#SBATCH --error=results/slurm_logs/benchmark_%A_%a.err
#SBATCH --array=1-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -A=naiss2024-22-1657

# Load the required modules (adjust based on your cluster setup)

# Define benchmark parameters
BENCHMARK="Ackley"
DIMENSION=8
ITERATIONS=100
NOISE=0.01
ACQ="jes"

python main.py \
    --acq ${ACQ} \
    --dim ${DIMENSION} \
    --f ${BENCHMARK} \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --iters ${ITERATIONS} \
    --noise ${NOISE}

echo "All tasks completed for seed ${SLURM_ARRAY_TASK_ID}"
