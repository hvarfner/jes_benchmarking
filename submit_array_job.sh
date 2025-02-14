#!/bin/bash


BENCHMARK=$1
DIMENSION=$2
NOISE=$3
ITERATIONS=$4
NUM_SEEDS=$5
ACQUISITION_FUNCTION=$6

#SBATCH --job-name=${BENCHMARK}_${ACQUISITION_FUNCTION}
#SBATCH --output=slurm_logs/${BENCHMARK}_${ACQUISITION_FUNCTION}_%A_%a.out
#SBATCH --error=slurm_logs/${BENCHMARK}_${ACQUISITION_FUNCTION}_%A_%a.err
#SBATCH --array=0-$((${NUM_SEEDS}-1))
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -A naiss2024-22-1657


python main.py \
    --acq ${ACQUISITION_FUNCTION} \
    --dim ${DIMENSION} \
    --f ${BENCHMARK} \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --iters ${ITERATIONS} \
    --noise ${NOISE}

echo "All tasks completed for seed ${SLURM_ARRAY_TASK_ID}"
