#!/bin/bash
#SBATCH --job-name=whatever_logei
#SBATCH --output=slurm_logs/job-%A_%a.out
#SBATCH --error=slurm_logs/job-%A_%a.err
#SBATCH --array=0-19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH -A naiss2024-22-1657


python main.py     --acq logei     --dim 52     --f whatever     --seed      --iters 100     --noise 0.01
