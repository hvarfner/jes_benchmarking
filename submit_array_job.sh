#!/bin/bash

# Input arguments from command line
BENCHMARK=$1
DIMENSION=$2
NOISE=$3
ITERATIONS=$4
NUM_SEEDS=$5
ACQUISITION_FUNCTIONS=("jes" "pes" "logei")

# Slurm script output
SBATCH_SCRIPT="submit_botorch_job.slurm"

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

# Run for each acquisition function in parallel

for ACQ in "${ACQUISITION_FUNCTIONS[@]}"; do
    echo "Running ${ACQ} with seed ${SLURM_ARRAY_TASK_ID}"
    python main.py \
        --acq ${ACQ} \
        --dim ${DIMENSION} \
        --f ${BENCHMARK} \
        --seed ${SLURM_ARRAY_TASK_ID} \
        --iters ${ITERATIONS} \
        --noise ${NOISE} &
done

EOF

# Submit the batch job
sbatch $SBATCH_SCRIPT