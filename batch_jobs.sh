#!/bin/bash

NOISE=$1
RESULTS_DIR=$2

for acq in logei jes pes
do
    sbatch submit array_job.sh Michalewicz 5 100 $NOISE $acq $RESULTS_DIR
    sbatch submit array_job.sh Michalewicz 10 150 $NOISE $acq $RESULTS_DIR
    sbatch submit array_job.sh Ackley 8 100 $NOISE $acq $RESULTS_DIR
    sbatch submit array_job.sh Ackley 16 150 $NOISE $acq $RESULTS_DIR
    sbatch submit array_job.sh Hartmann 4 50 $NOISE $acq $RESULTS_DIR
    sbatch submit array_job.sh Hartmann 6 50 $NOISE $acq $RESULTS_DIR
    sbatch submit array_job.sh Levy 4 50 $NOISE $acq $RESULTS_DIR
    sbatch submit array_job.sh Levy 8 100 $NOISE $acq $RESULTS_DIR
done