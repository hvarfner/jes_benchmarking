#!/bin/bash

NOISE=$1
RESULTS_DIR=$2

for acq in logei jes pes
do
    sbatch submit_array_job.sh Ackley 6 50 2.5 $acq $RESULTS_DIR
    sbatch submit_array_job.sh Hartmann 4 50 0.5 $acq $RESULTS_DIR
    sbatch submit_array_job.sh Hartmann 6 50 0.5 $acq $RESULTS_DIR
    sbatch submit_array_job.sh Levy 4 50 2.5 $acq $RESULTS_DIR
done