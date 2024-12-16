#!/usr/bin/env bash

# first arg as number of times to run the script (at least 2)
# second arg as last job id to base first run on
# third arg as sbatch args
# fourth arg as script to run
# rest of args as args to script

num_runs=$1
LAST_JOB_ID=$2
sbatch_args=$3
script=$4
shift 4
args=$@

# create helper function to run sbatch
run_sbatch() {
    sbatch --parsable $sbatch_args $1 $script $args $2
}
echo "Num runs: $num_runs"
echo "sbatch_args: $sbatch_args"
echo "script: $script"
echo "args: $args"


# LAST_JOB_ID=$(run_sbatch)
echo "Job runned after: $LAST_JOB_ID"
LAST_JOB_ID=$(run_sbatch "--dependency=afterany:$LAST_JOB_ID")
echo "Job id 1: $LAST_JOB_ID"
for ((i=2; i<=$num_runs; i++))
do
    LAST_JOB_ID=$(run_sbatch "--dependency=afterany:$LAST_JOB_ID" "checkpoint_path=/app/model_checkpoints/${LAST_JOB_ID}/last.ckpt")
    echo "Job id $i: $LAST_JOB_ID"
done
