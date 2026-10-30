#!/bin/bash -l
# Batch job for the smoketest. The #SBATCH lines below are the defaults (A100 debug partition);
# slurm/submit.sh overrides partition/cpus/time per GPU type -- command-line flags beat #SBATCH lines.
#SBATCH --job-name=rl-smoketest
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc38501
#SBATCH --partition=gpu_a100_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:59:00
#SBATCH --output=logs/%x-%j.out

set -eu
cd "$SLURM_SUBMIT_DIR"
source env.sh
export HF_HUB_OFFLINE=1  # models/data were pre-downloaded by setup_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1  # stream training logs to logs/*.out as they happen

echo "Job $SLURM_JOB_ID on $(hostname) | partition=$SLURM_JOB_PARTITION | cpus=$SLURM_CPUS_PER_TASK | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export RUN_DIR=$RUN_ROOT/$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
# Sample GPU utilisation/memory every 5 s for later inspection.
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$RUN_DIR/gpu_usage.csv" &

bash smoketest/run_all.sh
