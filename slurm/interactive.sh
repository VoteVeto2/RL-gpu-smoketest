#!/bin/bash
# Get an interactive shell on a GPU node (for debugging). Exit the shell to release the GPU.
#   slurm/interactive.sh a100-debug   then, on the node:  bash smoketest/run_all.sh
set -euo pipefail
ACCOUNT=${VSC_ACCOUNT:-intro_vsc38501}
case "${1:-a100-debug}" in
    v100)       RES=(--partition=gpu_v100       --cpus-per-task=4  --time=01:00:00) ;;
    a100)       RES=(--partition=gpu_a100       --cpus-per-task=18 --time=01:00:00) ;;
    a100-debug) RES=(--partition=gpu_a100_debug --cpus-per-task=16 --time=00:59:00) ;;
    *) echo "usage: $0 <v100|a100|a100-debug>"; exit 1 ;;
esac
exec srun --clusters=wice --account="$ACCOUNT" "${RES[@]}" --nodes=1 --ntasks=1 --gpus-per-node=1 --pty bash -l
