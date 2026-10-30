#!/bin/bash
# Submit the smoketest to a given GPU type.
#   slurm/submit.sh v100            # gpu_v100 partition (Tesla V100, sm_70, fp16 AMP)
#   slurm/submit.sh a100            # gpu_a100 partition (A100-SXM4 80GB, bf16)
#   slurm/submit.sh a100-debug      # gpu_a100_debug (1x A100-PCIe 80GB, max 1h, usually starts fast)
# Extra args are passed to sbatch, e.g.  slurm/submit.sh v100 --time=00:30:00 --export=ALL,STAGES=grpo_lora
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ACCOUNT=${VSC_ACCOUNT:-intro_vsc38501}

# CPU/memory per GPU follow the node layout: V100 node = 36 cores / 8 GPUs (21 GB RAM per core),
# A100 node = 72 cores / 4 GPUs (7 GB per core). Memory defaults to cores x DefMemPerCPU.
case "${1:-}" in
    v100)       RES=(--partition=gpu_v100       --cpus-per-task=4  --time=01:00:00) ;;
    a100)       RES=(--partition=gpu_a100       --cpus-per-task=18 --time=01:00:00) ;;
    a100-debug) RES=(--partition=gpu_a100_debug --cpus-per-task=16 --time=00:59:00) ;;
    *) echo "usage: $0 <v100|a100|a100-debug> [extra sbatch args]"; exit 1 ;;
esac
shift
sbatch --account="$ACCOUNT" "${RES[@]}" "$@" slurm/smoketest.slurm
