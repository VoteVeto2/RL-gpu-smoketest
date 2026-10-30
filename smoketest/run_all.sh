#!/bin/bash
# Run all smoketest stages in sequence. Used by the Slurm job, and by hand inside an interactive GPU session.
# Select stages with STAGES="sft_lora grpo_lora" bash smoketest/run_all.sh
set -u
cd "$(dirname "$0")/.."
source env.sh
STAGES=${STAGES:-"check_gpu sft_lora grpo_lora distill_onpolicy"}
RUN_DIR=${RUN_DIR:-$RUN_ROOT/${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}}
mkdir -p "$RUN_DIR"
echo "Run dir: $RUN_DIR"

status=0
for stage in $STAGES; do
    echo "================ $stage ($(date +%T)) ================"
    if ! python -m "smoketest.$stage" --output-dir "$RUN_DIR/$stage"; then
        echo "!!!!!!!! $stage FAILED"
        status=1
    fi
done
echo "================ summary ================"
python -m smoketest.summarize "$RUN_DIR"
exit $status
