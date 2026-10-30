#!/bin/bash
# One-time setup, run on the LOGIN node (it has internet; downloads go to $VSC_DATA).
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    export PATH=$HOME/.local/bin:$PATH
fi

uv venv --python 3.12 "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt

# Pre-download models + data so jobs can run with HF_HUB_OFFLINE=1.
# The login node caps thread count, hence the small thread limits.
export OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4
hf download Qwen/Qwen3.5-0.8B
hf download Qwen/Qwen3.5-2B
python -c 'from datasets import load_dataset; load_dataset("openai/gsm8k", "main")'
echo "Setup done. Venv: $VENV_DIR  HF cache: $HF_HOME"
