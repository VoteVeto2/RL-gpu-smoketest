# Source me (`source env.sh`) on the login node or inside a job.
# VSC storage layout: $VSC_HOME is tiny (3 GB) -> venv + HF cache live on $VSC_DATA,
# run outputs go to $VSC_SCRATCH (large, fast, but purged when unused).
export VENV_DIR=${VENV_DIR:-$VSC_DATA/venvs/rl-smoketest}
export HF_HOME=${HF_HOME:-$VSC_DATA/hf_home}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$VSC_DATA/.cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$VSC_DATA/.local/uv-python}
export RUN_ROOT=${RUN_ROOT:-$VSC_SCRATCH/rl-smoketest}
export TOKENIZERS_PARALLELISM=false
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi
