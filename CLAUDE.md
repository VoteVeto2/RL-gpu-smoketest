# CLAUDE.md

**Before starting any task, invoke the `karpathy-guidelines` skill** (`.claude/skills/karpathy-guidelines/SKILL.md`)
and follow it throughout: think before coding, simplicity first, surgical changes, goal-driven verification.

## Project

Tiny LLM post-training smoketest (SFT/GRPO with LoRA, on-policy distillation Qwen3.5-2B -> 0.8B, via TRL)
whose only purpose is learning to schedule GPUs on VSC (KU Leuven, cluster `wice`) and comparing V100 vs A100.
See `README.md` for layout and the GPU booking tutorial.

## Workflow

- Commit early and often, and push to `origin` (https://github.com/VoteVeto2/RL-gpu-smoketest.git, branch `main`).
- Verify changes on a real GPU: `slurm/submit.sh v100|a100|a100-debug` (add `--export=ALL,STAGES=<stage>` to run one stage).
  A stage only passes if its loss is finite and grad_norm is non-zero at some step.

## VSC gotchas

- Login node: no training; it caps threads, so set `OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4` for any local python.
- All Slurm commands need `-M wice`; jobs need `--account=intro_vsc38501`.
- `$VSC_HOME` is 3 GB: venv + HF cache live on `$VSC_DATA` (see `env.sh`), run outputs on `$VSC_SCRATCH/rl-smoketest/<jobid>/`.
- Jobs run with `HF_HUB_OFFLINE=1`; new models/datasets must be pre-downloaded on the login node first.
- V100 (sm_70) has no bf16 -> fp16 AMP; keep torch on the cu126 wheel so one venv covers sm_70 and sm_80.
