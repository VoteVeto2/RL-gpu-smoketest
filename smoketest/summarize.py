"""Print a table of every summary.json under the given run dirs, e.g. to compare V100 vs A100:
python -m smoketest.summarize $VSC_SCRATCH/rl-smoketest/*
"""

import glob
import json
import os
import sys

COLS = ["slurm_job_id", "stage", "gpu", "precision", "steps", "runtime_s", "s_per_step", "peak_mem_gb", "first_loss", "last_loss", "last_reward"]


def main():
    rows = []
    for d in sys.argv[1:]:
        for path in sorted(glob.glob(os.path.join(d, "**", "summary.json"), recursive=True)):
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        print("no summary.json found")
        return
    table = [COLS] + [["-" if r.get(c) is None else (f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])) for c in COLS] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(COLS))]
    for row in table:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


if __name__ == "__main__":
    main()
