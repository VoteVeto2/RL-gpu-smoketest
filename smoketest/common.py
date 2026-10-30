"""Shared helpers: precision per GPU generation, models, data, LoRA, run summary."""

import argparse
import json
import math
import os
import socket

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDENT = os.environ.get("STUDENT_MODEL", "Qwen/Qwen3.5-0.8B")
TEACHER = os.environ.get("TEACHER_MODEL", "Qwen/Qwen3.5-2B")
# "briefly" matters: without it the 0.8B model rambles past max_completion_length, never writes
# '####', and GRPO gets all-zero rewards (zero advantage -> zero gradient).
PROMPT_SUFFIX = "Solve briefly in a few short steps, then give the final answer on the last line as '#### <number>'."


def parse_args(default_steps):
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=default_steps)
    p.add_argument("--num-samples", type=int, default=64, help="number of GSM8K training examples")
    return p.parse_args()


def precision():
    """A100 (sm_80+): bf16 weights + bf16 autocast.
    V100 (sm_70) has no bf16: fp32 master weights + fp16 autocast (classic AMP).
    Override with PRECISION=bf16|fp16|fp32."""
    mode = os.environ.get("PRECISION", "auto")
    if mode == "auto":
        if not torch.cuda.is_available():
            mode = "fp32"
        elif torch.cuda.get_device_capability()[0] >= 8:
            mode = "bf16"
        else:
            mode = "fp16"
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    return {"name": mode, "dtype": dtype, "bf16": mode == "bf16", "fp16": mode == "fp16"}


def trainer_kwargs(args, prec):
    """Config fields shared by SFTConfig / GRPOConfig / DistillationConfig."""
    return dict(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=prec["bf16"],
        fp16=prec["fp16"],
        gradient_checkpointing=True,
        seed=0,
    )


def load_model(name, dtype):
    # Qwen3.5 checkpoints are multimodal; AutoModelForCausalLM loads only the text decoder.
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")


def load_tokenizer(name=STUDENT):
    return AutoTokenizer.from_pretrained(name)


def lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def gsm8k(n):
    """GSM8K in TRL's conversational prompt/completion format, plus the numeric `solution`."""
    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{n}]")

    def fmt(ex):
        return {
            "prompt": [{"role": "user", "content": f"{ex['question']}\n{PROMPT_SUFFIX}"}],
            "completion": [{"role": "assistant", "content": ex["answer"]}],
            "solution": ex["answer"].split("####")[-1].strip().replace(",", ""),
        }

    return ds.map(fmt, remove_columns=ds.column_names)


def train_and_report(trainer, stage, args, prec):
    """Train, save the LoRA adapter, write summary.json, and fail loudly on a non-finite loss
    or when no step produced a gradient (training ran but learned nothing)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    trainer.save_model(args.output_dir)

    logs = [h for h in trainer.state.log_history if "loss" in h]
    summary = {
        "stage": stage,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "precision": prec["name"],
        "steps": result.global_step,
        "runtime_s": round(result.metrics["train_runtime"], 1),
        "s_per_step": round(result.metrics["train_runtime"] / max(result.global_step, 1), 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
        "first_loss": logs[0]["loss"] if logs else None,
        "last_loss": logs[-1]["loss"] if logs else None,
        "last_reward": logs[-1].get("reward") if logs else None,
        "max_grad_norm": max((h.get("grad_norm", 0.0) for h in logs), default=0.0),
        "node": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SMOKETEST_SUMMARY", json.dumps(summary), flush=True)

    if summary["last_loss"] is None or not math.isfinite(summary["last_loss"]):
        raise SystemExit(f"[{stage}] loss is not finite: {summary['last_loss']}")
    if not summary["max_grad_norm"] > 0:
        raise SystemExit(f"[{stage}] grad_norm was 0 at every step -- no learning signal")
