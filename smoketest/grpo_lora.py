"""Stage 2: GRPO with LoRA on GSM8K, rule-based rewards (student model)."""

import re

from trl import GRPOConfig, GRPOTrainer

from smoketest.common import STUDENT, gsm8k, load_model, load_tokenizer, lora_config, parse_args, precision, train_and_report, trainer_kwargs

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _final_number(text):
    tail = text.split("####")[-1]
    nums = NUMBER_RE.findall(tail)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def correctness_reward(completions, solution, **kwargs):
    rewards = []
    for completion, sol in zip(completions, solution):
        pred = _final_number(completion[0]["content"])
        try:
            rewards.append(1.0 if pred is not None and float(pred) == float(sol) else 0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs):
    return [0.2 if "####" in c[0]["content"] else 0.0 for c in completions]


def main():
    args = parse_args(default_steps=10)
    prec = precision()
    config = GRPOConfig(
        **trainer_kwargs(args, prec),
        per_device_train_batch_size=8,  # counted in completions: 2 prompts x 4 generations
        num_generations=4,
        max_completion_length=128,
        learning_rate=1e-5,
        temperature=1.0,
    )
    trainer = GRPOTrainer(
        model=load_model(STUDENT, prec["dtype"]),
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=gsm8k(args.num_samples).remove_columns("completion"),
        processing_class=load_tokenizer(),
        peft_config=lora_config(),
    )
    train_and_report(trainer, "grpo_lora", args, prec)


if __name__ == "__main__":
    main()
