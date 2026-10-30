"""Stage 3: on-policy distillation Qwen3.5-2B (teacher) -> Qwen3.5-0.8B (student, LoRA).

The student samples completions itself; the frozen teacher scores every generated token and the
student minimises the per-token reverse KL to the teacher (beta=1.0), as in GKD / "on-policy distillation".
"""

from trl import DistillationConfig, DistillationTrainer

from smoketest.common import STUDENT, TEACHER, gsm8k, load_model, load_tokenizer, lora_config, parse_args, precision, train_and_report, trainer_kwargs


def main():
    args = parse_args(default_steps=10)
    prec = precision()
    config = DistillationConfig(
        **trainer_kwargs(args, prec),
        per_device_train_batch_size=4,
        max_completion_length=128,
        learning_rate=1e-4,
        beta=1.0,  # 0 = forward KL, 0.5 = JSD, 1 = reverse KL
        temperature=1.0,
    )
    teacher = load_model(TEACHER, prec["dtype"]).eval()
    trainer = DistillationTrainer(
        model=load_model(STUDENT, prec["dtype"]),
        teacher_model=teacher,
        args=config,
        train_dataset=gsm8k(args.num_samples).select_columns(["prompt"]),
        processing_class=load_tokenizer(),
        peft_config=lora_config(),
    )
    train_and_report(trainer, "distill_onpolicy", args, prec)


if __name__ == "__main__":
    main()
