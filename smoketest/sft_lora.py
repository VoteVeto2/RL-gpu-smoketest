"""Stage 1: supervised fine-tuning with LoRA on GSM8K (student model)."""

from trl import SFTConfig, SFTTrainer

from smoketest.common import STUDENT, gsm8k, load_model, load_tokenizer, lora_config, parse_args, precision, train_and_report, trainer_kwargs


def main():
    args = parse_args(default_steps=20)
    prec = precision()
    config = SFTConfig(
        **trainer_kwargs(args, prec),
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        max_length=512,
    )
    trainer = SFTTrainer(
        model=load_model(STUDENT, prec["dtype"]),
        args=config,
        train_dataset=gsm8k(args.num_samples).remove_columns("solution"),
        processing_class=load_tokenizer(),
        peft_config=lora_config(),
    )
    train_and_report(trainer, "sft_lora", args, prec)


if __name__ == "__main__":
    main()
