import os
import mlflow
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import IntervalStrategy
from peft import prepare_model_for_kbit_training
from smartrisk.data.kaggle_disaster import get_dataset
from smartrisk.features.text import build_tokenizer
from smartrisk.llm.utils import make_lora_config, wrap_with_lora


def run_qlora():
    cfg = OmegaConf.load("src/smartrisk/configs/config.yaml")
    data_cfg = OmegaConf.load("src/smartrisk/configs/data/kaggle_disaster.yaml")
    train_cfg = OmegaConf.load(
        "src/smartrisk/configs/train/llm_text_classification.yaml"
    )
    llm_cfg = OmegaConf.load("src/smartrisk/configs/llm/tinyllama_qlora.yaml")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment("tinyllama-qlora-disaster")

    ds = get_dataset(data_cfg)

    def to_sft(ex):
        answer = "yes" if ex["label"] == 1 else "no"
        return {"text": f"Tweet: {ex['text']}\nLabel: {answer}"}

    ds = ds.map(to_sft)
    ds = ds.train_test_split(test_size=0.1, seed=42)
    tok = build_tokenizer(llm_cfg.base_model)

    tokenized = ds.map(lambda e: tok(e["text"], truncation=True), batched=True)

    model = AutoModelForCausalLM.from_pretrained(
        llm_cfg.base_model, load_in_4bit=(llm_cfg.bits == 4), device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    lora_cfg = make_lora_config(
        r=llm_cfg.lora_r,
        alpha=llm_cfg.lora_alpha,
        dropout=llm_cfg.lora_dropout,
        target_modules=list(llm_cfg.target_modules),
    )
    model = wrap_with_lora(model, lora_cfg)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir="outputs/tinyllama-qlora",
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_steps=train_cfg.logging_steps,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=train_cfg.eval_steps,
        save_steps=train_cfg.save_steps,
        bf16=train_cfg.bf16,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()

    save_dir = "artifacts/tinyllama-qlora"
    os.makedirs(save_dir, exist_ok=True)
    trainer.model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    mlflow.log_params(
        {
            "base_model": llm_cfg.base_model,
            "lora_r": llm_cfg.lora_r,
            "lora_alpha": llm_cfg.lora_alpha,
            "lora_dropout": llm_cfg.lora_dropout,
        }
    )
    mlflow.log_artifacts(save_dir, artifact_path="model")
