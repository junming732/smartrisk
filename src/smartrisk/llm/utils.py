from peft import LoraConfig, get_peft_model


def make_lora_config(r: int, alpha: int, dropout: float, target_modules: list[str]):
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def wrap_with_lora(model, lora_cfg):
    return get_peft_model(model, lora_cfg)
