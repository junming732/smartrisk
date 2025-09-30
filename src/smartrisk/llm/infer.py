from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def classify(text: str, model_dir: str = "artifacts/tinyllama-qlora") -> str:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompt = f"Tweet: {text}\nLabel:"
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=3)
    gen = tok.decode(out[0], skip_special_tokens=True).strip().lower()
    return "yes" if gen.endswith("yes") else "no"
