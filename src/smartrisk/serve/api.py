from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="SmartRisk API")


class PredictIn(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global model, tok, device
    model_dir = "artifacts/tinyllama-qlora"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


@app.post("/classify")
def classify(inp: PredictIn):
    prompt = f"Tweet: {inp.text}\nLabel:"
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=3)
    gen = tok.decode(out[0], skip_special_tokens=True)
    label = "yes" if gen.strip().lower().endswith("yes") else "no"
    return {"label": label}
