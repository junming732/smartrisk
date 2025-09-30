from typer import Typer
from smartrisk.llm.finetune_qlora import run_qlora
from smartrisk.models.tabular import run_tabular_baseline

app = Typer(no_args_is_help=True)


@app.command(name="train.tabular")
def train_tabular_cmd():
    """Train a classical baseline on the disaster tweets dataset."""
    run_tabular_baseline()


@app.command(name="train.llm")
def train_llm_cmd():
    """Fine-tune TinyLlama with QLoRA for tweet classification."""
    run_qlora()


if __name__ == "__main__":
    app()
