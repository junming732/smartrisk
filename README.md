## Usage
1. **Install and bootstrap**
```bash
python -m pip install -e .[llm]
make bootstrap
```

2. **Get the dataset** (Kaggle Disaster Tweets as an example):
```bash
./scripts/download_data.sh
```

3. **Train models**
- Tabular baseline:
```bash
python -m smartrisk.cli train.tabular
```
- LLM fine‑tuning with TinyLlama QLoRA:
```bash
python -m smartrisk.cli train.llm
```

4. **Serve predictions**
```bash
uvicorn smartrisk.serve.api:app --reload
```

---
# Project layout

```
smartrisk/
├── .github/
│   └── workflows/ci.yml
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── Makefile
├── README.md
├── pyproject.toml
├── scripts/bootstrap.sh
├── scripts/download_data.sh
└── src/smartrisk/
    ├── __init__.py
    ├── cli.py
    ├── configs/
    │   ├── config.yaml
    │   ├── data/kaggle_disaster.yaml
    │   ├── train/tabular_baseline.yaml
    │   ├── train/llm_text_classification.yaml
    │   └── llm/tinyllama_qlora.yaml
    ├── data/datamodules.py
    ├── data/kaggle_disaster.py
    ├── features/text.py
    ├── features/preprocess.py
    ├── llm/utils.py
    ├── llm/finetune_qlora.py
    ├── llm/infer.py
    ├── models/tabular.py
    ├── models/registry.py
    └── serve/api.py
