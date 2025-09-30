from dataclasses import dataclass
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class TabularConfig:
    train_csv: str = "data/raw/kaggle-disaster-tweets/train.csv"


def run_tabular_baseline(cfg: TabularConfig | None = None):
    cfg = cfg or TabularConfig()
    df = pd.read_csv(cfg.train_csv)
    X = df[["text"]]
    y = df["target"]

    pre = ColumnTransformer(
        [
            (
                "text",
                OneHotEncoder(handle_unknown="ignore", max_categories=1000),
                ["text"],
            )
        ]
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    pipe.fit(X, y)
    preds = pipe.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, preds)

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("tabular-baseline")
    with mlflow.start_run():
        mlflow.log_metric("auc", auc)
