from omegaconf import DictConfig
from .datamodules import load_disaster_tweets


def get_dataset(cfg: DictConfig):
    return load_disaster_tweets(cfg.train_csv, cfg.text_col, cfg.label_col)
