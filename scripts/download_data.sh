#!/usr/bin/env bash
set -euo pipefail

# Requires Kaggle API credentials at ~/.kaggle/kaggle.json
# https://www.kaggle.com/docs/api
# Installs to ./data/raw/kaggle-disaster-tweets

mkdir -p data/raw/kaggle-disaster-tweets
cd data/raw/kaggle-disaster-tweets

if ! command -v kaggle &> /dev/null; then
  python -m pip install kaggle
fi

kaggle competitions download -c nlp-getting-started
unzip -o nlp-getting-started.zip
rm -f nlp-getting-started.zip
ls -lh
