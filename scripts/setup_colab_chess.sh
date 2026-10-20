#!/usr/bin/env bash
set -euo pipefail

# Colab setup for the fly-brain chess agent.
# Run from the repository root after cloning.

python -m pip install -q chess python-chess

if ! command -v stockfish >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq stockfish
fi

mkdir -p data

if [ ! -f data/2025_Completeness_783.csv ]; then
  echo "Downloading completeness data..."
  wget -q https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/data/2025_Completeness_783.csv \
    -O data/2025_Completeness_783.csv
fi

if [ ! -f data/2025_Connectivity_783.parquet ]; then
  echo "Downloading connectivity data (large)..."
  wget -q https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/data/2025_Connectivity_783.parquet \
    -O data/2025_Connectivity_783.parquet
fi

echo "Checking connectome data..."
test -f data/2025_Completeness_783.csv
test -f data/2025_Connectivity_783.parquet
echo "Setup complete."
