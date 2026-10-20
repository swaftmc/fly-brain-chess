#!/usr/bin/env bash
set -euo pipefail

# Colab setup for the fly-brain chess agent.
# Run from the repository root after cloning.

python -m pip install -q chess python-chess

if ! command -v stockfish >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq stockfish
fi

echo "Checking connectome data..."
test -f data/2025_Completeness_783.csv
test -f data/2025_Connectivity_783.parquet
echo "Setup complete."
