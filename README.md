# Fly-Brain Chess Agent

This folder adds a trainable chess agent on top of the Drosophila fly-brain connectome model.

## Setup
1. Obtain the fly-brain data files (`2025_Completeness_783.csv`, `2025_Connectivity_783.parquet`) from the upstream repository and place them in `data/`.
2. Install dependencies:
   - `pip install torch chess python-chess`
   - Stockfish binary available on PATH (Colab: `apt-get install -y stockfish`)
3. (Optional) Build cached weights by running any benchmark; weights are cached in `data/weight_*.pkl`.

## Train
```bash
cd code
python chess_train.py --data-dir ../data --device cuda --sim-ms 50 --games 50 --train-backbone
```

## Play
```bash
cd code
python chess_play.py --checkpoint ../checkpoints/chess_agent_game50.pt --opponent stockfish
```

## Notes
- Default simulation window is 50 ms (500 timesteps).
- Reward combines Stockfish centipawn delta with terminal outcome.
- Backbone weights are frozen unless `--train-backbone` is set.
