# Adaptive sequence agent experiments

This folder contains a small experimentation harness to test simple adaptive reward-allocator policies
that interact with the existing `neural_ql_agent` as the simulated human. The goal is to quickly compare
architectures (MLP, LSTM, GRU) using a lightweight REINFORCE training loop.

Files:
- `models.py`: simple `MLPPolicy`, `LSTMPolicy`, `GRUPolicy` implementations.
- `train_adaptive.py`: quick runner that imports `experiment.rl_agents.neural_ql.neural_ql_agent` and trains
  each architecture for a few episodes, saving short summaries under `output/`.

Notes & Limitations:
- The harness expects the repository's `neural_ql_agent` to provide a callable API to simulate a single trial.
  The script tries several common attribute names (`simulate_single_trial`, `get_action`, `NeuralQLAgent`).
- This is a lightweight prototype for architecture comparison. If the real agent API differs, the script will
  raise an error and suggest how to adapt the call site.
- The training objective is simple REINFORCE optimizing fraction of trials the agent chooses the target.

Quick start (from repo root):
```
python experiment/sequence_adaptive_agent/train_adaptive.py --episodes 6 --trials 50 --lr 1e-3
```
