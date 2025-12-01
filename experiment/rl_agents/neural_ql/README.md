# Neural Network Enhanced Q-Learning Model

This module implements a neural network enhanced Q-learning model that learns from human behavioral data to make decisions that closely mimic human choice patterns.

## Architecture Overview

The model uses a sophisticated architecture combining several components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Layer                                   │
│  ┌──────────────────────┐  ┌─────────────────────────────────┐  │
│  │  Sequence Features    │  │  Global Features                │  │
│  │  - Action history     │  │  - Trial count                  │  │
│  │  - Reward history     │  │  - Left/Right choice rates     │  │
│  │  - RT (normalized)    │  │  - Avg reward per side         │  │
│  │  [seq_len × 3]        │  │  [6]                            │  │
│  └──────────┬───────────┘  └──────────────┬──────────────────┘  │
│             │                              │                     │
│             ▼                              ▼                     │
│  ┌──────────────────────┐     ┌─────────────────────────────┐   │
│  │  Input Projection     │     │  Global Feature Network     │   │
│  │  Linear + LayerNorm   │     │  2-layer MLP                │   │
│  │  + ReLU + Dropout     │     │  Linear → ReLU → Linear     │   │
│  └──────────┬───────────┘     └──────────────┬──────────────┘   │
│             │                                 │                  │
│             ▼                                 │                  │
│  ┌──────────────────────┐                    │                  │
│  │  LSTM Encoder         │                    │                  │
│  │  2 layers, hidden=128 │                    │                  │
│  │  Captures temporal    │                    │                  │
│  │  dependencies         │                    │                  │
│  └──────────┬───────────┘                    │                  │
│             │                                 │                  │
│             ▼                                 │                  │
│  ┌──────────────────────┐                    │                  │
│  │  Self-Attention       │                    │                  │
│  │  Multi-head (4 heads) │                    │                  │
│  │  + Residual + Norm    │                    │                  │
│  └──────────┬───────────┘                    │                  │
│             │                                 │                  │
│             └──────────────┬─────────────────┘                  │
│                            ▼                                     │
│             ┌──────────────────────┐                            │
│             │  Fusion Layer        │                            │
│             │  Concatenate +       │                            │
│             │  2-layer MLP         │                            │
│             └──────────┬───────────┘                            │
│                        │                                        │
│          ┌─────────────┴─────────────┐                         │
│          ▼                           ▼                          │
│  ┌──────────────────┐     ┌──────────────────┐                 │
│  │  Q-Value Head     │     │  Policy Head      │                 │
│  │  Output: Q(s,a)   │     │  Output: π(a|s)   │                 │
│  │  for LEFT/RIGHT   │     │  with temperature │                 │
│  └──────────────────┘     └──────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Architecture?

### 1. LSTM for Temporal Dependencies
Human decisions are heavily influenced by recent history. The LSTM captures:
- Win-stay, lose-shift patterns
- Momentum in choices
- Recent reward streaks

### 2. Self-Attention for Selective Memory
Not all past trials are equally important. Attention mechanism:
- Learns which historical events matter most
- Can focus on particularly rewarding or punishing trials
- Handles variable-length history gracefully

### 3. Global Features for Context
Aggregate statistics provide broader context:
- Overall bias towards left/right
- Success rate per side
- Total experience level

### 4. Dual Output Heads
- **Q-Value Head**: Traditional RL output for value estimation
- **Policy Head**: Direct action probability for behavioral cloning
- Temperature parameter: Controls exploration/exploitation balance

## Files

- `data_loader.py` - Loads and preprocesses human behavioral data
- `model.py` - Neural network architecture definitions
- `train.py` - Training script with early stopping and learning rate scheduling
- `neural_ql_agent.py` - Agent for use in experiments
- `evaluate.py` - Model evaluation and visualization tools

## Usage

### 1. Prepare Human Data

Place CSV files in the `results/` directory. Expected format:
```csv
trial_number, time, schedule_type, schedule_name, is_biased_choice, side_choice, RT, observed_reward, unobserved_reward, biased_reward, unbiased_reward
0, 2025-11-27 10:34:55am, STATIC, random_0, false, LEFT, 938, 1, 0, 0, 1
...
```

You can also place additional data in `experiment/rl_agents/neural_ql/human_data/`.

### 2. Train the Model

```bash
cd experiment/rl_agents/neural_ql
source ../../../.venv/bin/activate
python train.py --epochs 100 --lr 0.001 --batch_size 32
```

Options:
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size (default: 32)
- `--hidden_dim`: Hidden dimension (default: 128)
- `--lstm_layers`: Number of LSTM layers (default: 2)
- `--seq_length`: History sequence length (default: 10)
- `--patience`: Early stopping patience (default: 15)
- `--ensemble`: Use model ensemble for better robustness
- `--num_models`: Number of ensemble models (default: 3)

### 3. Evaluate the Model

```bash
python evaluate.py --visualize
```

This generates:
- Overall accuracy metrics
- Choice distribution comparison
- Behavioral pattern analysis
- Temporal accuracy analysis
- Visualization plots

### 4. Use in Experiments

The trained model is automatically used by `neural_ql_agent.py`:

```bash
python neural_ql_agent.py <user_id> <last_action> <last_reward>
```

Or through the RL Dashboard with model selection.

## Training Objective

The model is trained using a combination of:

1. **Behavioral Cloning (Primary)**
   - Cross-entropy loss between model predictions and human actions
   - Directly learns to imitate human decision patterns

2. **Q-Learning Regularization (Optional)**
   - MSE loss between predicted Q-values and observed rewards
   - Helps model understand reward structure

Loss = BC_weight × CrossEntropy(π, human_action) + Q_weight × MSE(Q, reward)

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 128 | LSTM and attention hidden size |
| num_lstm_layers | 2 | Depth of LSTM encoder |
| num_attention_heads | 4 | Multi-head attention heads |
| sequence_length | 10 | Number of past trials to consider |
| dropout | 0.2 | Regularization strength |
| temperature | 1.0 | Softmax temperature (learned) |

## Evaluation Metrics

- **Overall Accuracy**: Match rate with human choices
- **Choice Distribution Error**: Difference in left/right preference
- **Switch Rate Comparison**: How often agent switches vs. human
- **Temporal Analysis**: Accuracy in early vs. late trials
- **Per-Session Accuracy**: Consistency across different subjects

## Tips for Best Results

1. **More Data = Better**: The model benefits from larger datasets
2. **Diverse Subjects**: Include data from various human subjects
3. **Multiple Schedules**: Train on data from different reward schedules
4. **Ensemble Models**: Use `--ensemble` for improved robustness
5. **Longer Sequences**: Increase `--seq_length` if subjects show long-term patterns
