#!/usr/bin/env python3
"""
Neural Q-Learning Agent for Choice Experiment
Uses a trained neural network to make decisions that mimic human behavior.

Usage: python3 neural_ql_agent.py <user_id> <last_action> <last_reward>
Output: LEFT or RIGHT
"""
import sys
import os
import json
import numpy as np

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "trained_models")
STATE_DIR = os.path.join(SCRIPT_DIR, "agent_states")
os.makedirs(STATE_DIR, exist_ok=True)

SEQUENCE_LENGTH = 10
DEVICE = torch.device("cpu")


def load_model():
    """Load the latest trained model."""
    latest_file = os.path.join(MODEL_DIR, "latest_model.txt")
    
    if not os.path.exists(latest_file):
        return None, None
        
    with open(latest_file, 'r') as f:
        model_name = f.read().strip()
        
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    if not os.path.exists(model_path):
        return None, None
        
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    from model import NeuralQLModel, ModelEnsemble
    
    model_kwargs = checkpoint['model_kwargs']
    use_ensemble = checkpoint.get('use_ensemble', False)
    
    if use_ensemble:
        num_models = checkpoint.get('num_ensemble_models', 3)
        model = ModelEnsemble(num_models=num_models, **model_kwargs)
    else:
        model = NeuralQLModel(**model_kwargs)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    seq_length = checkpoint.get('sequence_length', SEQUENCE_LENGTH)
    
    return model, seq_length


def get_fresh_state() -> dict:
    """Get a fresh/empty agent state."""
    return {
        'actions': [],
        'rewards': [],
        'trial_count': 0,
        'left_count': 0,
        'right_count': 0,
        'left_reward': 0.0,
        'right_reward': 0.0,
        'total_reward': 0.0
    }

def load_agent_state(user_id: str, reset: bool = False) -> dict:
    """Load agent state for a user."""
    state_path = os.path.join(STATE_DIR, f"state_{user_id}.json")
    
    if reset:
        if os.path.exists(state_path):
            os.remove(state_path)
        return get_fresh_state()
    
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
            
    return get_fresh_state()

def cleanup_old_states(max_age_hours: int = 24):
    """Clean up state files older than max_age_hours."""
    import time
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(STATE_DIR):
        if filename.startswith('state_') and filename.endswith('.json'):
            filepath = os.path.join(STATE_DIR, filename)
            try:
                if now - os.path.getmtime(filepath) > max_age_seconds:
                    os.remove(filepath)
            except:
                pass


def save_agent_state(user_id: str, state: dict):
    """Save agent state for a user."""
    state_path = os.path.join(STATE_DIR, f"state_{user_id}.json")
    with open(state_path, 'w') as f:
        json.dump(state, f)


def update_state(state: dict, action: str, reward: float) -> dict:
    """Update state with new action and reward."""
    action_idx = 0 if action == "LEFT" else 1
    
    state['actions'].append(action_idx)
    state['rewards'].append(reward)
    state['trial_count'] += 1
    state['total_reward'] += reward
    
    if action == "LEFT":
        state['left_count'] += 1
        state['left_reward'] += reward
    else:
        state['right_count'] += 1
        state['right_reward'] += reward
        
    max_history = 100
    if len(state['actions']) > max_history:
        state['actions'] = state['actions'][-max_history:]
        state['rewards'] = state['rewards'][-max_history:]
        
    return state


def prepare_input(state: dict, seq_length: int) -> tuple:
    """Prepare input tensors for the model."""
    actions = state['actions']
    rewards = state['rewards']
    
    seq_actions = np.zeros(seq_length)
    seq_rewards = np.zeros(seq_length)
    seq_rt = np.zeros(seq_length)
    
    history_len = min(len(actions), seq_length)
    if history_len > 0:
        offset = seq_length - history_len
        seq_actions[offset:] = actions[-history_len:]
        seq_rewards[offset:] = rewards[-history_len:]
        
    sequence = np.stack([seq_actions, seq_rewards, seq_rt], axis=-1)
    
    total_trials = state['trial_count']
    left_count = state['left_count']
    right_count = state['right_count']
    left_reward = state['left_reward']
    right_reward = state['right_reward']
    
    left_rate = left_count / total_trials if total_trials > 0 else 0.5
    right_rate = right_count / total_trials if total_trials > 0 else 0.5
    left_avg = left_reward / left_count if left_count > 0 else 0.5
    right_avg = right_reward / right_count if right_count > 0 else 0.5
    avg_reward = state['total_reward'] / total_trials if total_trials > 0 else 0.5
    
    global_features = np.array([
        total_trials / 100.0,
        left_rate,
        right_rate,
        left_avg,
        right_avg,
        avg_reward
    ])
    
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
    global_tensor = torch.FloatTensor(global_features).unsqueeze(0)
    
    return sequence_tensor, global_tensor


def fallback_action(state: dict) -> str:
    """Simple Q-learning fallback when no model is available."""
    import random
    
    epsilon = 0.1
    if random.random() < epsilon:
        return random.choice(["LEFT", "RIGHT"])
        
    left_count = state['left_count']
    right_count = state['right_count']
    left_reward = state['left_reward']
    right_reward = state['right_reward']
    
    left_q = left_reward / left_count if left_count > 0 else 0.5
    right_q = right_reward / right_count if right_count > 0 else 0.5
    
    return "LEFT" if left_q >= right_q else "RIGHT"


def main():
    if len(sys.argv) < 4:
        print("LEFT")
        return
        
    user_id = sys.argv[1]
    last_action = sys.argv[2] if sys.argv[2] not in ["None", "null", ""] else None
    
    try:
        last_reward = float(sys.argv[3]) if sys.argv[3] not in ["None", "null", ""] else 0.0
    except:
        last_reward = 0.0
    
    import random
    if random.random() < 0.01:
        cleanup_old_states(max_age_hours=24)
    
    reset_session = (last_action is None and last_reward == 0.0)
    state = load_agent_state(user_id, reset=reset_session)
    
    if last_action is not None:
        state = update_state(state, last_action, last_reward)
        save_agent_state(user_id, state)
        
    model, seq_length = load_model()
    
    if model is None:
        action = fallback_action(state)
        print(action)
        return
        
    sequence, global_features = prepare_input(state, seq_length)
    
    action_idx = model.get_action(sequence, global_features, use_policy=True, epsilon=0.05)
    action = "LEFT" if action_idx == 0 else "RIGHT"
    
    print(action)


if __name__ == "__main__":
    main()
