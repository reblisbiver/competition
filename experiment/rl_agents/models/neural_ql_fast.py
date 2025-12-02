#!/usr/bin/env python3
"""
Fast Neural Q-Learning Agent
A lightweight version that uses learned patterns without PyTorch overhead.
Approximates the neural network's behavior using probabilistic rules.

Usage: python3 neural_ql_fast.py <user_id> <last_action> <last_reward>
Output: LEFT or RIGHT
"""
import sys
import os
import json
import random

STATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'neural_ql', 'agent_states')
os.makedirs(STATE_DIR, exist_ok=True)

LEFT_BIAS = 0.62
STAY_PROB = 0.66
WIN_STAY_BOOST = 0.15
LOSE_SHIFT_BOOST = 0.10

def get_state_path(user_id):
    return os.path.join(STATE_DIR, f"fast_state_{user_id}.json")

def load_state(user_id, reset=False):
    path = get_state_path(user_id)
    if reset and os.path.exists(path):
        os.remove(path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {
        'last_action': None,
        'last_reward': None,
        'left_rewards': 0,
        'right_rewards': 0,
        'left_count': 0,
        'right_count': 0,
        'trial': 0
    }

def save_state(user_id, state):
    with open(get_state_path(user_id), 'w') as f:
        json.dump(state, f)

def decide(state):
    trial = state['trial']
    last_action = state['last_action']
    last_reward = state['last_reward']
    
    if trial == 0 or last_action is None:
        return 'LEFT' if random.random() < LEFT_BIAS else 'RIGHT'
    
    stay_prob = STAY_PROB
    
    if last_reward == 1:
        stay_prob += WIN_STAY_BOOST
    else:
        stay_prob -= LOSE_SHIFT_BOOST
    
    left_count = max(state['left_count'], 1)
    right_count = max(state['right_count'], 1)
    left_rate = state['left_rewards'] / left_count
    right_rate = state['right_rewards'] / right_count
    
    if abs(left_rate - right_rate) > 0.1:
        if left_rate > right_rate:
            stay_prob = stay_prob if last_action == 'LEFT' else (1 - stay_prob)
        else:
            stay_prob = stay_prob if last_action == 'RIGHT' else (1 - stay_prob)
    
    stay_prob = max(0.1, min(0.9, stay_prob))
    
    if random.random() < stay_prob:
        return last_action
    else:
        return 'RIGHT' if last_action == 'LEFT' else 'LEFT'

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 neural_ql_fast.py <user_id> <last_action> <last_reward>")
        sys.exit(1)
    
    user_id = sys.argv[1]
    last_action_arg = sys.argv[2]
    last_reward_arg = sys.argv[3]
    
    reset = last_action_arg == "None" and last_reward_arg == "None"
    state = load_state(user_id, reset=reset)
    
    if not reset and last_action_arg != "None":
        last_action = last_action_arg.upper()
        last_reward = float(last_reward_arg) if last_reward_arg != "None" else 0
        
        state['last_action'] = last_action
        state['last_reward'] = last_reward
        
        if last_action == 'LEFT':
            state['left_count'] += 1
            state['left_rewards'] += last_reward
        else:
            state['right_count'] += 1
            state['right_rewards'] += last_reward
    
    action = decide(state)
    state['trial'] += 1
    save_state(user_id, state)
    
    print(action)

if __name__ == "__main__":
    main()
