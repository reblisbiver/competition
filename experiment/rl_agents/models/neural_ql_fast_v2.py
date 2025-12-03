#!/usr/bin/env python3
"""
Fast Neural Q-Learning Agent v2
Enhanced version with additional behavioral rules derived from human data.

Rules (with configurable toggles for ablation study):
1. LEFT_BIAS: Initial preference for left side (62%)
2. BASE_STAY: Base consistency probability (66%)
3. WIN_STAY: Boost after winning (+15%)
4. LOSE_SHIFT: Reduction after losing (-10%)
5. STREAK_WIN: Additional boost for consecutive wins (+10% per streak)
6. MOMENTUM: Stay more likely after previous stay (+20%)
7. RECENCY: Adjust based on recent reward rate (+/-15%)
8. PHASE: Increase stay probability over time (+10% late game)

Usage: python3 neural_ql_fast_v2.py <user_id> <last_action> <last_reward>
Output: LEFT or RIGHT
"""
import sys
import os
import json
import random
from typing import Dict, List, Optional

STATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'neural_ql', 'agent_states')
os.makedirs(STATE_DIR, exist_ok=True)

RULE_PARAMS = {
    'LEFT_BIAS': 0.62,
    'BASE_STAY': 0.66,
    'WIN_STAY_BOOST': 0.15,
    'LOSE_SHIFT_BOOST': 0.00,
    'STREAK_WIN_BOOST': 0.05,
    'MOMENTUM_BOOST': 0.20,
    'RECENCY_WEIGHT': 0.05,
    'PHASE_BOOST': 0.10,
    'RECENCY_WINDOW': 5,
    'TOTAL_TRIALS': 100,
}

RULE_ENABLED = {
    'LEFT_BIAS': True,
    'BASE_STAY': True,
    'WIN_STAY': True,
    'LOSE_SHIFT': True,
    'STREAK_WIN': True,
    'MOMENTUM': True,
    'RECENCY': True,
    'PHASE': True,
}


def get_state_path(user_id: str) -> str:
    return os.path.join(STATE_DIR, f"fast_v2_state_{user_id}.json")


def load_state(user_id: str, reset: bool = False) -> Dict:
    path = get_state_path(user_id)
    if reset and os.path.exists(path):
        os.remove(path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {
        'last_action': None,
        'last_reward': None,
        'prev_stayed': None,
        'left_rewards': 0,
        'right_rewards': 0,
        'left_count': 0,
        'right_count': 0,
        'trial': 0,
        'reward_history': [],
        'action_history': [],
        'win_streak': 0,
    }


def save_state(user_id: str, state: Dict) -> None:
    with open(get_state_path(user_id), 'w') as f:
        json.dump(state, f)


def decide(state: Dict, rules_enabled: Optional[Dict] = None) -> str:
    """
    Make a decision based on enabled rules.
    
    Returns:
        'LEFT' or 'RIGHT'
    """
    if rules_enabled is None:
        rules_enabled = RULE_ENABLED
    
    trial = state['trial']
    last_action = state['last_action']
    last_reward = state['last_reward']
    
    if trial == 0 or last_action is None:
        if rules_enabled.get('LEFT_BIAS', True):
            return 'LEFT' if random.random() < RULE_PARAMS['LEFT_BIAS'] else 'RIGHT'
        else:
            return 'LEFT' if random.random() < 0.5 else 'RIGHT'
    
    stay_prob = RULE_PARAMS['BASE_STAY'] if rules_enabled.get('BASE_STAY', True) else 0.5
    
    if rules_enabled.get('WIN_STAY', True) and last_reward == 1:
        stay_prob += RULE_PARAMS['WIN_STAY_BOOST']
    
    if rules_enabled.get('LOSE_SHIFT', True) and last_reward == 0:
        stay_prob -= RULE_PARAMS['LOSE_SHIFT_BOOST']
    
    if rules_enabled.get('STREAK_WIN', True):
        win_streak = state.get('win_streak', 0)
        if win_streak >= 2:
            stay_prob += RULE_PARAMS['STREAK_WIN_BOOST'] * min(win_streak - 1, 3)
    
    if rules_enabled.get('MOMENTUM', True):
        prev_stayed = state.get('prev_stayed')
        if prev_stayed is not None:
            if prev_stayed:
                stay_prob += RULE_PARAMS['MOMENTUM_BOOST']
            else:
                stay_prob -= RULE_PARAMS['MOMENTUM_BOOST']
    
    if rules_enabled.get('RECENCY', True):
        reward_history = state.get('reward_history', [])
        window = RULE_PARAMS['RECENCY_WINDOW']
        if len(reward_history) >= window:
            recent_rate = sum(reward_history[-window:]) / window
            stay_prob += (recent_rate - 0.5) * RULE_PARAMS['RECENCY_WEIGHT'] * 2
    
    if rules_enabled.get('PHASE', True):
        progress = trial / RULE_PARAMS['TOTAL_TRIALS']
        if progress > 0.67:
            stay_prob += RULE_PARAMS['PHASE_BOOST']
    
    stay_prob = max(0.05, min(0.95, stay_prob))
    
    if random.random() < stay_prob:
        return last_action
    else:
        return 'RIGHT' if last_action == 'LEFT' else 'LEFT'


def update_state(state: Dict, action: str, reward: float) -> Dict:
    """Update state after an action is taken."""
    last_action = state.get('last_action')
    
    if last_action is not None:
        state['prev_stayed'] = (action == last_action)
    
    if reward == 1:
        if state.get('last_reward') == 1:
            state['win_streak'] = state.get('win_streak', 0) + 1
        else:
            state['win_streak'] = 1
    else:
        state['win_streak'] = 0
    
    state['last_action'] = action
    state['last_reward'] = reward
    state['reward_history'].append(reward)
    state['action_history'].append(0 if action == 'LEFT' else 1)
    
    if action == 'LEFT':
        state['left_count'] += 1
        state['left_rewards'] += reward
    else:
        state['right_count'] += 1
        state['right_rewards'] += reward
    
    state['trial'] += 1
    
    return state


class FastModelV2:
    """Wrapper class for evaluation compatibility."""
    
    def __init__(self, rules_enabled: Optional[Dict] = None):
        self.rules_enabled = rules_enabled if rules_enabled else RULE_ENABLED.copy()
        self.state = None
        self.reset()
    
    def reset(self):
        self.state = {
            'last_action': None,
            'last_reward': None,
            'prev_stayed': None,
            'left_rewards': 0,
            'right_rewards': 0,
            'left_count': 0,
            'right_count': 0,
            'trial': 0,
            'reward_history': [],
            'action_history': [],
            'win_streak': 0,
        }
    
    def predict(self) -> int:
        """Predict next action (0=LEFT, 1=RIGHT)."""
        action = decide(self.state, self.rules_enabled)
        return 0 if action == 'LEFT' else 1
    
    def update(self, action: int, reward: float):
        """Update state after human action."""
        action_str = 'LEFT' if action == 0 else 'RIGHT'
        self.state = update_state(self.state, action_str, reward)


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 neural_ql_fast_v2.py <user_id> <last_action> <last_reward>")
        sys.exit(1)
    
    user_id = sys.argv[1]
    last_action_arg = sys.argv[2]
    last_reward_arg = sys.argv[3]
    
    reset = last_action_arg == "None" and last_reward_arg == "None"
    state = load_state(user_id, reset=reset)
    
    if not reset and last_action_arg != "None":
        last_action = last_action_arg.upper()
        last_reward = float(last_reward_arg) if last_reward_arg != "None" else 0
        state = update_state(state, last_action, last_reward)
    
    action = decide(state)
    save_state(user_id, state)
    
    print(action)


if __name__ == "__main__":
    main()
