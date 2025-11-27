#!/usr/bin/env python3
"""
Upper Confidence Bound (UCB) Agent
Balances exploration and exploitation using confidence intervals.
"""
import sys
import os
import math

STATE_DIR = os.path.join(os.path.dirname(__file__), "../q_tables")
os.makedirs(STATE_DIR, exist_ok=True)

def load_state(user_id):
    state_path = os.path.join(STATE_DIR, f"ucb_{user_id}.txt")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                lines = f.readlines()
                state = {}
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) == 3:
                        state[parts[0]] = {"count": int(parts[1]), "total": float(parts[2])}
                return state
        except:
            pass
    return {"LEFT": {"count": 0, "total": 0.0}, "RIGHT": {"count": 0, "total": 0.0}}

def save_state(user_id, state):
    state_path = os.path.join(STATE_DIR, f"ucb_{user_id}.txt")
    with open(state_path, "w") as f:
        for action, data in state.items():
            f.write(f"{action},{data['count']},{data['total']}\n")

def ucb_value(action_data, total_count, c=2.0):
    if action_data["count"] == 0:
        return float('inf')
    avg = action_data["total"] / action_data["count"]
    exploration = c * math.sqrt(math.log(total_count + 1) / action_data["count"])
    return avg + exploration

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
    
    state = load_state(user_id)
    
    if last_action is not None and last_action in state:
        state[last_action]["count"] += 1
        state[last_action]["total"] += last_reward
        save_state(user_id, state)
    
    total_count = sum(d["count"] for d in state.values())
    
    left_ucb = ucb_value(state["LEFT"], total_count)
    right_ucb = ucb_value(state["RIGHT"], total_count)
    
    if left_ucb > right_ucb:
        print("LEFT")
    elif right_ucb > left_ucb:
        print("RIGHT")
    else:
        import random
        print(random.choice(["LEFT", "RIGHT"]))

if __name__ == "__main__":
    main()
