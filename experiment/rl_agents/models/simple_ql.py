
''''''
Simple Q-Learning Agent
A basic epsilon-greedy Q-learning agent for the choice experiment.

Usage: python3 simple_ql.py <user_id> <last_action> <last_reward>
Output: LEFT or RIGHT

Parameters:
    user_id: Unique identifier for this experiment session
    last_action: Previous action taken (LEFT/RIGHT/None)
    last_reward: Reward received for last action (0/1/None)
"""
import sys
import os
import random

ALPHA = 0.1
EPSILON = 0.1
Q_TABLE_DIR = os.path.join(os.path.dirname(__file__), "q_tables")
os.makedirs(Q_TABLE_DIR, exist_ok=True)

def load_q_table(user_id):
    q_table_path = os.path.join(Q_TABLE_DIR, f"q_{user_id}.txt")
    if os.path.exists(q_table_path):
        try:
            with open(q_table_path, "r") as f:
                lines = f.readlines()
                return {line.split(",")[0].strip(): float(line.split(",")[1].strip()) for line in lines if "," in line}
        except:
            pass
    return {"LEFT": 0.0, "RIGHT": 0.0}

def save_q_table(user_id, q_table):
    q_table_path = os.path.join(Q_TABLE_DIR, f"q_{user_id}.txt")
    with open(q_table_path, "w") as f:
        for action, value in q_table.items():
            f.write(f"{action},{value}\n")

def select_action(q_table):
    if random.random() < EPSILON:
        return random.choice(["LEFT", "RIGHT"])
    else:
        return max(q_table.keys(), key=lambda k: q_table[k])

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
    
    q_table = load_q_table(user_id)
    
    if last_action is not None and last_action in q_table:
        old_value = q_table[last_action]
        q_table[last_action] = old_value + ALPHA * (last_reward - old_value)
        save_q_table(user_id, q_table)
    
    current_action = select_action(q_table)
    print(current_action)

if __name__ == "__main__":
    main()
