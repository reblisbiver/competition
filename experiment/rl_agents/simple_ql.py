# experiment/rl_agents/simple_ql.py
import sys
import os
import random


# 参数：sys.argv[1]=user_id, sys.argv[2]=last_action, sys.argv[3]=last_reward
user_id = sys.argv[1]
last_action = sys.argv[2] if sys.argv[2] != "None" else None
last_reward = float(sys.argv[3]) if sys.argv[3] != "None" else 0.0

# 超参数（极简版）
ALPHA = 0.1    # 学习率
EPSILON = 0.1  # 探索率（10%随机选，90%选Q值大的）
Q_TABLE_DIR = os.path.join(os.path.dirname(__file__), "q_tables")
os.makedirs(Q_TABLE_DIR, exist_ok=True)
q_table_path = os.path.join(Q_TABLE_DIR, f"q_{user_id}.txt")

# 初始化/加载Q表
def load_q_table():
    if os.path.exists(q_table_path):
        with open(q_table_path, "r") as f:
            lines = f.readlines()
            return {line.split(",")[0]: float(line.split(",")[1]) for line in lines}
    else:
        return {"LEFT": 0.0, "RIGHT": 0.0}

def save_q_table(q_table):
    with open(q_table_path, "w") as f:
        for action, value in q_table.items():
            f.write(f"{action},{value}\n")

# 核心逻辑：选动作+更新Q表
q_table = load_q_table()

# 1. 更新上一次动作的Q值（如果不是第一次试次）
if last_action is not None:
    q_table[last_action] = q_table[last_action] + ALPHA * (last_reward - q_table[last_action])
    save_q_table(q_table)

# 2. ε-greedy选本次动作
if random.random() < EPSILON:
    current_action = random.choice(["LEFT", "RIGHT"])  # 探索：随机选
else:
    current_action = max(q_table.keys(), key=lambda k: q_table[k])     # 利用：选Q值大的

# 输出本次动作（给PHP接收）
print(current_action)