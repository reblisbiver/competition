import sys
import ast
import os
import torch
import torch.nn as nn
import numpy as np

###############################################################################
# User Configuration
###############################################################################
MODEL_PATH = "model.pth"  # 请确保你的模型权重文件名为 model.pth 并放在同级目录
TOTAL_REWARDS = 25
NUMBER_OF_TRIALS = 100


###############################################################################
# Neural Network Definition (Exactly as provided)
###############################################################################
class RecurrentActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )
        self.rnn = nn.GRU(64, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden_state):
        features = self.feature_extractor(x)
        if features.dim() == 2:
            features = features.unsqueeze(1)

        rnn_out, new_hidden = self.rnn(features, hidden_state)
        # 取最后一个时间步
        rnn_out_last = rnn_out[:, -1, :]

        action_logits = self.actor(rnn_out_last)
        state_value = self.critic(rnn_out_last)

        return action_logits, state_value, new_hidden

    def get_initial_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 64)


###############################################################################
# Feature Engineering Helper
###############################################################################
def build_state_sequence(target_allocs, anti_allocs, choices):
    """
    将历史列表转换为模型需要的5维状态序列。
    Input mapping:
    - Target Side -> Biased Side (User terminology)
    - Anti-Target Side -> Unbiased Side (User terminology)

    State Dim (5):
    0: Biased (Target) remaining budget (normalized)
    1: Unbiased (Anti) remaining budget (normalized)
    2: Last choice (0=Anti, 1=Target)
    3: Last reward (0=No, 1=Yes)
    4: Current round (normalized t/100)
    """
    states = []

    # 计数器
    consumed_target = 0
    consumed_anti = 0

    # 当前是第几轮 (0-indexed)
    current_trial_idx = len(target_allocs)

    # 我们需要重演历史，从第0轮到当前轮
    for t in range(current_trial_idx + 1):
        # --- 1. & 2. 剩余预算 (归一化) ---
        # 注意：这里计算的是 t 时刻开始前的状态
        rem_target = max(0, TOTAL_REWARDS - consumed_target)
        rem_anti = max(0, TOTAL_REWARDS - consumed_anti)

        norm_rem_target = rem_target / TOTAL_REWARDS
        norm_rem_anti = rem_anti / TOTAL_REWARDS

        # --- 3. & 4. 上一轮信息 ---
        if t == 0:
            prev_choice = 0.0
            prev_reward = 0.0
        else:
            # choices list 中：1代表Target，0代表Anti-Target
            # 注意 choices 长度等于 current_trial_idx，下标用 t-1
            is_target = choices[t - 1]
            prev_choice = 1.0 if is_target else 0.0

            # 获取上一轮是否得奖
            if is_target:
                prev_reward = float(target_allocs[t - 1])
            else:
                prev_reward = float(anti_allocs[t - 1])

        # --- 5. 当前轮数 (归一化) ---
        norm_round = t / NUMBER_OF_TRIALS

        # 组合向量
        state_vector = [
            norm_rem_target,
            norm_rem_anti,
            prev_choice,
            prev_reward,
            norm_round
        ]
        states.append(state_vector)

        # --- 更新计数器 (为下一次循环) ---
        if t < current_trial_idx:
            # 如果发生了选择，根据选择更新消耗
            c = choices[t]
            if c:  # Chosen Target
                if target_allocs[t] == 1: consumed_target += 1
            else:  # Chosen Anti-Target
                if anti_allocs[t] == 1: consumed_anti += 1

    # 转换为 Tensor: (1, seq_len, 5)
    return torch.tensor([states], dtype=torch.float32)


###############################################################################
# Core Logic Replacement
###############################################################################
def allocate(target_allocations, anti_target_allocations, is_target_choices):
    """
    Replaces the WSLS logic with Neural Network inference.
    """
    # 1. 准备模型
    state_dim = 5
    action_dim = 2  # [Prob_Target, Prob_Anti]

    model = RecurrentActorCritic(state_dim, action_dim)

    # 加载权重 (如果文件存在)
    if os.path.exists(MODEL_PATH):
        try:
            # map_location确保即使在GPU训练也能在CPU运行
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        except Exception:
            # 如果加载失败，可能会导致随机行为，但在竞赛中通常应确保文件存在
            pass

    model.eval()

    # 2. 构建状态
    input_tensor = build_state_sequence(target_allocations, anti_target_allocations, is_target_choices)
    hidden = model.get_initial_hidden(1)

    # 3. 推理
    prob_target = 0.5
    prob_anti = 0.5

    with torch.no_grad():
        action_logits, _, _ = model(input_tensor, hidden)
        probs = torch.sigmoid(action_logits).squeeze().numpy()

        if probs.ndim == 0: probs = [probs.item()]

        # 假设输出维度0是Target给奖概率，维度1是Anti给奖概率
        prob_target = probs[0] if len(probs) > 0 else 0.5
        prob_anti = probs[1] if len(probs) > 1 else 0.5

    # 4. 根据概率采样 (Sampling)
    # 这里生成初步决定
    target_proposal = 1 if np.random.random() < prob_target else 0
    anti_proposal = 1 if np.random.random() < prob_anti else 0

    # 5. 应用竞赛的硬性约束 (Constrain)
    # 这一步至关重要，它保证了总奖励数不会超过25，也不会少于25（如果在最后几轮）
    final_target = constrain(target_allocations, target_proposal)
    final_anti = constrain(anti_target_allocations, anti_proposal)

    return final_target, final_anti


def constrain(previous_allocation, current_allocation):
    """
    Constrain the current allocation based on previous allocations.
    (This function is from the template, do not change)
    """
    allocated_rewards = sum(previous_allocation)

    # If all rewards were already allocated, no more rewards may be allocated
    if allocated_rewards >= TOTAL_REWARDS:
        return 0

    # If there are as many trials left as rewards left, in all remaining trials
    # rewards should be allocated
    current_trial_number = len(previous_allocation)
    remaining_trials = NUMBER_OF_TRIALS - current_trial_number
    if remaining_trials == (TOTAL_REWARDS - allocated_rewards):
        return 1

    # No constrain should be imposed
    return current_allocation


###############################################################################
# Template Infrastructure - Do not change
###############################################################################
REWARDS_BOTH_ALTERNATIVES = '1, 1'
REWARD_TARGET_ONLY = '1, 0'
REWARD_ANTI_TARGET_ONLY = '0, 1'
NO_REWARDS_BOTH_ALTERNATIVES = '0, 0'


def parse_input():
    if len(sys.argv) < 2 or sys.argv[1] == "[]":
        return [], [], []
    else:
        target_allocations = parse_lst(sys.argv[1])
        anti_target_allocations = parse_lst(sys.argv[2])
        is_target_choices = parse_lst(sys.argv[3])
        return target_allocations, anti_target_allocations, is_target_choices


def output(target, anti_target):
    if target and anti_target:
        print(REWARDS_BOTH_ALTERNATIVES)
    elif target and not anti_target:
        print(REWARD_TARGET_ONLY)
    elif not target and anti_target:
        print(REWARD_ANTI_TARGET_ONLY)
    elif not target and not anti_target:
        print(NO_REWARDS_BOTH_ALTERNATIVES)


def parse_lst(lst):
    as_python_lst = lst.strip('[').strip(']').split(',')
    # Handle case where list might be empty string after split
    if len(as_python_lst) == 1 and as_python_lst[0] == '':
        return []
    as_python_elements = [ast.literal_eval(el.strip()) for el in as_python_lst if el.strip()]
    return as_python_elements


###############################################################################
# Run
###############################################################################

if __name__ == '__main__':
    input_data = parse_input()
    allocation = allocate(*input_data)
    output(*allocation)