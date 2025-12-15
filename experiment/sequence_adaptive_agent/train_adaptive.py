import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

# ensure repo root is on sys.path so `experiment.*` packages can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment.sequence_adaptive_agent import models


def safe_import_agent():
    # Try to import the existing neural agent harness; return module or raise
    try:
        import experiment.rl_agents.neural_ql.neural_ql_agent as human_agent
        return human_agent
    except Exception as e:
        print('Failed to import `neural_ql_agent`:', e)
        raise


def make_state_vector(trial_idx, remaining_t, remaining_a, last_choice, last_reward, current_human_choice):
    # normalized simple feature vector
    max_trials = 100.0
    return np.array([
        trial_idx / max_trials,
        remaining_t / 25.0,
        remaining_a / 25.0,
        1.0 if last_choice == 1 else 0.0,
        1.0 if last_choice == 0 else 0.0,
        1.0 if last_reward else 0.0,
        1.0 if current_human_choice == 1 else 0.0,
    ], dtype=np.float32)


def rollout_and_train(policy, policy_type, human_module, device, episodes=10, trials_per_episode=100, lr=1e-3):
    opt = optim.Adam(policy.parameters(), lr=lr)
    results = []
    for ep in range(episodes):
        # budgets
        rem_t = 25
        rem_a = 25
        last_choice = -1
        last_reward = 0

        log_probs = []
        rewards = []

        # prepare LSTM/GRU hidden if needed
        hx = None

        # load human model for simulation; ensure human module directory is on sys.path
        human_dir = os.path.dirname(human_module.__file__)
        if human_dir not in sys.path:
            sys.path.insert(0, human_dir)
        human_model, seq_len = human_module.load_model()
        if human_model is None:
            raise RuntimeError('No trained human model found; please ensure there is a checkpoint in trained_models')

        # fresh agent state for simulated human
        state = human_module.get_fresh_state()

        for t in range(trials_per_episode):
            # First: get human choice based on current state (history up to previous trial)
            seq_tensor, glob_tensor = human_module.prepare_input(state, seq_len)
            choice = human_model.get_action(seq_tensor, glob_tensor, use_policy=True, epsilon=0.0)

            # now build policy observation including current human choice
            s = make_state_vector(t, rem_t, rem_a, last_choice, last_reward, choice)
            x = torch.from_numpy(s).unsqueeze(0).to(device)

            # forward (policy samples allocation: 1 -> reward target (RIGHT), 0 -> reward anti (LEFT))
            if policy_type == 'mlp':
                logits = policy(x)
                prob = torch.sigmoid(logits)
                m = torch.distributions.Bernoulli(prob)
                act = int(m.sample().item())
                log_prob = m.log_prob(torch.tensor(float(act)).to(device))
            else:
                logits, hx = policy(x.unsqueeze(1), hx)
                prob = torch.sigmoid(logits)
                m = torch.distributions.Bernoulli(prob)
                act = int(m.sample().item())
                log_prob = m.log_prob(torch.tensor(float(act)).to(device))

            # respect budgets
            if rem_t <= 0:
                act = 0
            if rem_a <= 0:
                act = 1

            # assign reward: human receives reward if their chosen side matches allocator
            r = 1.0 if act == choice else 0.0

            # update human state using textual action names expected by update_state
            action_str = 'RIGHT' if choice == 1 else 'LEFT'
            state = human_module.update_state(state, action_str, float(r))

            # update budgets and history
            if act == 1:
                rem_t -= 1
            else:
                rem_a -= 1
            last_choice = choice
            last_reward = 1 if r > 0 else 0

            log_probs.append(log_prob)
            rewards.append(r)

        # compute REINFORCE return (simple baseline = mean)
        returns = torch.tensor(rewards, dtype=torch.float32, device=device)
        baseline = returns.mean()
        loss = 0.0
        for lp, R in zip(log_probs, returns):
            loss = loss - lp * (R - baseline)

        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_mean = returns.mean().item()
        results.append(ep_mean)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--only_arch', type=str, default=None, help='If set, run only this architecture (e.g. bilstm)')
    parser.add_argument('--outdir', type=str, default='experiment/sequence_adaptive_agent/output')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    human = safe_import_agent()

    device = torch.device(args.device)

    archs = {
        'mlp': models.MLPPolicy(input_dim=7, hidden_dim=64).to(device),
        'lstm': models.LSTMPolicy(input_dim=7, hidden_dim=64, num_layers=1).to(device),
        'gru': models.GRUPolicy(input_dim=7, hidden_dim=64, num_layers=1).to(device),
        'bilstm': models.BiLSTMPolicy(input_dim=7, hidden_dim=64, num_layers=1).to(device),
        'conv': models.ConvPolicy(input_dim=7, hidden_dim=64).to(device),
        'transformer': models.TransformerPolicy(input_dim=7, hidden_dim=64, nhead=4, nlayers=2).to(device),
    }

    summary = {}
    for name, policy in archs.items():
        if args.only_arch and name != args.only_arch:
            continue
        print('Testing arch:', name)
        res = rollout_and_train(policy, name, human, device, episodes=args.episodes, trials_per_episode=args.trials, lr=args.lr)
        summary[name] = res
        # save short summary
        with open(os.path.join(args.outdir, f'summary_{name}.txt'), 'w') as f:
            f.write('\n'.join([f'{v:.4f}' for v in res]))

    # write a combined report
    rep = os.path.join(args.outdir, 'arch_compare_report.txt')
    with open(rep, 'w') as f:
        for k, v in summary.items():
            f.write(f'{k}: mean_epoch={np.mean(v):.4f} std_epoch={np.std(v):.4f}\n')

    print('Done. Reports saved in', args.outdir)


if __name__ == '__main__':
    main()
