#!/usr/bin/env python3
"""Overnight trainer for BiLSTM allocator policy.

Features:
- Uses `neural_ql_agent` as the simulated human for rollouts.
- REINFORCE-style training loop with checkpointing.
- Early stopping (patience on validation reward), ReduceLROnPlateau, periodic reporting.
- Autonomous: runs until max_seconds or early-stop; writes `best_bilstm.php`-style outputs (policy snapshots) and a `bilstm_report.txt` summary.

This script is designed to run unattended and produce periodic logs in `outdir`.
"""
import os
import sys
import time
import argparse
import json
import random
import numpy as np
import torch
from torch import optim

HERE = os.path.dirname(__file__)
# Repository root (one level above the `experiment` directory)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment.sequence_adaptive_agent import models

# local import of human agent loader
import importlib.util

def load_human_module():
    """Load the neural_ql_agent module reliably.

    Try package import first, fallback to file-based import relative to repo.
    """
    try:
        # Preferred: package import
        return importlib.import_module('experiment.rl_agents.neural_ql.neural_ql_agent')
    except Exception:
        # Fallback: load by file path relative to repository root
        agent_dir = os.path.join(ROOT, 'rl_agents', 'neural_ql')
        agent_path = os.path.join(agent_dir, 'neural_ql_agent.py')
        spec = importlib.util.spec_from_file_location('neural_ql_agent', agent_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def rollout_return(policy, human_module, device, episodes=10, trials_per_episode=100):
    # run episodes using human model as environment, return mean reward per episode
    policy.eval()
    all_means = []
    human_model, seq_len = human_module.load_model()
    if human_model is None:
        raise RuntimeError('No human model found')
    for ep in range(episodes):
        rem_t, rem_a = 25, 25
        state = human_module.get_fresh_state()
        hx = None
        ep_rewards = []
        for t in range(trials_per_episode):
            seq_tensor, glob_tensor = human_module.prepare_input(state, seq_len)
            choice = human_model.get_action(seq_tensor, glob_tensor, use_policy=True, epsilon=0.0)
            # prepare policy input
            s = np.array([t/100.0, rem_t/25.0, rem_a/25.0, 0.0, 0.0, 0.0, 1.0 if choice==1 else 0.0], dtype=np.float32)
            x = torch.from_numpy(s).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, hx = call_policy(policy, x, hx)
                prob = torch.sigmoid(logits)
                act = int((prob>0.5).item())
            # enforce budgets
            if rem_t <= 0:
                act = 0
            if rem_a <= 0:
                act = 1
            r = 1.0 if act == choice else 0.0
            action_str = 'RIGHT' if choice==1 else 'LEFT'
            state = human_module.update_state(state, action_str, float(r))
            if act==1:
                rem_t -= 1
            else:
                rem_a -= 1
            ep_rewards.append(r)
        all_means.append(np.mean(ep_rewards))
    policy.train()
    return float(np.mean(all_means))


def save_policy_snapshot(policy, outpath):
    # save PyTorch state_dict
    torch.save({'state_dict': policy.state_dict()}, outpath)


def call_policy(policy, x, hx=None):
    """Call policy with either RNN-style (batch,seq,feat) or MLP-style (batch,feat) inputs.

    Ensures `x` is shaped correctly for RNN policies and returns (logits, hx_or_none).
    """
    # if x is 2D (batch, feat), convert to (batch, seq=1, feat)
    if x.dim() == 2:
        x_seq = x.unsqueeze(1)
    else:
        x_seq = x
    try:
        out = policy(x_seq, hx)
        if isinstance(out, tuple):
            return out
        else:
            return out, None
    except TypeError:
        # fallback to MLP policy that expects 2D input
        out = policy(x)
        return out, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--outdir', default='experiment/sequence_adaptive_agent/output_overnight')
    parser.add_argument('--max_seconds', type=int, default=24*3600)
    parser.add_argument('--report_interval', type=int, default=300)
    parser.add_argument('--episodes_eval', type=int, default=20)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--batch_episodes', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dim for BiLSTM')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout for BiLSTM')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    human_module = load_human_module()

    # set seeds if provided
    if args.seed is not None:
        import random as _py_rand
        _py_rand.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # build BiLSTM policy (reuse existing model definition)
    policy = models.BiLSTMPolicy(input_dim=7, hidden_dim=args.hidden_dim, num_layers=1, dropout=args.dropout).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_score = -1.0
    best_snapshot = None
    no_improve = 0
    start = time.time()
    last_report = start
    epoch = 0

    # main loop: run batches until time up or early stop
    while True:
        epoch += 1
        # perform a batch of REINFORCE episodes
        # this uses same rollout_and_train logic simplified
        policy.train()
        batch_rewards = []
        for ep in range(args.batch_episodes):
            rem_t, rem_a = 25, 25
            state = human_module.get_fresh_state()
            hx = None
            log_probs = []
            rewards = []
            for t in range(args.trials):
                seq_tensor, glob_tensor = human_module.prepare_input(state, human_module.load_model()[1])
                choice = human_module.load_model()[0].get_action(seq_tensor, glob_tensor, use_policy=True, epsilon=0.0)
                s = np.array([t/100.0, rem_t/25.0, rem_a/25.0, 0.0, 0.0, 0.0, 1.0 if choice==1 else 0.0], dtype=np.float32)
                x = torch.from_numpy(s).unsqueeze(0).to(device)
                # forward: compute policy probability and sample a proposal
                logits, hx = call_policy(policy, x, hx)
                prob = torch.sigmoid(logits)
                prob_scalar = prob.squeeze()
                # sample proposed action from policy distribution (on-device)
                m = torch.distributions.Bernoulli(prob_scalar)
                sampled_act_tensor = m.sample()
                sampled_act = int(sampled_act_tensor.item())

                # determine executed action after enforcing budgets
                if rem_t <= 0:
                    final_act = 0
                elif rem_a <= 0:
                    final_act = 1
                else:
                    final_act = sampled_act

                # compute log_prob for the actually executed action so the
                # gradient target matches the action that produced the reward
                try:
                    final_act_tensor = torch.tensor(float(final_act), dtype=sampled_act_tensor.dtype, device=prob.device)
                    log_prob = m.log_prob(final_act_tensor)
                except Exception:
                    # fallback: compute log_prob on CPU scalar
                    final_act_tensor = torch.tensor(float(final_act))
                    log_prob = m.log_prob(final_act_tensor.to(device))

                # reward is based on executed (final) action
                r = 1.0 if final_act == choice else 0.0
                action_str = 'RIGHT' if choice==1 else 'LEFT'
                # update state with the actual received reward (immediate feedback)
                state = human_module.update_state(state, action_str, float(r))
                # decrement budget according to executed action
                if final_act == 1:
                    rem_t -= 1
                else:
                    rem_a -= 1
                # use the log_prob of the policy's sampled proposal for learning
                log_probs.append(log_prob)
                rewards.append(r)
            returns = torch.tensor(rewards, dtype=torch.float32, device=device)
            baseline = returns.mean()
            loss = 0.0
            for lp, R in zip(log_probs, returns):
                loss = loss - lp * (R - baseline)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_rewards.append(float(returns.mean().item()))

        # evaluate current policy
        eval_score = rollout_return(policy, human_module, device, episodes=args.episodes_eval, trials_per_episode=args.trials)

        now = time.time()
        # write periodic report line
        with open(os.path.join(args.outdir, 'bilstm_report.txt'), 'a') as f:
            f.write(f'EPOCH {epoch} TIME {now-start:.1f}s EVAL {eval_score:.6f} LR {optimizer.param_groups[0]["lr"]:.6e}\n')

        # scheduler step
        scheduler.step(eval_score)

        # check improvement
        if eval_score > best_score + args.min_delta:
            best_score = eval_score
            best_snapshot = os.path.join(args.outdir, f'bilstm_best_epoch{epoch}.pt')
            save_policy_snapshot(policy, best_snapshot)
            no_improve = 0
            # also write a php-like allocation by using policy greedy rollout once
            try:
                # create a representative static php by greedy allocating according to policy over 100 trials
                t_arr = np.zeros(100, dtype=int)
                a_arr = np.zeros(100, dtype=int)
                # simple greedy fill first 25 target=1 positions based on policy sampling
                # Here we emulate a policy choice over blank human history; this is heuristic but creates an artifact
                rem_t, rem_a = 25, 25
                state = human_module.get_fresh_state()
                hx = None
                for tt in range(100):
                    seq_tensor, glob_tensor = human_module.prepare_input(state, human_module.load_model()[1])
                    choice = human_module.load_model()[0].get_action(seq_tensor, glob_tensor, use_policy=True, epsilon=0.0)
                    s = np.array([tt/100.0, rem_t/25.0, rem_a/25.0, 0.0,0.0,0.0,1.0 if choice==1 else 0.0], dtype=np.float32)
                    x = torch.from_numpy(s).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits, hx = call_policy(policy, x, hx)
                        prob = torch.sigmoid(logits)
                        act = int((prob>0.5).item())
                    if rem_t <= 0:
                        act = 0
                    if rem_a <= 0:
                        act = 1
                    if act==1:
                        t_arr[tt]=1
                        rem_t -=1
                    else:
                        a_arr[tt]=1
                        rem_a -=1
                # save php
                from experiment.sequence_optimization_static.optimize_static import save_php
                save_php(t_arr, a_arr, os.path.join(args.outdir, 'bilstm_best.php'))
            except Exception:
                pass
        else:
            no_improve += 1

        # stopping checks
        if (time.time() - start) > args.max_seconds:
            break
        if no_improve >= args.patience:
            # no improvement for patience epochs
            break

        # periodic flush
        if now - last_report >= args.report_interval:
            last_report = now

    # finalize
    with open(os.path.join(args.outdir, 'bilstm_summary.txt'), 'w') as f:
        f.write(json.dumps({'best_eval': best_score, 'elapsed_s': time.time()-start, 'epochs': epoch}))

    print('Training finished. Best eval:', best_score)

if __name__ == '__main__':
    main()
