#!/usr/bin/env python3
"""BiLSTM trainer (overnight) for dynamic reward allocation.

Goal
- Maximize the probability that the simulated human (neural_ql) chooses the
  biased side (the target alternative) across 100 trials.

Rules
- Each episode has exactly 100 trials.
- The allocator chooses reward availability each trial for both sides.
- Each side must receive exactly 25 rewards over the 100 trials.
- Overlap is allowed: both sides may have reward on the same trial (1,1).

Interface constraints
- Inference interface is fixed in `infer.py` as `bilstm_infer(target_allocations,
  anti_target_allocations, is_target_choices) -> (int,int)`.
- Training must match the same allocation logic (sampling + forcing).
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
from torch import optim

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment.sequence_adaptive_agent import models


TOTAL_TRIALS = 100
PER_SIDE = 25
INPUT_DIM = 7


def save_php_exact(target_arr: List[int], anti_arr: List[int], out_path: str):
    """Write a static PHP allocation file (exact 100 length, exact 25 ones each)."""
    if len(target_arr) != TOTAL_TRIALS or len(anti_arr) != TOTAL_TRIALS:
        raise ValueError('save_php_exact expects 100-length arrays')
    if int(sum(target_arr)) != PER_SIDE or int(sum(anti_arr)) != PER_SIDE:
        raise ValueError(f'save_php_exact expects exactly {PER_SIDE} ones per side')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(" <?php \n")
        f.write("$biased_rewards = [")
        f.write(', '.join(str(int(x)) for x in target_arr))
        f.write("]; \n")
        f.write("$unbiased_rewards = [")
        f.write(', '.join(str(int(x)) for x in anti_arr))
        f.write("]; \n")
        f.write('?>\n')


def load_human_module():
    import importlib

    return importlib.import_module('experiment.rl_agents.neural_ql.neural_ql_agent')


def _build_feature_seq(
    target_allocations: List[int],
    anti_target_allocations: List[int],
    is_target_choices: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Must match `infer.py` feature construction exactly."""
    targ = list(target_allocations)
    anti = list(anti_target_allocations)
    choices = list(is_target_choices)

    trial = max(len(targ), len(anti), len(choices))

    if len(targ) < trial:
        targ.extend([0] * (trial - len(targ)))
    if len(anti) < trial:
        anti.extend([0] * (trial - len(anti)))
    if len(choices) < trial:
        choices.extend([0] * (trial - len(choices)))

    feats = []
    for i in range(trial + 1):
        rem_t = max(0, PER_SIDE - sum(targ[:i]))
        rem_a = max(0, PER_SIDE - sum(anti[:i]))

        choice_rate = float(np.mean(choices[:i])) if i > 0 else 0.5
        last_choice = float(choices[i - 1]) if i > 0 else 0.0

        if i > 0:
            last_reward = 1.0 if ((choices[i - 1] == 1 and targ[i - 1] == 1) or (choices[i - 1] == 0 and anti[i - 1] == 1)) else 0.0
            last_alloc_total = float(targ[i - 1] + anti[i - 1]) / 2.0
        else:
            last_reward = 0.0
            last_alloc_total = 0.0

        feat = np.array(
            [
                float(i) / float(TOTAL_TRIALS),
                float(rem_t) / float(PER_SIDE),
                float(rem_a) / float(PER_SIDE),
                choice_rate,
                last_choice,
                last_reward,
                last_alloc_total,
            ],
            dtype=np.float32,
        )
        feats.append(feat)

    x = torch.from_numpy(np.stack(feats, axis=0)).unsqueeze(0).to(device)
    return x


def _forced_action(rem: int, trials_left_including_current: int) -> Tuple[bool, int]:
    if rem <= 0:
        return True, 0
    if rem >= trials_left_including_current:
        return True, 1
    return False, 0


def _sample_action_with_forcing(
    p: torch.Tensor,
    rem: int,
    trials_left_including_current: int,
    deterministic: bool,

) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Return (action_int, log_prob_tensor, entropy_tensor).

    If forced, log_prob is 0 (no gradient contribution).
    """
    forced, forced_val = _forced_action(rem, trials_left_including_current)
    if forced:
        return forced_val, p.new_zeros(()), p.new_zeros(())

    if deterministic:
        action = 1 if float(p.item()) > 0.5 else 0
        return action, p.new_zeros(()), p.new_zeros(())

    m = torch.distributions.Bernoulli(probs=p.clamp(1e-6, 1 - 1e-6))
    a = m.sample()
    action = int(a.item())

    # Feasibility check after action
    trials_left_after = trials_left_including_current - 1
    rem_after = rem - action
    if rem_after > trials_left_after:
        # forced to 1 for feasibility
        return 1, p.new_zeros(()), p.new_zeros(())

    logp = m.log_prob(a)
    ent = m.entropy()
    return action, logp, ent


def _noisy_human_choice(choice: int, lapse_prob: float, flip_prob: float) -> int:
    """Apply simple behavioral noise to reduce overfitting to neural_ql.

    - lapse_prob: with this prob, choose randomly (0/1).
    - flip_prob: with this prob (and not lapsed), flip the choice.
    """
    if lapse_prob > 0.0:
        if float(np.random.rand()) < float(lapse_prob):
            return int(np.random.randint(0, 2))
    if flip_prob > 0.0:
        if float(np.random.rand()) < float(flip_prob):
            return int(1 - int(choice))
    return int(choice)


def _simulate_episode(
    policy: torch.nn.Module,
    human_module,
    human_model,
    human_seq_len: int,
    device: torch.device,
    biased_side: int,
    deterministic_policy: bool,
    human_epsilon: float = 0.0,
    human_lapse_prob: float = 0.0,
    human_flip_prob: float = 0.0,
) -> Tuple[float, torch.Tensor, dict, List[int], List[int]]:
    """Run one 100-trial episode.

    Returns:
    - R: fraction of choices == biased_side
    - logp_sum: sum of log-probs of non-forced allocator decisions
    - stats: dict for debugging
    """
    targ: List[int] = []
    anti: List[int] = []
    choices: List[int] = []

    state = human_module.get_fresh_state()

    logp_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    entropy_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    forced_count = 0
    p_t_list = []
    p_a_list = []

    hx = None

    for t in range(TOTAL_TRIALS):
        allocated_t = sum(targ)
        allocated_a = sum(anti)
        rem_t = max(0, PER_SIDE - allocated_t)
        rem_a = max(0, PER_SIDE - allocated_a)
        trials_left_including_current = TOTAL_TRIALS - t

        x = _build_feature_seq(targ, anti, choices, device)
        out = policy(x, hx) if callable(getattr(policy, 'forward', None)) else policy(x)
        if isinstance(out, tuple):
            logits, hx = out
        else:
            logits, hx = out, None

        logits = logits.squeeze(0)
        probs = torch.sigmoid(logits)
        p_t = probs[0]
        p_a = probs[1]
        p_t_list.append(float(p_t.detach().cpu().item()))
        p_a_list.append(float(p_a.detach().cpu().item()))

        act_t, logp_t, ent_t = _sample_action_with_forcing(p_t, rem_t, trials_left_including_current, deterministic_policy)
        act_a, logp_a, ent_a = _sample_action_with_forcing(p_a, rem_a, trials_left_including_current, deterministic_policy)

        if logp_t.numel() == 0 or logp_a.numel() == 0:
            pass
        logp_sum = logp_sum + logp_t + logp_a
        entropy_sum = entropy_sum + ent_t + ent_a

        if logp_t.item() == 0.0 and (rem_t <= 0 or rem_t >= trials_left_including_current):
            forced_count += 1
        if logp_a.item() == 0.0 and (rem_a <= 0 or rem_a >= trials_left_including_current):
            forced_count += 1

        targ.append(int(act_t))
        anti.append(int(act_a))

        # Human chooses
        seq_tensor, glob_tensor = human_module.prepare_input(state, human_seq_len)
        choice = human_model.get_action(seq_tensor, glob_tensor, use_policy=True, epsilon=float(human_epsilon))
        choice = _noisy_human_choice(int(choice), lapse_prob=human_lapse_prob, flip_prob=human_flip_prob)
        choices.append(int(choice))

        # Human reward is revealed after choice
        if choice == 1:
            human_r = 1.0 if act_t == 1 else 0.0
            action_str = 'RIGHT'
        else:
            human_r = 1.0 if act_a == 1 else 0.0
            action_str = 'LEFT'

        state = human_module.update_state(state, action_str, float(human_r))

    # Sanity: exact budgets
    if sum(targ) != PER_SIDE or sum(anti) != PER_SIDE:
        raise RuntimeError(f'Budget violated: targ={sum(targ)} anti={sum(anti)}')

    if biased_side == 1:
        R = float(np.mean(choices))
    else:
        R = float(np.mean([1 - c for c in choices]))

    stats = {
        'sum_t': int(sum(targ)),
        'sum_a': int(sum(anti)),
        'mean_p_t': float(np.mean(p_t_list)) if p_t_list else 0.0,
        'mean_p_a': float(np.mean(p_a_list)) if p_a_list else 0.0,
        'forced_count': int(forced_count),
        'entropy_sum': float(entropy_sum.detach().cpu().item()),
    }

    return R, logp_sum, stats, targ, anti, entropy_sum


def _eval_policy(
    policy,
    human_module,
    device: torch.device,
    episodes: int,
    biased_side: int,
    human_epsilon: float,
    human_lapse_prob: float,
    human_flip_prob: float,
) -> float:
    policy.eval()
    human_model, seq_len = human_module.load_model()
    if human_model is None:
        raise RuntimeError('No human model found (neural_ql)')

    scores = []
    # deterministic policy during eval for stability (still uses forcing)
    with torch.no_grad():
        for _ in range(episodes):
            R, _logp, _stats, _t_arr, _a_arr, _entropy = _simulate_episode(
                policy=policy,
                human_module=human_module,
                human_model=human_model,
                human_seq_len=seq_len,
                device=device,
                biased_side=biased_side,
                deterministic_policy=True,
                human_epsilon=human_epsilon,
                human_lapse_prob=human_lapse_prob,
                human_flip_prob=human_flip_prob,
            )
            scores.append(R)

    policy.train()
    return float(np.mean(scores))


def _save_checkpoint(policy, outpath: str, hidden_dim: int, num_layers: int, dropout: float):
    torch.save(
        {
            'state_dict': policy.state_dict(),
            'meta': {
                'hidden_dim': int(hidden_dim),
                'num_layers': int(num_layers),
                'dropout': float(dropout),
                'input_dim': int(INPUT_DIM),
            },
        },
        outpath,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--outdir', default='experiment/sequence_adaptive_agent/output_overnight')
    parser.add_argument('--max_seconds', type=int, default=24 * 3600)
    parser.add_argument('--report_interval', type=int, default=300)
    parser.add_argument('--episodes_eval', type=int, default=20)
    parser.add_argument('--biased_side', type=int, choices=[0, 1], default=1)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--batch_episodes', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--entropy_coef', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--human_epsilon_train', type=float, default=0.0)
    parser.add_argument('--human_epsilon_eval', type=float, default=0.0)
    parser.add_argument('--human_lapse_prob_train', type=float, default=0.0)
    parser.add_argument('--human_flip_prob_train', type=float, default=0.0)
    parser.add_argument('--human_lapse_prob_eval', type=float, default=0.0)
    parser.add_argument('--human_flip_prob_eval', type=float, default=0.0)
    parser.add_argument('--robust_eval', action='store_true')
    parser.add_argument('--robust_eval_mix', type=float, default=0.5)
    parser.add_argument('--robust_eval_lapse', type=float, default=0.02)
    parser.add_argument('--robust_eval_flip', type=float, default=0.02)
    parser.add_argument('--human_noise_domain_randomize', action='store_true')
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.trials != TOTAL_TRIALS:
        raise ValueError('This trainer is hard-coded to 100 trials to match competition rules')

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)

    if args.seed is not None:
        import random

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    human_module = load_human_module()
    human_model, seq_len = human_module.load_model()
    if human_model is None:
        raise RuntimeError('neural_ql model not found (trained_models/latest_model.txt missing?)')

    policy = models.BiLSTMPolicy(input_dim=INPUT_DIM, hidden_dim=args.hidden_dim, num_layers=1, dropout=args.dropout).to(device)

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        state = ckpt.get('state_dict', ckpt)
        policy.load_state_dict(state, strict=True)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    baseline = 0.5  # running baseline for R
    start = time.time()
    last_report = start
    no_improve = 0
    epoch = 0

    report_path = os.path.join(args.outdir, 'bilstm_report.txt')

    while True:
        epoch += 1
        batch_R = []
        batch_stats = []
        batch_entropy = []

        policy.train()
        for _ in range(args.batch_episodes):
            if args.human_noise_domain_randomize:
                lapse_train = float(np.random.rand()) * float(args.human_lapse_prob_train)
                flip_train = float(np.random.rand()) * float(args.human_flip_prob_train)
            else:
                lapse_train = float(args.human_lapse_prob_train)
                flip_train = float(args.human_flip_prob_train)

            R, logp_sum, stats, _t_arr, _a_arr, entropy_sum = _simulate_episode(
                policy=policy,
                human_module=human_module,
                human_model=human_model,
                human_seq_len=seq_len,
                device=device,
                biased_side=args.biased_side,
                deterministic_policy=False,
                human_epsilon=args.human_epsilon_train,
                human_lapse_prob=lapse_train,
                human_flip_prob=flip_train,
            )

            adv = float(R - baseline)
            loss = -logp_sum * adv
            if args.entropy_coef and args.entropy_coef > 0.0:
                loss = loss - float(args.entropy_coef) * entropy_sum

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), float(args.grad_clip))
            optimizer.step()

            baseline = 0.99 * baseline + 0.01 * R

            batch_R.append(R)
            batch_stats.append(stats)
            batch_entropy.append(float(entropy_sum.detach().cpu().item()))

        eval_clean = _eval_policy(
            policy,
            human_module,
            device,
            episodes=args.episodes_eval,
            biased_side=args.biased_side,
            human_epsilon=args.human_epsilon_eval,
            human_lapse_prob=args.human_lapse_prob_eval,
            human_flip_prob=args.human_flip_prob_eval,
        )

        if args.robust_eval:
            eval_noisy = _eval_policy(
                policy,
                human_module,
                device,
                episodes=args.episodes_eval,
                biased_side=args.biased_side,
                human_epsilon=args.human_epsilon_eval,
                human_lapse_prob=float(args.robust_eval_lapse),
                human_flip_prob=float(args.robust_eval_flip),
            )
            mix = float(args.robust_eval_mix)
            mix = max(0.0, min(1.0, mix))
            eval_score = mix * float(eval_clean) + (1.0 - mix) * float(eval_noisy)
        else:
            eval_noisy = None
            eval_score = float(eval_clean)

        now = time.time()
        if now - last_report >= args.report_interval:
            last_report = now

        mean_train_R = float(np.mean(batch_R)) if batch_R else 0.0
        mean_p_t = float(np.mean([s['mean_p_t'] for s in batch_stats])) if batch_stats else 0.0
        mean_p_a = float(np.mean([s['mean_p_a'] for s in batch_stats])) if batch_stats else 0.0
        mean_entropy = float(np.mean(batch_entropy)) if batch_entropy else 0.0

        with open(report_path, 'a', encoding='utf-8') as f:
            line = (
                f'EPOCH {epoch} TIME {now - start:.1f}s TRAIN_R {mean_train_R:.4f} '
                f'EVAL {eval_score:.4f} EVAL_CLEAN {float(eval_clean):.4f} '
            )
            if eval_noisy is not None:
                line += f'EVAL_NOISY {float(eval_noisy):.4f} '
            line += (
                f'BASELINE {baseline:.4f} LR {optimizer.param_groups[0]["lr"]:.3e} '
                f'MEAN_P_T {mean_p_t:.3f} MEAN_P_A {mean_p_a:.3f} ENT {mean_entropy:.2f}\n'
            )
            f.write(line)

        if eval_score > best_score + args.min_delta:
            best_score = eval_score
            no_improve = 0
            best_epoch_path = os.path.join(args.outdir, f'bilstm_best_epoch{epoch}.pt')
            _save_checkpoint(policy, best_epoch_path, hidden_dim=args.hidden_dim, num_layers=1, dropout=args.dropout)
            _save_checkpoint(policy, os.path.join(args.outdir, 'bilstm_best.pt'), hidden_dim=args.hidden_dim, num_layers=1, dropout=args.dropout)

            # Export a representative allocation to bilstm_best.php by running one
            # seeded stochastic rollout under the current policy + neural_ql.
            try:
                # Seed RNGs so the exported PHP is reproducible while still
                # reflecting the stochastic policy used during training.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(args.seed if args.seed is not None else 0)
                    try:
                        import random as _py_random
                        _py_random.seed(args.seed if args.seed is not None else 0)
                    except Exception:
                        pass
                    try:
                        np.random.seed(args.seed if args.seed is not None else 0)
                    except Exception:
                        pass

                    was_training = policy.training
                    policy.eval()
                    _Rexp, _lp, _st, t_arr, a_arr, _entropy = _simulate_episode(
                        policy=policy,
                        human_module=human_module,
                        human_model=human_model,
                        human_seq_len=seq_len,
                        device=device,
                        biased_side=args.biased_side,
                        deterministic_policy=False,
                        human_epsilon=args.human_epsilon_eval,
                        human_lapse_prob=args.human_lapse_prob_eval,
                        human_flip_prob=args.human_flip_prob_eval,
                    )
                    if was_training:
                        policy.train()
                save_php_exact(t_arr, a_arr, os.path.join(args.outdir, 'bilstm_best.php'))
            except Exception:
                # Never fail training due to export
                pass
        else:
            no_improve += 1

        if (time.time() - start) >= args.max_seconds:
            break
        if no_improve >= args.patience:
            break

    with open(os.path.join(args.outdir, 'bilstm_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps({'best_eval': best_score, 'elapsed_s': time.time() - start, 'epochs': epoch}))

    print('Training finished. Best eval:', best_score)


if __name__ == '__main__':
    main()
