#!/usr/bin/env python3
"""
Ablation Study for Fast Model Rules

This script evaluates the contribution of each rule by:
1. Running the full model (all rules enabled)
2. Running with each rule disabled one at a time
3. Computing the accuracy drop when each rule is disabled
4. The accuracy drop = contribution of that rule

Output: JSON with accuracy for each configuration and rule contributions
"""
import os
import sys
import json
import random
import argparse
import numpy as np
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.neural_ql_fast_v2 import FastModelV2, RULE_ENABLED

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from data_loader import HumanDataLoader

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'ablation_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(model, sessions: List, n_runs: int = 1) -> Dict:
    """Evaluate model accuracy on human sessions."""
    all_run_accuracies = []
    
    for run in range(n_runs):
        session_correct = 0
        session_total = 0
        
        for session_data in sessions:
            actions = session_data['actions']
            rewards = session_data['rewards']
            
            if len(actions) == 0:
                continue
            
            model.reset()
            correct = 0
            
            for t in range(len(actions)):
                human_action = actions[t]
                model_pred = model.predict()
                
                if model_pred == human_action:
                    correct += 1
                
                reward = rewards[t] if t < len(rewards) else 0
                model.update(human_action, reward)
            
            session_correct += correct
            session_total += len(actions)
        
        if session_total > 0:
            run_accuracy = session_correct / session_total
            all_run_accuracies.append(run_accuracy)
    
    if len(all_run_accuracies) == 0:
        return {'accuracy': 0, 'std': 0}
    
    return {
        'accuracy': np.mean(all_run_accuracies),
        'std': np.std(all_run_accuracies) if len(all_run_accuracies) > 1 else 0
    }


def run_ablation_study(sessions: List, n_runs: int = 5, verbose: bool = True) -> Dict:
    """
    Run ablation study to compute contribution of each rule.
    
    Returns:
        Dictionary with full model accuracy, per-rule ablation results,
        and computed contributions.
    """
    results = {
        'full_model': None,
        'ablations': {},
        'contributions': {},
        'n_sessions': len(sessions),
        'n_trials': sum(len(s['actions']) for s in sessions),
        'n_runs': n_runs
    }
    
    if verbose:
        print("=" * 60)
        print("Ablation Study: Rule Contribution Analysis")
        print("=" * 60)
        print(f"Sessions: {results['n_sessions']}, Trials: {results['n_trials']:,}")
        print(f"Runs per config: {n_runs}")
    
    all_rules = list(RULE_ENABLED.keys())
    
    if verbose:
        print("\n[1/{}] Evaluating Full Model (all rules enabled)...".format(len(all_rules) + 1))
    
    full_model = FastModelV2(rules_enabled={r: True for r in all_rules})
    full_result = evaluate_model(full_model, sessions, n_runs)
    results['full_model'] = full_result
    
    if verbose:
        print(f"    Full Model Accuracy: {full_result['accuracy']:.2%} (+/- {full_result['std']:.2%})")
    
    for i, rule in enumerate(all_rules):
        if verbose:
            print(f"\n[{i+2}/{len(all_rules)+1}] Ablating rule: {rule}...")
        
        ablated_rules = {r: True for r in all_rules}
        ablated_rules[rule] = False
        
        ablated_model = FastModelV2(rules_enabled=ablated_rules)
        ablated_result = evaluate_model(ablated_model, sessions, n_runs)
        results['ablations'][rule] = ablated_result
        
        contribution = full_result['accuracy'] - ablated_result['accuracy']
        results['contributions'][rule] = {
            'absolute': contribution,
            'relative': contribution / full_result['accuracy'] if full_result['accuracy'] > 0 else 0
        }
        
        if verbose:
            print(f"    Without {rule}: {ablated_result['accuracy']:.2%}")
            print(f"    Contribution: {contribution:+.2%} ({results['contributions'][rule]['relative']:.1%} of total)")
    
    if verbose:
        print("\n[Bonus] Evaluating baseline (no rules, random 50%)...")
    
    no_rules = {r: False for r in all_rules}
    baseline_model = FastModelV2(rules_enabled=no_rules)
    baseline_result = evaluate_model(baseline_model, sessions, n_runs)
    results['baseline'] = baseline_result
    
    if verbose:
        print(f"    Baseline (random): {baseline_result['accuracy']:.2%}")
        improvement = full_result['accuracy'] - baseline_result['accuracy']
        print(f"    Total improvement over random: {improvement:+.2%}")
    
    return results


def print_contribution_report(results: Dict):
    """Print a formatted contribution report."""
    print("\n" + "=" * 60)
    print("RULE CONTRIBUTION REPORT")
    print("=" * 60)
    
    full_acc = results['full_model']['accuracy']
    baseline_acc = results['baseline']['accuracy']
    total_improvement = full_acc - baseline_acc
    
    print(f"\nFull Model Accuracy:  {full_acc:.2%}")
    print(f"Baseline (random):    {baseline_acc:.2%}")
    print(f"Total Improvement:    {total_improvement:+.2%}")
    
    print("\n" + "-" * 60)
    print(f"{'Rule':<20} {'Ablated Acc':<12} {'Contribution':<12} {'% of Improv':<12}")
    print("-" * 60)
    
    sorted_rules = sorted(
        results['contributions'].items(),
        key=lambda x: x[1]['absolute'],
        reverse=True
    )
    
    for rule, contrib in sorted_rules:
        ablated_acc = results['ablations'][rule]['accuracy']
        abs_contrib = contrib['absolute']
        pct_of_improvement = (abs_contrib / total_improvement * 100) if total_improvement > 0 else 0
        
        print(f"{rule:<20} {ablated_acc:.2%}        {abs_contrib:+.2%}        {pct_of_improvement:.1f}%")
    
    print("-" * 60)
    
    sum_contributions = sum(c['absolute'] for c in results['contributions'].values())
    print(f"{'Sum of contributions':<20} {'':<12} {sum_contributions:+.2%}        {sum_contributions/total_improvement*100:.1f}%")
    print(f"{'(Note: may not equal total due to rule interactions)'}")
    
    print("\n" + "=" * 60)
    print("TOP CONTRIBUTORS (sorted by impact)")
    print("=" * 60)
    
    for i, (rule, contrib) in enumerate(sorted_rules[:5], 1):
        bar_len = int(contrib['absolute'] * 200)
        bar = "█" * max(bar_len, 1)
        print(f"{i}. {rule:<15} {contrib['absolute']:+.2%} {bar}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Study for Fast Model Rules")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per configuration")
    parser.add_argument("--sample", type=int, default=0, help="Sample size (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    verbose = not args.quiet
    
    if verbose:
        print("Loading human data...")
    
    loader = HumanDataLoader(sequence_length=10)
    all_sessions_df = loader.load_all_human_data()
    
    if not all_sessions_df:
        print("Error: No human data found!")
        return
    
    sessions = []
    for df in all_sessions_df:
        session_data = loader.preprocess_session(df)
        if len(session_data['actions']) > 0:
            sessions.append(session_data)
    
    if args.sample > 0 and args.sample < len(sessions):
        random.shuffle(sessions)
        sessions = sessions[:args.sample]
        if verbose:
            print(f"Sampled {len(sessions)} sessions")
    
    results = run_ablation_study(sessions, n_runs=args.runs, verbose=verbose)
    
    if verbose:
        print_contribution_report(results)
    
    output_file = args.output or os.path.join(RESULTS_DIR, 'ablation_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to: {output_file}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
