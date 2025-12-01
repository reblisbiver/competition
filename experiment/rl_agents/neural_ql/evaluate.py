#!/usr/bin/env python3
"""
Model Evaluation and Visualization Tools
Analyzes how well the neural model matches human behavior patterns.

Usage:
    python evaluate.py [--model_name <name>] [--visualize]
"""
import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import NeuralQLModel, ModelEnsemble
from data_loader import HumanDataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "trained_models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: Optional[str] = None) -> Tuple[torch.nn.Module, int]:
    """Load a trained model."""
    if model_name is None:
        latest_file = os.path.join(MODEL_DIR, "latest_model.txt")
        if not os.path.exists(latest_file):
            raise FileNotFoundError("No trained model found. Please train a model first.")
        with open(latest_file, 'r') as f:
            model_name = f.read().strip()
            
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model_kwargs = checkpoint['model_kwargs']
    use_ensemble = checkpoint.get('use_ensemble', False)
    
    if use_ensemble:
        num_models = checkpoint.get('num_ensemble_models', 3)
        model = ModelEnsemble(num_models=num_models, **model_kwargs)
    else:
        model = NeuralQLModel(**model_kwargs)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    seq_length = checkpoint.get('sequence_length', 10)
    
    return model, seq_length


def simulate_session(
    model: torch.nn.Module,
    human_session: Dict,
    seq_length: int
) -> Dict:
    """
    Simulate model decisions on the same sequence as a human session.
    
    Returns comparison metrics.
    """
    actions = human_session['actions']
    rewards = human_session['rewards']
    n_trials = len(actions)
    
    model_actions = []
    agreements = []
    
    state = {
        'actions': [],
        'rewards': [],
        'left_count': 0,
        'right_count': 0,
        'left_reward': 0.0,
        'right_reward': 0.0,
        'total_reward': 0.0
    }
    
    for t in range(n_trials):
        seq_actions = np.zeros(seq_length)
        seq_rewards = np.zeros(seq_length)
        seq_rt = np.zeros(seq_length)
        
        history_len = min(len(state['actions']), seq_length)
        if history_len > 0:
            offset = seq_length - history_len
            seq_actions[offset:] = state['actions'][-history_len:]
            seq_rewards[offset:] = state['rewards'][-history_len:]
            
        sequence = np.stack([seq_actions, seq_rewards, seq_rt], axis=-1)
        
        total_trials = len(state['actions'])
        if total_trials > 0:
            left_rate = state['left_count'] / total_trials
            right_rate = state['right_count'] / total_trials
            left_avg = state['left_reward'] / state['left_count'] if state['left_count'] > 0 else 0.5
            right_avg = state['right_reward'] / state['right_count'] if state['right_count'] > 0 else 0.5
            avg_reward = state['total_reward'] / total_trials
        else:
            left_rate = right_rate = 0.5
            left_avg = right_avg = avg_reward = 0.5
            
        global_features = np.array([
            total_trials / 100.0,
            left_rate, right_rate,
            left_avg, right_avg,
            avg_reward
        ])
        
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
        global_tensor = torch.FloatTensor(global_features).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            _, action_probs = model(sequence_tensor, global_tensor)
            model_action = action_probs.argmax(dim=-1).item()
            
        model_actions.append(model_action)
        human_action = actions[t]
        agreements.append(1 if model_action == human_action else 0)
        
        state['actions'].append(human_action)
        state['rewards'].append(rewards[t])
        if human_action == 0:
            state['left_count'] += 1
            state['left_reward'] += rewards[t]
        else:
            state['right_count'] += 1
            state['right_reward'] += rewards[t]
        state['total_reward'] += rewards[t]
        
    return {
        'human_actions': actions,
        'model_actions': model_actions,
        'rewards': rewards,
        'agreements': agreements,
        'accuracy': np.mean(agreements),
        'n_trials': n_trials
    }


def evaluate_model(
    model: torch.nn.Module,
    seq_length: int,
    data_dir: Optional[str] = None
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Returns detailed metrics comparing model to human behavior.
    """
    loader = HumanDataLoader(sequence_length=seq_length)
    all_sessions = loader.load_all_human_data(data_dir)
    
    results = {
        'session_results': [],
        'overall_accuracy': 0,
        'pattern_similarity': {},
        'choice_distribution': {},
        'temporal_analysis': {}
    }
    
    all_agreements = []
    all_human_actions = []
    all_model_actions = []
    
    for df in all_sessions:
        session_data = loader.preprocess_session(df)
        sim_result = simulate_session(model, session_data, seq_length)
        
        results['session_results'].append({
            'accuracy': sim_result['accuracy'],
            'n_trials': sim_result['n_trials']
        })
        
        all_agreements.extend(sim_result['agreements'])
        all_human_actions.extend(sim_result['human_actions'])
        all_model_actions.extend(sim_result['model_actions'])
        
    results['overall_accuracy'] = np.mean(all_agreements)
    
    human_left_rate = 1 - np.mean(all_human_actions)
    model_left_rate = 1 - np.mean(all_model_actions)
    results['choice_distribution'] = {
        'human_left_rate': human_left_rate,
        'model_left_rate': model_left_rate,
        'distribution_error': abs(human_left_rate - model_left_rate)
    }
    
    def compute_switch_rate(actions):
        switches = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        return switches / (len(actions) - 1) if len(actions) > 1 else 0
        
    human_switch = compute_switch_rate(all_human_actions)
    model_switch = compute_switch_rate(all_model_actions)
    
    def compute_wsls(actions, rewards):
        wsls_count = 0
        total = 0
        for i in range(1, min(len(actions), len(rewards))):
            if rewards[i-1] == 1:
                if actions[i] == actions[i-1]:
                    wsls_count += 1
            else:
                if actions[i] != actions[i-1]:
                    wsls_count += 1
            total += 1
        return wsls_count / total if total > 0 else 0
        
    results['pattern_similarity'] = {
        'human_switch_rate': human_switch,
        'model_switch_rate': model_switch,
        'switch_rate_error': abs(human_switch - model_switch)
    }
    
    n_bins = 10
    trials_per_bin = len(all_agreements) // n_bins
    temporal_accuracy = []
    for i in range(n_bins):
        start = i * trials_per_bin
        end = start + trials_per_bin
        bin_acc = np.mean(all_agreements[start:end])
        temporal_accuracy.append(bin_acc)
        
    results['temporal_analysis'] = {
        'accuracy_over_time': temporal_accuracy,
        'early_accuracy': np.mean(temporal_accuracy[:3]),
        'late_accuracy': np.mean(temporal_accuracy[-3:])
    }
    
    return results


def create_visualizations(results: Dict, save_dir: str = RESULTS_DIR):
    """Create visualization plots for evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1 = axes[0, 0]
    accuracies = [r['accuracy'] for r in results['session_results']]
    ax1.hist(accuracies, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(results['overall_accuracy'], color='red', linestyle='--', 
                label=f'Mean: {results["overall_accuracy"]:.2%}')
    ax1.set_xlabel('Session Accuracy')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Session Accuracies')
    ax1.legend()
    
    ax2 = axes[0, 1]
    cd = results['choice_distribution']
    x = ['Human', 'Model']
    left_rates = [cd['human_left_rate'], cd['model_left_rate']]
    right_rates = [1 - cd['human_left_rate'], 1 - cd['model_left_rate']]
    
    bar_width = 0.35
    ax2.bar(x, left_rates, bar_width, label='LEFT', color='steelblue')
    ax2.bar(x, right_rates, bar_width, bottom=left_rates, label='RIGHT', color='coral')
    ax2.set_ylabel('Proportion')
    ax2.set_title('Choice Distribution Comparison')
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    ax3 = axes[1, 0]
    temporal = results['temporal_analysis']['accuracy_over_time']
    ax3.plot(range(1, len(temporal) + 1), temporal, 'o-', linewidth=2, markersize=8)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax3.set_xlabel('Time Bin')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy Over Time')
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    ax4 = axes[1, 1]
    ps = results['pattern_similarity']
    metrics = ['Switch Rate']
    human_vals = [ps['human_switch_rate']]
    model_vals = [ps['model_switch_rate']]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x_pos - width/2, human_vals, width, label='Human', color='steelblue')
    ax4.bar(x_pos + width/2, model_vals, width, label='Model', color='coral')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics)
    ax4.set_ylabel('Rate')
    ax4.set_title('Behavioral Pattern Comparison')
    ax4.legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, 'evaluation_plots.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Plots saved to: {plot_path}")
    return plot_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural Q-Learning Model")
    parser.add_argument("--model_name", type=str, default=None, help="Model name to evaluate")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--visualize", action="store_true", help="Create visualization plots")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Neural Q-Learning Model Evaluation")
    print("=" * 60)
    
    try:
        model, seq_length = load_model(args.model_name)
        print(f"Model loaded successfully")
        print(f"Sequence length: {seq_length}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print("\nEvaluating model on human data...")
    results = evaluate_model(model, seq_length, args.data_dir)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"Sessions Evaluated: {len(results['session_results'])}")
    
    print("\nChoice Distribution:")
    cd = results['choice_distribution']
    print(f"  Human LEFT rate: {cd['human_left_rate']:.2%}")
    print(f"  Model LEFT rate: {cd['model_left_rate']:.2%}")
    print(f"  Distribution Error: {cd['distribution_error']:.4f}")
    
    print("\nBehavioral Patterns:")
    ps = results['pattern_similarity']
    print(f"  Human Switch Rate: {ps['human_switch_rate']:.2%}")
    print(f"  Model Switch Rate: {ps['model_switch_rate']:.2%}")
    
    print("\nTemporal Analysis:")
    ta = results['temporal_analysis']
    print(f"  Early Trial Accuracy: {ta['early_accuracy']:.2%}")
    print(f"  Late Trial Accuracy: {ta['late_accuracy']:.2%}")
    
    if args.visualize:
        print("\nGenerating visualizations...")
        create_visualizations(results)
        
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
        
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj
        
    serializable_results = json.loads(
        json.dumps(results, default=convert_to_serializable)
    )
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
