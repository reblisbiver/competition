#!/usr/bin/env python3
"""
Fast Model Evaluation Tool
Tests the rule-based fast model against human behavioral data.

Usage:
    python evaluate_fast.py [--data_dir <path>] [--verbose]
    
Example:
    python evaluate_fast.py --verbose
"""
import os
import sys
import json
import random
import argparse
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data_loader import HumanDataLoader

LEFT_BIAS = 0.62
STAY_PROB = 0.66
WIN_STAY_BOOST = 0.15
LOSE_SHIFT_BOOST = 0.10

class FastModelSimulator:
    """Simulates the fast model's decision logic."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.last_action = None
        self.last_reward = None
        self.left_rewards = 0
        self.right_rewards = 0
        self.left_count = 0
        self.right_count = 0
        self.trial = 0
    
    def decide(self) -> str:
        """Make a decision based on current state."""
        if self.trial == 0 or self.last_action is None:
            return 'LEFT' if random.random() < LEFT_BIAS else 'RIGHT'
        
        stay_prob = STAY_PROB
        
        if self.last_reward == 1:
            stay_prob += WIN_STAY_BOOST
        else:
            stay_prob -= LOSE_SHIFT_BOOST
        
        left_count = max(self.left_count, 1)
        right_count = max(self.right_count, 1)
        left_rate = self.left_rewards / left_count
        right_rate = self.right_rewards / right_count
        
        if abs(left_rate - right_rate) > 0.1:
            if left_rate > right_rate:
                stay_prob = stay_prob if self.last_action == 'LEFT' else (1 - stay_prob)
            else:
                stay_prob = stay_prob if self.last_action == 'RIGHT' else (1 - stay_prob)
        
        stay_prob = max(0.1, min(0.9, stay_prob))
        
        if random.random() < stay_prob:
            return self.last_action
        else:
            return 'RIGHT' if self.last_action == 'LEFT' else 'LEFT'
    
    def update(self, action: str, reward: float):
        """Update state after making a decision."""
        self.last_action = action
        self.last_reward = reward
        
        if action == 'LEFT':
            self.left_count += 1
            self.left_rewards += reward
        else:
            self.right_count += 1
            self.right_rewards += reward
        
        self.trial += 1


def simulate_session(human_session: Dict, n_simulations: int = 10) -> Dict:
    """
    Simulate fast model on the same sequence as a human session.
    Run multiple times to account for randomness.
    """
    actions = human_session['actions']
    rewards = human_session['rewards']
    n_trials = len(actions)
    
    all_agreements = []
    
    for _ in range(n_simulations):
        model = FastModelSimulator()
        session_agreements = []
        
        for t in range(n_trials):
            human_action = 'LEFT' if actions[t] == 0 else 'RIGHT'
            model_action = model.decide()
            
            agreement = 1 if model_action == human_action else 0
            session_agreements.append(agreement)
            
            reward = rewards[t] if t < len(rewards) else 0
            model.update(human_action, reward)
        
        all_agreements.append(np.mean(session_agreements))
    
    return {
        'accuracy': np.mean(all_agreements),
        'accuracy_std': np.std(all_agreements),
        'n_trials': n_trials
    }


def evaluate_fast_model(data_dir: str = None, verbose: bool = False) -> Dict:
    """
    Comprehensive evaluation of the fast model.
    """
    loader = HumanDataLoader(sequence_length=10)
    all_sessions = loader.load_all_human_data(data_dir)
    
    if not all_sessions:
        print("Error: No human data found!")
        return {}
    
    print(f"Loaded {len(all_sessions)} human sessions")
    
    results = {
        'session_results': [],
        'overall_accuracy': 0,
        'accuracy_std': 0
    }
    
    session_accuracies = []
    
    for i, df in enumerate(all_sessions):
        session_data = loader.preprocess_session(df)
        sim_result = simulate_session(session_data, n_simulations=10)
        
        results['session_results'].append(sim_result)
        session_accuracies.append(sim_result['accuracy'])
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(all_sessions)} sessions...")
    
    results['overall_accuracy'] = np.mean(session_accuracies)
    results['accuracy_std'] = np.std(session_accuracies)
    results['n_sessions'] = len(all_sessions)
    results['total_trials'] = sum(r['n_trials'] for r in results['session_results'])
    
    return results


def compare_with_neural(fast_results: Dict) -> Dict:
    """Compare fast model with neural model if available."""
    neural_results_path = os.path.join(SCRIPT_DIR, 'evaluation_results', 'evaluation_results.json')
    
    if os.path.exists(neural_results_path):
        with open(neural_results_path, 'r') as f:
            neural_results = json.load(f)
        
        return {
            'fast_accuracy': fast_results['overall_accuracy'],
            'neural_accuracy': neural_results.get('overall_accuracy', 'N/A'),
            'accuracy_gap': neural_results.get('overall_accuracy', 0) - fast_results['overall_accuracy']
        }
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fast Model")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Fast Model Evaluation")
    print("=" * 60)
    
    print("\nModel Parameters:")
    print(f"  LEFT_BIAS:        {LEFT_BIAS:.2%}")
    print(f"  STAY_PROB:        {STAY_PROB:.2%}")
    print(f"  WIN_STAY_BOOST:   +{WIN_STAY_BOOST:.2%}")
    print(f"  LOSE_SHIFT_BOOST: -{LOSE_SHIFT_BOOST:.2%}")
    
    print("\nEvaluating on human data...")
    results = evaluate_fast_model(args.data_dir, args.verbose)
    
    if not results:
        print("Evaluation failed!")
        return
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%} (+/- {results['accuracy_std']:.2%})")
    print(f"Sessions Evaluated: {results['n_sessions']}")
    print(f"Total Trials: {results['total_trials']}")
    
    comparison = compare_with_neural(results)
    if comparison:
        print("\n" + "-" * 40)
        print("Comparison with Neural Model:")
        print(f"  Fast Model:   {comparison['fast_accuracy']:.2%}")
        print(f"  Neural Model: {comparison['neural_accuracy']:.2%}")
        print(f"  Gap:          {comparison['accuracy_gap']:.2%}")
    
    print("\n" + "=" * 60)
    print("Rule Breakdown:")
    print("=" * 60)
    print("""
    1. First trial: Choose LEFT with {:.0%} probability
    
    2. After winning (reward=1):
       Stay probability = {:.0%} + {:.0%} = {:.0%}
    
    3. After losing (reward=0):
       Stay probability = {:.0%} - {:.0%} = {:.0%}
    
    4. If one side has >10% higher reward rate:
       Bias toward that side
    """.format(
        LEFT_BIAS,
        STAY_PROB, WIN_STAY_BOOST, STAY_PROB + WIN_STAY_BOOST,
        STAY_PROB, LOSE_SHIFT_BOOST, STAY_PROB - LOSE_SHIFT_BOOST
    ))
    
    output_path = os.path.join(SCRIPT_DIR, 'evaluation_results', 'fast_model_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable = {
        'overall_accuracy': float(results['overall_accuracy']),
        'accuracy_std': float(results['accuracy_std']),
        'n_sessions': results['n_sessions'],
        'total_trials': results['total_trials'],
        'parameters': {
            'LEFT_BIAS': LEFT_BIAS,
            'STAY_PROB': STAY_PROB,
            'WIN_STAY_BOOST': WIN_STAY_BOOST,
            'LOSE_SHIFT_BOOST': LOSE_SHIFT_BOOST
        }
    }
    
    if comparison:
        serializable['comparison'] = comparison
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
