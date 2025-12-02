#!/usr/bin/env python3
"""
Universal Model Evaluation Tool
Tests any model (neural network or rule-based) against human behavioral data.

Usage:
    python evaluate_all.py --model neural     # Test neural network model
    python evaluate_all.py --model fast       # Test fast rule-based model
    python evaluate_all.py --model all        # Test all models and compare
    
Example:
    python evaluate_all.py --model all --verbose
"""
import os
import sys
import json
import random
import argparse
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from data_loader import HumanDataLoader

FAST_PARAMS = {
    'LEFT_BIAS': 0.62,
    'STAY_PROB': 0.66,
    'WIN_STAY_BOOST': 0.15,
    'LOSE_SHIFT_BOOST': 0.10
}


class FastModel:
    """Rule-based fast model."""
    
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
    
    def predict(self, history: Dict = None) -> int:
        """Predict next action. Returns 0=LEFT, 1=RIGHT."""
        if self.trial == 0 or self.last_action is None:
            return 0 if random.random() < FAST_PARAMS['LEFT_BIAS'] else 1
        
        stay_prob = FAST_PARAMS['STAY_PROB']
        
        if self.last_reward == 1:
            stay_prob += FAST_PARAMS['WIN_STAY_BOOST']
        else:
            stay_prob -= FAST_PARAMS['LOSE_SHIFT_BOOST']
        
        left_count = max(self.left_count, 1)
        right_count = max(self.right_count, 1)
        left_rate = self.left_rewards / left_count
        right_rate = self.right_rewards / right_count
        
        if abs(left_rate - right_rate) > 0.1:
            if left_rate > right_rate:
                stay_prob = stay_prob if self.last_action == 0 else (1 - stay_prob)
            else:
                stay_prob = stay_prob if self.last_action == 1 else (1 - stay_prob)
        
        stay_prob = max(0.1, min(0.9, stay_prob))
        
        if random.random() < stay_prob:
            return self.last_action
        else:
            return 1 - self.last_action
    
    def update(self, action: int, reward: float):
        """Update state after action."""
        self.last_action = action
        self.last_reward = reward
        
        if action == 0:
            self.left_count += 1
            self.left_rewards += reward
        else:
            self.right_count += 1
            self.right_rewards += reward
        
        self.trial += 1


class NeuralModel:
    """Wrapper for PyTorch neural network model."""
    
    def __init__(self, model_name: str = None):
        import torch
        from model import NeuralQLModel, ModelEnsemble
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.seq_length = self._load_model(model_name)
        self.reset()
    
    def _load_model(self, model_name: str = None):
        import torch
        from model import NeuralQLModel, ModelEnsemble
        
        MODEL_DIR = os.path.join(SCRIPT_DIR, "trained_models")
        
        if model_name is None:
            latest_file = os.path.join(MODEL_DIR, "latest_model.txt")
            if not os.path.exists(latest_file):
                raise FileNotFoundError("No trained model found.")
            with open(latest_file, 'r') as f:
                model_name = f.read().strip()
        
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model_kwargs = checkpoint['model_kwargs']
        use_ensemble = checkpoint.get('use_ensemble', False)
        
        if use_ensemble:
            num_models = checkpoint.get('num_ensemble_models', 3)
            model = ModelEnsemble(num_models=num_models, **model_kwargs)
        else:
            model = NeuralQLModel(**model_kwargs)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        seq_length = checkpoint.get('sequence_length', 10)
        return model, seq_length
    
    def reset(self):
        self.history = {
            'actions': [],
            'rewards': [],
            'left_count': 0,
            'right_count': 0,
            'left_reward': 0.0,
            'right_reward': 0.0,
            'total_reward': 0.0
        }
    
    def predict(self, history: Dict = None) -> int:
        """Predict next action. Returns 0=LEFT, 1=RIGHT."""
        import torch
        
        seq_actions = np.zeros(self.seq_length)
        seq_rewards = np.zeros(self.seq_length)
        seq_rt = np.zeros(self.seq_length)
        
        history_len = min(len(self.history['actions']), self.seq_length)
        if history_len > 0:
            offset = self.seq_length - history_len
            seq_actions[offset:] = self.history['actions'][-history_len:]
            seq_rewards[offset:] = self.history['rewards'][-history_len:]
        
        sequence = np.stack([seq_actions, seq_rewards, seq_rt], axis=-1)
        
        total_trials = len(self.history['actions'])
        if total_trials > 0:
            left_rate = self.history['left_count'] / total_trials
            right_rate = self.history['right_count'] / total_trials
            left_avg = self.history['left_reward'] / self.history['left_count'] if self.history['left_count'] > 0 else 0.5
            right_avg = self.history['right_reward'] / self.history['right_count'] if self.history['right_count'] > 0 else 0.5
            avg_reward = self.history['total_reward'] / total_trials
        else:
            left_rate = right_rate = 0.5
            left_avg = right_avg = avg_reward = 0.5
        
        global_features = np.array([
            total_trials / 100.0,
            left_rate, right_rate,
            left_avg, right_avg,
            avg_reward
        ])
        
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        global_tensor = torch.FloatTensor(global_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, action_probs = self.model(sequence_tensor, global_tensor)
            pred = action_probs.argmax(dim=-1).item()
        
        return pred
    
    def update(self, action: int, reward: float):
        """Update state after action."""
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['total_reward'] += reward
        
        if action == 0:
            self.history['left_count'] += 1
            self.history['left_reward'] += reward
        else:
            self.history['right_count'] += 1
            self.history['right_reward'] += reward


def evaluate_model(model, sessions: List, model_name: str, n_runs: int = 1, verbose: bool = False) -> Dict:
    """
    Evaluate a model on human session data.
    
    Args:
        model: Model instance with predict() and update() methods
        sessions: List of preprocessed session data
        model_name: Name for display
        n_runs: Number of runs (for stochastic models)
        verbose: Print progress
    
    Returns:
        Evaluation results dictionary
    """
    all_run_accuracies = []
    
    for run in range(n_runs):
        session_correct = 0
        session_total = 0
        
        for i, session_data in enumerate(sessions):
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
            
            if verbose and (i + 1) % 200 == 0:
                print(f"  [{model_name}] Processed {i+1}/{len(sessions)} sessions...")
        
        if session_total > 0:
            run_accuracy = session_correct / session_total
            all_run_accuracies.append(run_accuracy)
    
    if len(all_run_accuracies) == 0:
        return {'accuracy': 0, 'std': 0, 'n_sessions': 0, 'n_trials': 0}
    
    return {
        'accuracy': np.mean(all_run_accuracies),
        'std': np.std(all_run_accuracies) if len(all_run_accuracies) > 1 else 0,
        'n_sessions': len(sessions),
        'n_trials': session_total
    }


def main():
    parser = argparse.ArgumentParser(description="Universal Model Evaluation")
    parser.add_argument("--model", type=str, default="all", 
                        choices=["neural", "fast", "all"],
                        help="Model to evaluate")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs for stochastic models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("Universal Model Evaluation")
    print("=" * 60)
    
    print("\nLoading human data...")
    loader = HumanDataLoader(sequence_length=10)
    all_sessions_df = loader.load_all_human_data(args.data_dir)
    
    if not all_sessions_df:
        print("Error: No human data found!")
        return
    
    sessions = []
    for df in all_sessions_df:
        session_data = loader.preprocess_session(df)
        if len(session_data['actions']) > 0:
            sessions.append(session_data)
    
    print(f"Loaded {len(sessions)} valid sessions")
    total_trials = sum(len(s['actions']) for s in sessions)
    print(f"Total trials: {total_trials}")
    
    results = {}
    
    if args.model in ["fast", "all"]:
        print("\n" + "-" * 40)
        print("Evaluating Fast Model...")
        print(f"  Parameters: LEFT_BIAS={FAST_PARAMS['LEFT_BIAS']:.0%}, "
              f"STAY={FAST_PARAMS['STAY_PROB']:.0%}, "
              f"WIN_STAY=+{FAST_PARAMS['WIN_STAY_BOOST']:.0%}, "
              f"LOSE_SHIFT=-{FAST_PARAMS['LOSE_SHIFT_BOOST']:.0%}")
        
        fast_model = FastModel()
        fast_results = evaluate_model(
            fast_model, sessions, "Fast", 
            n_runs=args.runs, verbose=args.verbose
        )
        results['fast'] = fast_results
        print(f"  Accuracy: {fast_results['accuracy']:.2%} (+/- {fast_results['std']:.2%})")
    
    if args.model in ["neural", "all"]:
        print("\n" + "-" * 40)
        print("Evaluating Neural Model...")
        
        try:
            neural_model = NeuralModel()
            neural_results = evaluate_model(
                neural_model, sessions, "Neural",
                n_runs=1, verbose=args.verbose
            )
            results['neural'] = neural_results
            print(f"  Accuracy: {neural_results['accuracy']:.2%}")
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            results['neural'] = {'accuracy': 0, 'error': str(e)}
        except Exception as e:
            print(f"  Error loading neural model: {e}")
            results['neural'] = {'accuracy': 0, 'error': str(e)}
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n{:<20} {:<15} {:<15}".format("Model", "Accuracy", "Std Dev"))
    print("-" * 50)
    
    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<20} {'ERROR':<15} {'-':<15}")
        else:
            std_str = f"+/- {res['std']:.2%}" if res['std'] > 0 else "-"
            print(f"{name:<20} {res['accuracy']:.2%}          {std_str}")
    
    if 'fast' in results and 'neural' in results and 'error' not in results['neural']:
        gap = results['neural']['accuracy'] - results['fast']['accuracy']
        print(f"\nNeural vs Fast gap: {gap:+.2%}")
    
    output_path = os.path.join(SCRIPT_DIR, 'evaluation_results', 'all_models_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable = {}
    for name, res in results.items():
        serializable[name] = {
            'accuracy': float(res.get('accuracy', 0)),
            'std': float(res.get('std', 0)),
            'n_sessions': res.get('n_sessions', 0),
            'n_trials': res.get('n_trials', 0)
        }
        if 'error' in res:
            serializable[name]['error'] = res['error']
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
