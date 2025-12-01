#!/usr/bin/env python3
"""
Human Behavioral Data Loader
Loads and preprocesses human subject data for neural network training.

Data format expected:
- CSV files with columns: trial_number, time, schedule_type, schedule_name, 
  is_biased_choice, side_choice, RT, observed_reward, unobserved_reward, 
  biased_reward, unbiased_reward
"""
import os
import glob
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
HUMAN_DATA_DIR = os.path.join(os.path.dirname(__file__), "human_data")

class HumanDataLoader:
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.action_map = {"LEFT": 0, "RIGHT": 1}
        self.reverse_action_map = {0: "LEFT", 1: "RIGHT"}
        
    def load_all_human_data(self, data_dir: Optional[str] = None) -> List[pd.DataFrame]:
        """Load all CSV files from human data directories."""
        if data_dir is None:
            data_dir = RESULTS_DIR
            
        all_data = []
        csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df.columns = df.columns.str.strip()
                if 'side_choice' in df.columns and len(df) >= 5:
                    df['source_file'] = csv_file
                    all_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {csv_file}: {e}")
                
        return all_data
    
    def preprocess_session(self, df: pd.DataFrame) -> Dict:
        """Preprocess a single session into features and labels."""
        df = df.copy()
        df['action'] = df['side_choice'].map(self.action_map)
        df['reward'] = df['observed_reward'].astype(float)
        
        if 'RT' in df.columns:
            df['rt_normalized'] = (df['RT'] - df['RT'].mean()) / (df['RT'].std() + 1e-8)
        else:
            df['rt_normalized'] = 0.0
            
        df['cumulative_reward'] = df['reward'].cumsum()
        df['left_count'] = (df['action'] == 0).cumsum()
        df['right_count'] = (df['action'] == 1).cumsum()
        df['left_reward'] = (df['reward'] * (df['action'] == 0)).cumsum()
        df['right_reward'] = (df['reward'] * (df['action'] == 1)).cumsum()
        
        return {
            'actions': df['action'].values,
            'rewards': df['reward'].values,
            'rt_normalized': df['rt_normalized'].values,
            'cumulative_reward': df['cumulative_reward'].values,
            'left_count': df['left_count'].values,
            'right_count': df['right_count'].values,
            'left_reward': df['left_reward'].values,
            'right_reward': df['right_reward'].values,
            'n_trials': len(df)
        }
    
    def create_sequences(self, session_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Create training sequences from preprocessed session data."""
        n_trials = session_data['n_trials']
        sequences = []
        labels = []
        
        for t in range(1, n_trials):
            start_idx = max(0, t - self.sequence_length)
            
            seq_actions = np.zeros(self.sequence_length)
            seq_rewards = np.zeros(self.sequence_length)
            seq_rt = np.zeros(self.sequence_length)
            
            history_len = t - start_idx
            offset = self.sequence_length - history_len
            
            seq_actions[offset:] = session_data['actions'][start_idx:t]
            seq_rewards[offset:] = session_data['rewards'][start_idx:t]
            seq_rt[offset:] = session_data['rt_normalized'][start_idx:t]
            
            total_trials = t
            left_count = session_data['left_count'][t-1]
            right_count = session_data['right_count'][t-1]
            left_reward = session_data['left_reward'][t-1]
            right_reward = session_data['right_reward'][t-1]
            
            left_rate = left_count / total_trials if total_trials > 0 else 0.5
            right_rate = right_count / total_trials if total_trials > 0 else 0.5
            left_avg_reward = left_reward / left_count if left_count > 0 else 0.5
            right_avg_reward = right_reward / right_count if right_count > 0 else 0.5
            
            global_features = np.array([
                total_trials / 100.0,
                left_rate,
                right_rate,
                left_avg_reward,
                right_avg_reward,
                session_data['cumulative_reward'][t-1] / (total_trials + 1)
            ])
            
            feature_matrix = np.stack([
                seq_actions,
                seq_rewards,
                seq_rt
            ], axis=-1)
            
            sequences.append({
                'sequence': feature_matrix,
                'global': global_features
            })
            labels.append(session_data['actions'][t])
            
        return sequences, labels
    
    def prepare_training_data(self, data_dir: Optional[str] = None) -> Tuple[List, np.ndarray]:
        """Prepare all data for training."""
        all_sessions = self.load_all_human_data(data_dir)
        
        all_sequences = []
        all_labels = []
        
        for df in all_sessions:
            session_data = self.preprocess_session(df)
            sequences, labels = self.create_sequences(session_data)
            all_sequences.extend(sequences)
            all_labels.extend(labels)
            
        return all_sequences, np.array(all_labels)
    
    def get_statistics(self, data_dir: Optional[str] = None) -> Dict:
        """Get statistics about the human data."""
        all_sessions = self.load_all_human_data(data_dir)
        
        total_trials = sum(len(df) for df in all_sessions)
        total_sessions = len(all_sessions)
        
        all_actions = []
        all_rewards = []
        for df in all_sessions:
            df.columns = df.columns.str.strip()
            all_actions.extend(df['side_choice'].values)
            all_rewards.extend(df['observed_reward'].values)
            
        left_rate = sum(1 for a in all_actions if a == 'LEFT') / len(all_actions) if all_actions else 0
        avg_reward = np.mean(all_rewards) if all_rewards else 0
        
        return {
            'total_sessions': total_sessions,
            'total_trials': total_trials,
            'avg_trials_per_session': total_trials / total_sessions if total_sessions > 0 else 0,
            'left_choice_rate': left_rate,
            'right_choice_rate': 1 - left_rate,
            'avg_reward_rate': avg_reward
        }


if __name__ == "__main__":
    loader = HumanDataLoader(sequence_length=10)
    stats = loader.get_statistics()
    print("Human Data Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
