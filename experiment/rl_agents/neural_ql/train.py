#!/usr/bin/env python3
"""
Training Script for Neural Q-Learning Model
Trains the model on human behavioral data using behavioral cloning with Q-learning regularization.

Usage:
    python train.py [--epochs 100] [--lr 0.001] [--batch_size 32]
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime
from typing import List, Tuple, Dict

from model import NeuralQLModel, ModelEnsemble
from data_loader import HumanDataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)


class HumanBehaviorDataset(Dataset):
    """PyTorch Dataset for human behavioral data."""
    
    def __init__(self, sequences: List[Dict], labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_data = self.sequences[idx]
        return (
            torch.FloatTensor(seq_data['sequence']),
            torch.FloatTensor(seq_data['global']),
            torch.LongTensor([self.labels[idx]])[0]
        )


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_bc_loss = 0
    total_correct = 0
    total_samples = 0
    
    for sequence, global_features, labels in dataloader:
        sequence = sequence.to(DEVICE)
        global_features = global_features.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        loss, loss_dict = model.compute_loss(
            sequence, global_features, labels,
            bc_weight=1.0, q_weight=0.0
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        total_bc_loss += loss_dict['bc_loss'] * len(labels)
        total_correct += loss_dict['accuracy'] * len(labels)
        total_samples += len(labels)
        
    if scheduler is not None:
        scheduler.step()
        
    return {
        'loss': total_loss / total_samples,
        'bc_loss': total_bc_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def evaluate(model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequence, global_features, labels in dataloader:
            sequence = sequence.to(DEVICE)
            global_features = global_features.to(DEVICE)
            labels = labels.to(DEVICE)
            
            _, action_probs = model(sequence, global_features)
            
            loss = nn.functional.cross_entropy(action_probs, labels)
            preds = action_probs.argmax(dim=-1)
            
            total_loss += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    left_mask = all_labels == 0
    right_mask = all_labels == 1
    
    left_acc = (all_preds[left_mask] == all_labels[left_mask]).mean() if left_mask.sum() > 0 else 0
    right_acc = (all_preds[right_mask] == all_labels[right_mask]).mean() if right_mask.sum() > 0 else 0
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'left_accuracy': left_acc,
        'right_accuracy': right_acc
    }


def train_model(
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    hidden_dim: int = 128,
    num_lstm_layers: int = 2,
    sequence_length: int = 10,
    patience: int = 15,
    use_ensemble: bool = False,
    num_ensemble_models: int = 3,
    data_dir: str = None
) -> str:
    """
    Main training function.
    
    Returns:
        Path to saved model
    """
    print("=" * 60)
    print("Neural Q-Learning Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}, LR: {lr}, Batch Size: {batch_size}")
    print(f"Hidden Dim: {hidden_dim}, LSTM Layers: {num_lstm_layers}")
    print(f"Sequence Length: {sequence_length}")
    print()
    
    print("Loading human behavioral data...")
    loader = HumanDataLoader(sequence_length=sequence_length)
    stats = loader.get_statistics(data_dir)
    print(f"  Sessions: {stats['total_sessions']}")
    print(f"  Total trials: {stats['total_trials']}")
    print(f"  Left choice rate: {stats['left_choice_rate']:.2%}")
    print(f"  Avg reward rate: {stats['avg_reward_rate']:.2%}")
    print()
    
    if stats['total_sessions'] == 0:
        print("ERROR: No human data found!")
        print("Please add CSV files to the results/ directory")
        return None
        
    print("Preparing training sequences...")
    sequences, labels = loader.prepare_training_data(data_dir)
    print(f"  Total sequences: {len(sequences)}")
    print()
    
    dataset = HumanBehaviorDataset(sequences, labels)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    model_kwargs = {
        'input_dim': 3,
        'hidden_dim': hidden_dim,
        'num_lstm_layers': num_lstm_layers,
        'global_feature_dim': 6,
        'dropout': 0.2
    }
    
    if use_ensemble:
        model = ModelEnsemble(num_models=num_ensemble_models, **model_kwargs)
        print(f"Using ensemble of {num_ensemble_models} models")
    else:
        model = NeuralQLModel(**model_kwargs)
        
    model = model.to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=patience)
    
    best_val_acc = 0
    best_model_state = None
    training_history = []
    
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler)
        val_metrics = evaluate(model, val_loader)
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'lr': scheduler.get_last_lr()[0]
        })
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2%} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.2%}")
                  
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
            
    print("-" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"neural_ql_{timestamp}"
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
    
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_kwargs': model_kwargs,
        'use_ensemble': use_ensemble,
        'num_ensemble_models': num_ensemble_models if use_ensemble else 1,
        'sequence_length': sequence_length,
        'training_config': {
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'patience': patience
        },
        'data_stats': stats,
        'best_val_acc': best_val_acc,
        'training_history': training_history
    }
    
    torch.save(save_data, model_path)
    print(f"Model saved to: {model_path}")
    
    config_path = os.path.join(MODEL_DIR, f"{model_name}_config.json")
    with open(config_path, 'w') as f:
        config = {
            'model_name': model_name,
            'model_path': model_path,
            'sequence_length': sequence_length,
            'hidden_dim': hidden_dim,
            'num_lstm_layers': num_lstm_layers,
            'use_ensemble': use_ensemble,
            'best_val_acc': best_val_acc,
            'data_stats': stats,
            'trained_at': timestamp
        }
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    with open(os.path.join(MODEL_DIR, "latest_model.txt"), 'w') as f:
        f.write(model_name)
        
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train Neural Q-Learning Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lstm_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--seq_length", type=int, default=10, help="Sequence length for history")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--ensemble", action="store_true", help="Use model ensemble")
    parser.add_argument("--num_models", type=int, default=3, help="Number of ensemble models")
    parser.add_argument("--data_dir", type=str, default=None, help="Custom data directory")
    
    args = parser.parse_args()
    
    model_path = train_model(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_lstm_layers=args.lstm_layers,
        sequence_length=args.seq_length,
        patience=args.patience,
        use_ensemble=args.ensemble,
        num_ensemble_models=args.num_models,
        data_dir=args.data_dir
    )
    
    if model_path:
        print("\nTraining complete!")
        print(f"To use this model, run the neural_ql_agent.py with:")
        print(f"  python neural_ql_agent.py <user_id> <last_action> <last_reward>")


if __name__ == "__main__":
    main()
