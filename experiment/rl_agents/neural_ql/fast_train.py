#!/usr/bin/env python3
"""
Fast Training Script - Uses data sampling and optimizations for quicker training.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from datetime import datetime
import random

from model import NeuralQLModel
from data_loader import HumanDataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)


class HumanBehaviorDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        return (
            torch.FloatTensor(seq_data['sequence']),
            torch.FloatTensor(seq_data['global']),
            torch.LongTensor([self.labels[idx]])[0]
        )


def train_fast(sample_ratio=0.3, epochs=30, batch_size=512, lr=0.001):
    print("=" * 50)
    print("Fast Neural Q-Learning Training")
    print("=" * 50)
    print(f"Sample ratio: {sample_ratio:.0%}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print()
    
    loader = HumanDataLoader(sequence_length=10)
    sequences, labels = loader.prepare_training_data()
    
    total_samples = len(sequences)
    sample_size = int(total_samples * sample_ratio)
    
    indices = random.sample(range(total_samples), sample_size)
    sequences = [sequences[i] for i in indices]
    labels = labels[indices]
    
    print(f"Using {sample_size} samples (from {total_samples} total)")
    
    dataset = HumanBehaviorDataset(sequences, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = NeuralQLModel(
        input_dim=3,
        hidden_dim=128,
        num_lstm_layers=2,
        global_feature_dim=6,
        dropout=0.2
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_model_state = None
    
    print("\nTraining...")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        for sequence, global_features, labels_batch in train_loader:
            sequence = sequence.to(DEVICE)
            global_features = global_features.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            
            optimizer.zero_grad()
            loss, loss_dict = model.compute_loss(sequence, global_features, labels_batch, bc_weight=1.0, q_weight=0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_correct += loss_dict['accuracy'] * len(labels_batch)
            train_total += len(labels_batch)
        
        scheduler.step()
        train_acc = train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequence, global_features, labels_batch in val_loader:
                sequence = sequence.to(DEVICE)
                global_features = global_features.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)
                
                q_values, policy_probs = model(sequence, global_features)
                preds = policy_probs.argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += len(labels_batch)
        
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
    
    print("-" * 50)
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"neural_ql_{timestamp}.pt")
    
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_kwargs': {
            'input_dim': 3,
            'hidden_dim': 128,
            'num_lstm_layers': 2,
            'global_feature_dim': 6,
            'dropout': 0.2
        },
        'use_ensemble': False,
        'num_ensemble_models': 1,
        'sequence_length': 10,
        'best_val_acc': best_val_acc
    }
    
    torch.save(save_data, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model_path, best_val_acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_ratio', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    train_fast(
        sample_ratio=args.sample_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
