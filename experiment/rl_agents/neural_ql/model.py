#!/usr/bin/env python3
"""
Neural Network Enhanced Q-Learning Model
Uses LSTM with Attention mechanism to model human decision-making patterns.

Architecture:
1. Sequence Encoder (LSTM): Captures temporal patterns in choice history
2. Self-Attention: Weighs importance of different time steps
3. Global Feature Network: Processes aggregate statistics
4. Fusion Layer: Combines sequence and global representations
5. Q-Value Head: Outputs Q-values for LEFT and RIGHT actions
6. Policy Head: Outputs action probabilities (for behavioral cloning)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class AttentionLayer(nn.Module):
    """Self-attention mechanism to focus on important time steps."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = np.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        return self.output(attention_output)


class NeuralQLModel(nn.Module):
    """
    Neural Network Enhanced Q-Learning Model for Human Behavior Modeling.
    
    This model combines:
    1. Recurrent processing (LSTM) to capture sequential dependencies
    2. Self-attention to focus on relevant historical decisions
    3. Global feature processing for aggregate statistics
    4. Dual heads for Q-values and action probabilities
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        num_attention_heads: int = 4,
        global_feature_dim: int = 6,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        self.attention = AttentionLayer(hidden_dim, num_attention_heads)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        self.global_network = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        fusion_dim = hidden_dim + hidden_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        
    def forward(
        self, 
        sequence: torch.Tensor, 
        global_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            sequence: (batch, seq_len, input_dim) - Historical action/reward sequences
            global_features: (batch, global_dim) - Aggregate statistics
            return_attention: Whether to return attention weights
            
        Returns:
            q_values: (batch, 2) - Q-values for LEFT and RIGHT
            action_probs: (batch, 2) - Action probabilities
        """
        x = self.input_projection(sequence)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        attended = self.attention(lstm_out)
        attended = self.attention_norm(attended + lstm_out)
        
        sequence_repr = attended[:, -1, :]
        
        global_repr = self.global_network(global_features)
        
        combined = torch.cat([sequence_repr, global_repr], dim=-1)
        fused = self.fusion(combined)
        
        q_values = self.q_head(fused)
        
        policy_logits = self.policy_head(fused)
        action_probs = F.softmax(policy_logits / self.temperature, dim=-1)
        
        return q_values, action_probs
    
    def get_action(self, sequence: torch.Tensor, global_features: torch.Tensor, 
                   use_policy: bool = True, epsilon: float = 0.0) -> int:
        """
        Get action for a single state.
        
        Args:
            sequence: (1, seq_len, input_dim)
            global_features: (1, global_dim)
            use_policy: If True, sample from policy; if False, use Q-values
            epsilon: Exploration rate
            
        Returns:
            action: 0 (LEFT) or 1 (RIGHT)
        """
        self.eval()
        with torch.no_grad():
            q_values, action_probs = self.forward(sequence, global_features)
            
            if np.random.random() < epsilon:
                return np.random.randint(2)
                
            if use_policy:
                action = torch.multinomial(action_probs, 1).item()
            else:
                action = q_values.argmax(dim=-1).item()
                
        return action
    
    def compute_loss(
        self,
        sequence: torch.Tensor,
        global_features: torch.Tensor,
        target_actions: torch.Tensor,
        target_rewards: Optional[torch.Tensor] = None,
        bc_weight: float = 1.0,
        q_weight: float = 0.5
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss for training.
        
        Combines:
        1. Behavioral Cloning loss (cross-entropy with human actions)
        2. Q-learning loss (if rewards provided)
        
        Args:
            sequence: (batch, seq_len, input_dim)
            global_features: (batch, global_dim)
            target_actions: (batch,) - Human actions
            target_rewards: (batch,) - Rewards received (optional)
            bc_weight: Weight for behavioral cloning loss
            q_weight: Weight for Q-learning loss
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        q_values, action_probs = self.forward(sequence, global_features)
        
        bc_loss = F.cross_entropy(action_probs, target_actions)
        
        loss_dict = {'bc_loss': bc_loss.item()}
        total_loss = bc_weight * bc_loss
        
        if target_rewards is not None:
            selected_q = q_values.gather(1, target_actions.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(selected_q, target_rewards)
            total_loss = total_loss + q_weight * q_loss
            loss_dict['q_loss'] = q_loss.item()
            
        loss_dict['total_loss'] = total_loss.item()
        
        with torch.no_grad():
            predicted = action_probs.argmax(dim=-1)
            accuracy = (predicted == target_actions).float().mean()
            loss_dict['accuracy'] = accuracy.item()
            
        return total_loss, loss_dict


class ModelEnsemble(nn.Module):
    """Ensemble of multiple NeuralQL models for improved robustness."""
    
    def __init__(self, num_models: int = 3, **model_kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            NeuralQLModel(**model_kwargs) for _ in range(num_models)
        ])
        self.num_models = num_models
        
    def forward(self, sequence: torch.Tensor, global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average predictions from all models."""
        q_values_list = []
        probs_list = []
        
        for model in self.models:
            q, p = model(sequence, global_features)
            q_values_list.append(q)
            probs_list.append(p)
            
        avg_q = torch.stack(q_values_list).mean(dim=0)
        avg_probs = torch.stack(probs_list).mean(dim=0)
        
        return avg_q, avg_probs
    
    def get_action(self, sequence: torch.Tensor, global_features: torch.Tensor, 
                   use_policy: bool = True, epsilon: float = 0.0) -> int:
        """Get action using ensemble voting."""
        if np.random.random() < epsilon:
            return np.random.randint(2)
            
        votes = []
        for model in self.models:
            action = model.get_action(sequence, global_features, use_policy, epsilon=0)
            votes.append(action)
            
        return max(set(votes), key=votes.count)


if __name__ == "__main__":
    model = NeuralQLModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    batch_size = 4
    seq_len = 10
    sequence = torch.randn(batch_size, seq_len, 3)
    global_features = torch.randn(batch_size, 6)
    
    q_values, action_probs = model(sequence, global_features)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Action probs shape: {action_probs.shape}")
    print(f"Sample action probs: {action_probs[0].detach().numpy()}")
