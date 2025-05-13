#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Omic Transformer Implementation

This code implements a transformer-based deep learning model for integrating
spectral and metabolite data from plant samples. The model employs a cross-attention
mechanism to discover relationships between different omics data types and
analyze plant stress responses across different genotypes and treatments.

Features:
- Integration of spectral and metabolite data
- Cross-attention for multi-omic data fusion
- Multi-task classification of genotype, treatment, and time points
- Attention-based interpretability for feature importance
- Comparison with traditional machine learning baselines

Usage:
    python transformer_implementation.py --tissue leaf
    or
    python transformer_implementation.py --tissue root

Date: March 31, 2025
"""

import os
import time
import json
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

# For scaling and encoding
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

# Baseline models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F


# ===== CONFIGURATION =====
class TransformerConfig:
    """Configuration for the Multi-Omic Transformer"""
    
    def __init__(self, tissue: str = "leaf"):
        # Paths
        self.output_dir = "C:/Users/ms/Desktop/hyper/output/transformer"
        self.figure_dir = os.path.join(self.output_dir, "figures")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        
        # Ensure all directories exist
        for dir_path in [self.output_dir, self.figure_dir, self.logs_dir, self.checkpoints_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Tissue type (leaf or root)
        self.tissue = tissue.lower()
        if self.tissue not in ["leaf", "root"]:
            raise ValueError("Tissue must be either 'leaf' or 'root'")
            
        # Data paths
        self.mofa_dir = "C:/Users/ms/Desktop/hyper/output/mofa"
        self.spectral_path = os.path.join(self.mofa_dir, f"transformer_input_{self.tissue}_spectral.csv")
        self.metabolite_path = os.path.join(self.mofa_dir, f"transformer_input_{self.tissue}_metabolite.csv")
        
        # Metadata columns
        self.metadata_columns = [
            "Row_names", "Vac_id", "Genotype", "Entry", "Tissue.type", 
            "Batch", "Treatment", "Replication", "Day"
        ]
        
        # Target columns for multi-task classification
        self.target_columns = ["Genotype", "Treatment", "Day"]
        
        # Target encoding maps
        self.target_encoders = {
            "Genotype": {"G1": 0, "G2": 1},
            "Treatment": {"T0": 0, "T1": 1}
            # Day is numeric and will be encoded as 0, 1, 2
        }
        
        # Train/val/test split ratios
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Cross-validation
        self.use_cross_validation = False  # Start with False, implement CV later
        self.n_splits = 5
        
        # Model hyperparameters
        self.hidden_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.dropout = 0.1
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        
        # Training parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.early_stopping_patience = 10
        self.num_workers = 8
        
        # Analysis parameters
        self.attention_threshold = 0.1  # Threshold for strong attention pairs
        self.top_n_pairs = 100  # Number of top pairs to report
        
        # Random seed for reproducibility
        self.seed = 42


# ===== DATA LOADING =====
def load_and_preprocess_data(config: TransformerConfig) -> Dict[str, Any]:
    """
    Load and preprocess spectral and metabolite data for the specified tissue.
    
    Args:
        config: Configuration object with data paths and parameters
    
    Returns:
        Dictionary containing processed data and metadata
    """
    logging.info(f"Loading {config.tissue} data...")
    
    # Load spectral and metabolite data
    spectral_df = pd.read_csv(config.spectral_path)
    metabolite_df = pd.read_csv(config.metabolite_path)
    
    logging.info(f"Spectral data shape: {spectral_df.shape}")
    logging.info(f"Metabolite data shape: {metabolite_df.shape}")
    
    # Check that both dataframes have the same shape
    if spectral_df.shape[0] != metabolite_df.shape[0]:
        raise ValueError(f"Different number of samples: Spectral ({spectral_df.shape[0]}) vs Metabolite ({metabolite_df.shape[0]})")
    
    # Create a mapping between different naming conventions for later reference
    sample_mapping = pd.DataFrame({
        'spectral_id': spectral_df['Row_names'].values,
        'metabolite_id': metabolite_df['Row_names'].values,
        'index': range(len(spectral_df))
    })
    
    logging.info(f"Created sample mapping between spectral and metabolite IDs")
    
    # Extract metadata - use spectral dataframe's metadata since we're matching positions
    metadata = spectral_df[config.metadata_columns].copy()
    
    # Extract feature columns
    spectral_features = spectral_df.drop(columns=config.metadata_columns)
    metabolite_features = metabolite_df.drop(columns=config.metadata_columns)
    
    # Get feature names for later analysis
    spectral_feature_names = spectral_features.columns.tolist()
    metabolite_feature_names = metabolite_features.columns.tolist()
    
    # Get sample IDs for reference
    sample_ids = metadata['Row_names'].values
    
    # Encode target variables
    target_data = {}
    for col in config.target_columns:
        if col == 'Day':
            # Subtract 1 to make Days 1,2,3 into 0,1,2 (will use as classifier targets)
            target_data[col] = metadata[col].astype(int) - 1
        elif col in config.target_encoders:
            encoder = config.target_encoders[col]
            target_data[col] = metadata[col].map(encoder)
        else:
            # Fallback - shouldn't reach here given our configuration
            encoder = LabelEncoder()
            target_data[col] = encoder.fit_transform(metadata[col])
    
    # Convert to numpy arrays for PyTorch
    spectral_data = spectral_features.values.astype(np.float32)
    metabolite_data = metabolite_features.values.astype(np.float32)
    
    # Create target array for multi-task learning
    target_arrays = [target_data[col].values for col in config.target_columns]
    target_array = np.column_stack(target_arrays).astype(np.int64)
    
    # Standard scale features (will fit on train, transform on val/test later)
    spectral_scaler = StandardScaler()
    metabolite_scaler = StandardScaler()
    
    # Create train/val/test splits with stratification
    # Use Day as stratification target (most balanced classes)
    train_idx, temp_idx = train_test_split(
        np.arange(len(sample_ids)), 
        test_size=(config.val_ratio + config.test_ratio),
        random_state=config.seed,
        stratify=target_data['Day']
    )
    
    # Further split temp into validation and test
    relative_test_ratio = config.test_ratio / (config.val_ratio + config.test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        random_state=config.seed,
        stratify=target_data['Day'].iloc[temp_idx]
    )
    
    # Log split sizes
    logging.info(f"Train set: {len(train_idx)} samples")
    logging.info(f"Validation set: {len(val_idx)} samples")
    logging.info(f"Test set: {len(test_idx)} samples")
    
    # Fit scalers on training data
    spectral_scaler.fit(spectral_data[train_idx])
    metabolite_scaler.fit(metabolite_data[train_idx])
    
    # Transform all data
    spectral_data_scaled = spectral_scaler.transform(spectral_data)
    metabolite_data_scaled = metabolite_scaler.transform(metabolite_data)
    
    # Pack everything into a dictionary
    data_dict = {
        'spectral_data': spectral_data_scaled,
        'metabolite_data': metabolite_data_scaled,
        'target_data': target_array,
        'sample_ids': sample_ids,
        'sample_mapping': sample_mapping,
        'metadata': metadata,
        'spectral_feature_names': spectral_feature_names,
        'metabolite_feature_names': metabolite_feature_names,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'spectral_scaler': spectral_scaler,
        'metabolite_scaler': metabolite_scaler
    }
    
    return data_dict


class MultiOmicDataset(Dataset):
    """Dataset for paired spectral and metabolite data"""
    
    def __init__(self, 
                 spectral_data: np.ndarray, 
                 metabolite_data: np.ndarray, 
                 targets: np.ndarray,
                 indices: Optional[np.ndarray] = None):
        """
        Initialize the dataset.
        
        Args:
            spectral_data: Spectral features (scaled)
            metabolite_data: Metabolite features (scaled)
            targets: Target variables (encoded)
            indices: Indices to use (for train/val/test splitting)
        """
        if indices is not None:
            self.spectral_data = torch.FloatTensor(spectral_data[indices])
            self.metabolite_data = torch.FloatTensor(metabolite_data[indices])
            self.targets = torch.LongTensor(targets[indices])
        else:
            self.spectral_data = torch.FloatTensor(spectral_data)
            self.metabolite_data = torch.FloatTensor(metabolite_data)
            self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.spectral_data)
    
    def __getitem__(self, idx):
        return {
            'spectral': self.spectral_data[idx],
            'metabolite': self.metabolite_data[idx],
            'targets': self.targets[idx]
        }


def create_dataloaders(data_dict: Dict[str, Any], config: TransformerConfig) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        data_dict: Dictionary containing processed data
        config: Configuration object with parameters
    
    Returns:
        Dictionary containing train, validation, and test DataLoaders
    """
    # Create datasets
    train_dataset = MultiOmicDataset(
        data_dict['spectral_data'],
        data_dict['metabolite_data'],
        data_dict['target_data'],
        data_dict['train_idx']
    )
    
    val_dataset = MultiOmicDataset(
        data_dict['spectral_data'],
        data_dict['metabolite_data'],
        data_dict['target_data'],
        data_dict['val_idx']
    )
    
    test_dataset = MultiOmicDataset(
        data_dict['spectral_data'],
        data_dict['metabolite_data'],
        data_dict['target_data'],
        data_dict['test_idx']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


# ===== MODEL DEFINITION =====
class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for spectral and metabolite features"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize the cross-attention layer.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Self-attention for spectral
        self.spectral_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for metabolite
        self.metabolite_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: spectral queries, metabolite keys/values
        self.cross_attention_s2m = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: metabolite queries, spectral keys/values
        self.cross_attention_m2s = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward networks
        self.spectral_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.metabolite_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1_spectral = nn.LayerNorm(hidden_dim)
        self.norm2_spectral = nn.LayerNorm(hidden_dim)
        self.norm3_spectral = nn.LayerNorm(hidden_dim)
        
        self.norm1_metabolite = nn.LayerNorm(hidden_dim)
        self.norm2_metabolite = nn.LayerNorm(hidden_dim)
        self.norm3_metabolite = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, spectral_emb, metabolite_emb):
        """
        Forward pass through the cross-attention layer.
        
        Args:
            spectral_emb: Embedded spectral features
            metabolite_emb: Embedded metabolite features
            
        Returns:
            Updated embeddings and attention weights
        """
        # Self-attention for spectral
        spectral_attn, _ = self.spectral_attention(
            spectral_emb, spectral_emb, spectral_emb
        )
        spectral_emb = self.norm1_spectral(spectral_emb + self.dropout(spectral_attn))
        
        # Self-attention for metabolite
        metabolite_attn, _ = self.metabolite_attention(
            metabolite_emb, metabolite_emb, metabolite_emb
        )
        metabolite_emb = self.norm1_metabolite(metabolite_emb + self.dropout(metabolite_attn))
        
        # Cross-attention: spectral -> metabolite
        cross_s2m_output, cross_s2m_weights = self.cross_attention_s2m(
            spectral_emb, metabolite_emb, metabolite_emb
        )
        spectral_emb = self.norm2_spectral(spectral_emb + self.dropout(cross_s2m_output))
        
        # Cross-attention: metabolite -> spectral
        cross_m2s_output, cross_m2s_weights = self.cross_attention_m2s(
            metabolite_emb, spectral_emb, spectral_emb
        )
        metabolite_emb = self.norm2_metabolite(metabolite_emb + self.dropout(cross_m2s_output))
        
        # Feed-forward for spectral
        spectral_ffn_output = self.spectral_ffn(spectral_emb)
        spectral_emb = self.norm3_spectral(spectral_emb + self.dropout(spectral_ffn_output))
        
        # Feed-forward for metabolite
        metabolite_ffn_output = self.metabolite_ffn(metabolite_emb)
        metabolite_emb = self.norm3_metabolite(metabolite_emb + self.dropout(metabolite_ffn_output))
        
        # Return updated embeddings and cross-attention weights
        return spectral_emb, metabolite_emb, {
            's2m': cross_s2m_weights,
            'm2s': cross_m2s_weights
        }


class MultiOmicTransformer(nn.Module):
    """Transformer model for multi-omic integration with cross-attention"""
    
    def __init__(self, 
                 spectral_dim: int,
                 metabolite_dim: int,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 num_tasks: int = 3,
                 num_classes: List[int] = [2, 2, 3]):
        """
        Initialize the transformer model.
        
        Args:
            spectral_dim: Number of spectral features
            metabolite_dim: Number of metabolite features
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            num_tasks: Number of classification tasks
            num_classes: Number of classes for each task
        """
        super().__init__()
        
        self.spectral_dim = spectral_dim
        self.metabolite_dim = metabolite_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        
        # Input embedding layers
        self.spectral_embedding = nn.Linear(spectral_dim, hidden_dim)
        self.metabolite_embedding = nn.Linear(metabolite_dim, hidden_dim)
        
        # Positional encoding (simplified, learnable)
        self.spectral_pos_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.metabolite_pos_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification heads - one per task
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes[i])
            )
            for i in range(num_tasks)
        ])
        
        # Initialize parameters
        self._init_parameters()
        
        # Storage for attention weights
        self.attention_weights = []
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, spectral, metabolite):
        """
        Forward pass through the transformer.
        
        Args:
            spectral: Spectral features
            metabolite: Metabolite features
            
        Returns:
            Dictionary containing task outputs and attention weights
        """
        batch_size = spectral.size(0)
        
        # Reshape inputs for attention: [batch_size, 1, features]
        spectral = spectral.unsqueeze(1)
        metabolite = metabolite.unsqueeze(1)
        
        # Embed inputs
        spectral_emb = self.spectral_embedding(spectral) + self.spectral_pos_encoding
        metabolite_emb = self.metabolite_embedding(metabolite) + self.metabolite_pos_encoding
        
        # Clear stored attention weights
        self.attention_weights = []
        
        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            spectral_emb, metabolite_emb, attn_weights = layer(spectral_emb, metabolite_emb)
            self.attention_weights.append(attn_weights)
        
        # Create combined representation
        # Squeeze out the sequence dimension (which is 1)
        spectral_repr = spectral_emb.squeeze(1)
        metabolite_repr = metabolite_emb.squeeze(1)
        combined = torch.cat([spectral_repr, metabolite_repr], dim=1)
        
        # Apply classification heads
        outputs = [classifier(combined) for classifier in self.classifiers]
        
        return {
            'task_outputs': outputs,
            'attention_weights': self.attention_weights
        }
    
    def get_attention_weights(self):
        """
        Get stored attention weights from the last forward pass.
        
        Returns:
            List of attention weight dictionaries
        """
        return self.attention_weights


# ===== TRAINING FUNCTIONS =====
def compute_loss(outputs, targets, task_weights=None):
    """
    Compute the multi-task loss.
    
    Args:
        outputs: Model outputs (list of logits for each task)
        targets: Target values (batch_size, num_tasks)
        task_weights: Optional weights for each task's loss
        
    Returns:
        Total loss and individual task losses
    """
    # Default equal weights if not provided
    if task_weights is None:
        task_weights = [1.0] * len(outputs)
    
    criterion = nn.CrossEntropyLoss()
    task_losses = []
    
    for i, task_output in enumerate(outputs):
        task_targets = targets[:, i]
        task_loss = criterion(task_output, task_targets)
        task_losses.append(task_loss)
    
    # Weighted sum of task losses
    total_loss = sum(w * loss for w, loss in zip(task_weights, task_losses))
    
    return total_loss, task_losses


def train_epoch(model, dataloader, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The transformer model
        dataloader: Training data loader
        optimizer: Optimizer for parameter updates
        device: Device to train on (cpu/cuda)
        
    Returns:
        Average loss and accuracy metrics
    """
    model.train()
    total_loss = 0.0
    task_losses = [0.0] * 3  # Assuming 3 tasks (Genotype, Treatment, Day)
    correct_predictions = [0] * 3
    total_predictions = 0
    
    for batch in dataloader:
        # Get batch data
        spectral = batch['spectral'].to(device)
        metabolite = batch['metabolite'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(spectral, metabolite)
        task_outputs = outputs['task_outputs']
        
        # Compute loss
        loss, individual_losses = compute_loss(task_outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * spectral.size(0)
        for i, task_loss in enumerate(individual_losses):
            task_losses[i] += task_loss.item() * spectral.size(0)
        
        # Calculate accuracy
        for i, task_output in enumerate(task_outputs):
            _, predicted = torch.max(task_output, 1)
            correct_predictions[i] += (predicted == targets[:, i]).sum().item()
        
        total_predictions += spectral.size(0)
    
    # Calculate averages
    avg_loss = total_loss / total_predictions
    avg_task_losses = [loss / total_predictions for loss in task_losses]
    task_accuracies = [correct / total_predictions for correct in correct_predictions]
    
    return avg_loss, avg_task_losses, task_accuracies


def evaluate(model, dataloader, device):
    """
    Evaluate the model.
    
    Args:
        model: The transformer model
        dataloader: Validation/test data loader
        device: Device to evaluate on (cpu/cuda)
        
    Returns:
        Average loss, accuracy metrics, and predictions
    """
    model.eval()
    total_loss = 0.0
    task_losses = [0.0] * 3  # Assuming 3 tasks
    correct_predictions = [0] * 3
    total_predictions = 0
    
    all_predictions = [[] for _ in range(3)]
    all_targets = [[] for _ in range(3)]
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            spectral = batch['spectral'].to(device)
            metabolite = batch['metabolite'].to(device)
            targets = batch['targets'].to(device)
            
            # Forward pass
            outputs = model(spectral, metabolite)
            task_outputs = outputs['task_outputs']
            
            # Compute loss
            loss, individual_losses = compute_loss(task_outputs, targets)
            
            # Update metrics
            total_loss += loss.item() * spectral.size(0)
            for i, task_loss in enumerate(individual_losses):
                task_losses[i] += task_loss.item() * spectral.size(0)
            
            # Calculate accuracy and store predictions
            for i, task_output in enumerate(task_outputs):
                _, predicted = torch.max(task_output, 1)
                correct_predictions[i] += (predicted == targets[:, i]).sum().item()
                
                # Store predictions and targets for metrics
                all_predictions[i].extend(predicted.cpu().numpy())
                all_targets[i].extend(targets[:, i].cpu().numpy())
            
            total_predictions += spectral.size(0)
    
    # Calculate averages
    avg_loss = total_loss / total_predictions
    avg_task_losses = [loss / total_predictions for loss in task_losses]
    task_accuracies = [correct / total_predictions for correct in correct_predictions]
    
    return avg_loss, avg_task_losses, task_accuracies, all_predictions, all_targets


def train_model(model, dataloaders, config, device):
    """
    Train the transformer model with early stopping.
    
    Args:
        model: The transformer model
        dataloaders: Dictionary of train/val/test dataloaders
        config: Configuration object
        device: Device to train on (cpu/cuda)
        
    Returns:
        Trained model and training history
    """
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [[] for _ in range(3)],
        'val_acc': [[] for _ in range(3)]
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss, train_task_losses, train_accuracies = train_epoch(
            model, dataloaders['train'], optimizer, device
        )
        
        # Evaluate on validation set
        val_loss, val_task_losses, val_accuracies, _, _ = evaluate(
            model, dataloaders['val'], device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        for i in range(3):
            history['train_acc'][i].append(train_accuracies[i])
            history['val_acc'][i].append(val_accuracies[i])
        
        # Log progress
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1}/{config.num_epochs} - {epoch_time:.2f}s - "
                    f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        for i, task in enumerate(['Genotype', 'Treatment', 'Day']):
            logging.info(f"  {task} - Train Acc: {train_accuracies[i]:.4f} - "
                        f"Val Acc: {val_accuracies[i]:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                config.checkpoints_dir, 
                f"{config.tissue}_transformer_best.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'config': vars(config)
            }, checkpoint_path)
            logging.info(f"Saved best model checkpoint (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Loaded best model weights")
    
    return model, history


# ===== ANALYSIS FUNCTIONS =====
def extract_attention_weights(model, dataloader, data_dict, config, device):
    """
    Extract attention weights from the model for analysis.
    
    Args:
        model: Trained transformer model
        dataloader: Test data loader
        data_dict: Dictionary containing feature names and metadata
        config: Configuration object
        device: Device to run on (cpu/cuda)
        
    Returns:
        Dictionary containing processed attention data
    """
    model.eval()
    
    # Get feature names
    spectral_features = data_dict['spectral_feature_names']
    metabolite_features = data_dict['metabolite_feature_names']
    
    # Storage for attention scores
    all_s2m_attn = []  # spectral -> metabolite
    all_m2s_attn = []  # metabolite -> spectral
    
    # Sample metadata
    sample_idx = []
    metadata = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get batch data
            spectral = batch['spectral'].to(device)
            metabolite = batch['metabolite'].to(device)
            targets = batch['targets'].cpu().numpy()
            
            # Forward pass
            outputs = model(spectral, metabolite)
            attention_weights = model.get_attention_weights()
            
            # We'll focus on the last layer's attention weights (most refined)
            last_layer_weights = attention_weights[-1]
            
            # Extract weights
            s2m_weights = last_layer_weights['s2m'].cpu().numpy()
            m2s_weights = last_layer_weights['m2s'].cpu().numpy()
            
            # Store weights
            all_s2m_attn.append(s2m_weights)
            all_m2s_attn.append(m2s_weights)
            
            # Store sample indices for this batch
            batch_indices = data_dict['test_idx'][batch_idx * config.batch_size:
                                               min((batch_idx + 1) * config.batch_size,
                                                   len(data_dict['test_idx']))]
            sample_idx.extend(batch_indices)
            
            # Store metadata for this batch
            for i in range(len(targets)):
                meta_idx = batch_indices[i] if i < len(batch_indices) else None
                if meta_idx is not None:
                    metadata.append({
                        'Genotype': data_dict['metadata']['Genotype'].iloc[meta_idx],
                        'Treatment': data_dict['metadata']['Treatment'].iloc[meta_idx],
                        'Day': data_dict['metadata']['Day'].iloc[meta_idx],
                        'Batch': data_dict['metadata']['Batch'].iloc[meta_idx]
                    })
    
    # Concatenate all attention weights
    all_s2m_attn = np.concatenate(all_s2m_attn, axis=0)
    all_m2s_attn = np.concatenate(all_m2s_attn, axis=0)
    
    # Average across attention heads
    s2m_avg = np.mean(all_s2m_attn, axis=1)  # shape: [n_samples, n_spectral, n_metabolite]
    m2s_avg = np.mean(all_m2s_attn, axis=1)  # shape: [n_samples, n_metabolite, n_spectral]
    
    # Average across samples (for global importance)
    s2m_global = np.mean(s2m_avg, axis=0)  # shape: [n_spectral, n_metabolite]
    m2s_global = np.mean(m2s_avg, axis=0)  # shape: [n_metabolite, n_spectral]
    
    # Package attention data
    attention_data = {
        'spectral_features': spectral_features,
        'metabolite_features': metabolite_features,
        's2m_attention': s2m_avg,
        'm2s_attention': m2s_avg,
        's2m_global': s2m_global,
        'm2s_global': m2s_global,
        'sample_metadata': metadata,
        'sample_indices': sample_idx
    }
    
    return attention_data


def analyze_cross_modal_pairs(attention_data, config):
    """
    Analyze cross-modal pairs from attention weights.
    
    Args:
        attention_data: Dictionary containing attention data
        config: Configuration object
        
    Returns:
        DataFrame containing top spectral-metabolite pairs
    """
    # Extract global attention scores
    s2m_global = attention_data['s2m_global']
    
    # Get feature names
    spectral_features = attention_data['spectral_features']
    metabolite_features = attention_data['metabolite_features']
    
    # Create DataFrame for all pairs
    pairs_data = []
    
    for i, spectral_feat in enumerate(spectral_features):
        for j, metabolite_feat in enumerate(metabolite_features):
            attention_score = s2m_global[i, j]
            pairs_data.append({
                'Spectral_Feature': spectral_feat,
                'Metabolite_Feature': metabolite_feat,
                'Attention_Score': attention_score,
                'Spectral_Index': i,
                'Metabolite_Index': j
            })
    
    # Create DataFrame
    pairs_df = pd.DataFrame(pairs_data)
    
    # Sort by attention score
    pairs_df = pairs_df.sort_values('Attention_Score', ascending=False)
    
    # Filter top pairs
    top_pairs = pairs_df.head(config.top_n_pairs)
    
    # Save to CSV
    output_path = os.path.join(
        config.output_dir,
        f"transformer_cross_modal_pairs_{config.tissue}.csv"
    )
    top_pairs.to_csv(output_path, index=False)
    
    # Also save all pairs for potential later analysis
    all_pairs_path = os.path.join(
        config.output_dir,
        f"transformer_all_cross_modal_pairs_{config.tissue}.csv"
    )
    pairs_df.to_csv(all_pairs_path, index=False)
    
    logging.info(f"Saved top {config.top_n_pairs} cross-modal pairs to {output_path}")
    logging.info(f"Saved all cross-modal pairs to {all_pairs_path}")
    
    return top_pairs


def analyze_temporal_attention(attention_data, config):
    """
    Analyze temporal patterns in attention weights.
    
    Args:
        attention_data: Dictionary containing attention data
        config: Configuration object
        
    Returns:
        DataFrame containing attention patterns by day
    """
    # Extract attention scores and metadata
    s2m_attention = attention_data['s2m_attention']
    sample_metadata = attention_data['sample_metadata']
    
    # Get feature names
    spectral_features = attention_data['spectral_features']
    metabolite_features = attention_data['metabolite_features']
    
    # Group samples by day
    day_groups = {}
    for i, meta in enumerate(sample_metadata):
        day = meta['Day']
        if day not in day_groups:
            day_groups[day] = []
        day_groups[day].append(i)
    
    # Calculate average attention per day
    day_attention = {}
    for day, indices in day_groups.items():
        day_samples = s2m_attention[indices]
        day_avg = np.mean(day_samples, axis=0)
        day_attention[day] = day_avg
    
    # Create DataFrame for temporal analysis
    temporal_data = []
    
    # For each day, get top pairs
    for day in sorted(day_groups.keys()):
        day_avg = day_attention[day]
        
        # Find top pairs for this day
        for i, spectral_feat in enumerate(spectral_features):
            for j, metabolite_feat in enumerate(metabolite_features):
                attention_score = day_avg[i, j]
                
                # Only keep scores above threshold
                if attention_score > config.attention_threshold:
                    temporal_data.append({
                        'Day': day,
                        'Spectral_Feature': spectral_feat,
                        'Metabolite_Feature': metabolite_feat,
                        'Attention_Score': attention_score,
                        'Spectral_Index': i,
                        'Metabolite_Index': j
                    })
    
    # Create DataFrame
    temporal_df = pd.DataFrame(temporal_data)
    
    # Sort by day and attention score
    temporal_df = temporal_df.sort_values(['Day', 'Attention_Score'], ascending=[True, False])
    
    # Save to CSV
    output_path = os.path.join(
        config.output_dir,
        f"transformer_temporal_attention_{config.tissue}.csv"
    )
    temporal_df.to_csv(output_path, index=False)
    logging.info(f"Saved temporal attention patterns to {output_path}")
    
    return temporal_df


def analyze_feature_importance(attention_data, config):
    """
    Calculate feature importance based on attention weights.
    
    Args:
        attention_data: Dictionary containing attention data
        config: Configuration object
        
    Returns:
        DataFrame containing feature importance scores
    """
    # Extract global attention scores
    s2m_global = attention_data['s2m_global']
    m2s_global = attention_data['m2s_global']
    
    # Get feature names
    spectral_features = attention_data['spectral_features']
    metabolite_features = attention_data['metabolite_features']
    
    # Calculate spectral feature importance
    # For each spectral feature, sum its attention to all metabolite features
    spectral_importance = np.sum(s2m_global, axis=1)
    
    # Calculate metabolite feature importance
    # For each metabolite feature, sum its attention from all spectral features
    metabolite_importance = np.sum(s2m_global, axis=0)
    
    # Create DataFrames
    spectral_df = pd.DataFrame({
        'Feature': spectral_features,
        'Importance': spectral_importance,
        'Feature_Type': 'Spectral'
    })
    
    metabolite_df = pd.DataFrame({
        'Feature': metabolite_features,
        'Importance': metabolite_importance,
        'Feature_Type': 'Metabolite'
    })
    
    # Combine and normalize within feature types
    combined_df = pd.concat([spectral_df, metabolite_df])
    
    # Normalize within feature types
    for feat_type in ['Spectral', 'Metabolite']:
        mask = combined_df['Feature_Type'] == feat_type
        min_val = combined_df.loc[mask, 'Importance'].min()
        max_val = combined_df.loc[mask, 'Importance'].max()
        combined_df.loc[mask, 'Importance_Scaled'] = (
            (combined_df.loc[mask, 'Importance'] - min_val) / 
            (max_val - min_val) if max_val > min_val else 0.0
        )
    
    # Sort by importance
    combined_df = combined_df.sort_values(['Feature_Type', 'Importance'], ascending=[True, False])
    
    # Save to CSV
    output_path = os.path.join(
        config.output_dir,
        f"transformer_feature_importance_{config.tissue}.csv"
    )
    combined_df.to_csv(output_path, index=False)
    logging.info(f"Saved feature importance scores to {output_path}")
    
    return combined_df


def analyze_genotype_differences(attention_data, config):
    """
    Analyze genotype-specific attention patterns.
    
    Args:
        attention_data: Dictionary containing attention data
        config: Configuration object
        
    Returns:
        Dictionary containing genotype-specific attention data
    """
    # Extract attention scores and metadata
    s2m_attention = attention_data['s2m_attention']
    sample_metadata = attention_data['sample_metadata']
    
    # Get feature names
    spectral_features = attention_data['spectral_features']
    metabolite_features = attention_data['metabolite_features']
    
    # Group samples by genotype
    g1_indices = [i for i, meta in enumerate(sample_metadata) if meta['Genotype'] == 'G1']
    g2_indices = [i for i, meta in enumerate(sample_metadata) if meta['Genotype'] == 'G2']
    
    # Calculate average attention per genotype
    g1_avg = np.mean(s2m_attention[g1_indices], axis=0) if g1_indices else None
    g2_avg = np.mean(s2m_attention[g2_indices], axis=0) if g2_indices else None
    
    # Create DataFrame for genotype differences
    genotype_data = []
    
    if g1_avg is not None and g2_avg is not None:
        # Calculate absolute difference
        diff = np.abs(g1_avg - g2_avg)
        
        # Find pairs with significant differences
        for i, spectral_feat in enumerate(spectral_features):
            for j, metabolite_feat in enumerate(metabolite_features):
                g1_score = g1_avg[i, j]
                g2_score = g2_avg[i, j]
                difference = diff[i, j]
                
                # Only keep significant differences
                if difference > config.attention_threshold:
                    genotype_data.append({
                        'Spectral_Feature': spectral_feat,
                        'Metabolite_Feature': metabolite_feat,
                        'G1_Attention': g1_score,
                        'G2_Attention': g2_score,
                        'Abs_Difference': difference,
                        'Stronger_In': 'G1' if g1_score > g2_score else 'G2',
                        'Spectral_Index': i,
                        'Metabolite_Index': j
                    })
    
    # Create DataFrame
    genotype_df = pd.DataFrame(genotype_data)
    
    # Sort by difference
    if not genotype_df.empty:
        genotype_df = genotype_df.sort_values('Abs_Difference', ascending=False)
        
        # Save to CSV
        output_path = os.path.join(
            config.output_dir,
            f"transformer_genotype_diff_{config.tissue}.csv"
        )
        genotype_df.to_csv(output_path, index=False)
        logging.info(f"Saved genotype difference analysis to {output_path}")
    else:
        logging.warning("No significant genotype differences found in attention patterns")
    
    # Return genotype-specific attention for visualizations
    return {
        'g1_avg': g1_avg,
        'g2_avg': g2_avg,
        'genotype_df': genotype_df
    }


def evaluate_model_performance(model, dataloaders, config, device):
    """
    Evaluate model performance with detailed metrics.
    
    Args:
        model: Trained transformer model
        dataloaders: Dictionary of train/val/test dataloaders
        config: Configuration object
        device: Device to evaluate on (cpu/cuda)
        
    Returns:
        DataFrame containing performance metrics
    """
    # Evaluate on test set
    test_loss, test_task_losses, test_accuracies, predictions, targets = evaluate(
        model, dataloaders['test'], device
    )
    
    # Calculate detailed metrics for each task
    performance_data = []
    task_names = ['Genotype', 'Treatment', 'Day']
    
    for i, task in enumerate(task_names):
        y_true = targets[i]
        y_pred = predictions[i]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle binary vs. multi-class
        if len(np.unique(y_true)) == 2:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        else:
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Store metrics
        performance_data.append({
            'Task': task,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Loss': test_task_losses[i]
        })
    
    # Create DataFrame
    performance_df = pd.DataFrame(performance_data)
    
    # Add average row
    avg_row = {
        'Task': 'Average',
        'Accuracy': np.mean(performance_df['Accuracy']),
        'Precision': np.mean(performance_df['Precision']),
        'Recall': np.mean(performance_df['Recall']),
        'F1_Score': np.mean(performance_df['F1_Score']),
        'Loss': np.mean(performance_df['Loss'])
    }
    performance_df = pd.concat([performance_df, pd.DataFrame([avg_row])])
    
    # Save to CSV
    output_path = os.path.join(
        config.output_dir,
        f"transformer_class_performance_{config.tissue}.csv"
    )
    performance_df.to_csv(output_path, index=False)
    logging.info(f"Saved model performance metrics to {output_path}")
    
    # Generate and save confusion matrices
    for i, task in enumerate(task_names):
        cm = confusion_matrix(targets[i], predictions[i])
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {task}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save figure
        cm_path = os.path.join(
            config.figure_dir,
            f"confusion_matrix_{task.lower()}_{config.tissue}.png"
        )
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return performance_df


def run_baseline_models(data_dict, config):
    """
    Run baseline models for comparison.
    
    Args:
        data_dict: Dictionary containing processed data
        config: Configuration object
        
    Returns:
        DataFrame comparing baseline models with transformer
    """
    # Extract data
    spectral_data = data_dict['spectral_data']
    metabolite_data = data_dict['metabolite_data']
    target_data = data_dict['target_data']
    
    # Combine features
    X = np.concatenate([spectral_data, metabolite_data], axis=1)
    
    # Extract indices
    train_idx = data_dict['train_idx']
    val_idx = data_dict['val_idx']
    test_idx = data_dict['test_idx']
    
    # Combine train and validation for final evaluation
    train_val_idx = np.concatenate([train_idx, val_idx])
    
    # Task names
    task_names = ['Genotype', 'Treatment', 'Day']
    
    # Initialize results
    baseline_results = []
    
    # Models to evaluate
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=config.seed, n_jobs=-1
        ),
        'SVM': SVC(probability=True, random_state=config.seed),
        'LogisticRegression': LogisticRegression(
            max_iter=1000, random_state=config.seed, n_jobs=-1
        ),
    }
    
    # Train and evaluate models for each task
    for task_idx, task in enumerate(task_names):
        y = target_data[:, task_idx]
        
        for model_name, model in models.items():
            # Train on combined train+val
            model.fit(X[train_val_idx], y[train_val_idx])
            
            # Evaluate on test
            y_pred = model.predict(X[test_idx])
            
            # Calculate metrics
            accuracy = accuracy_score(y[test_idx], y_pred)
            
            if len(np.unique(y)) == 2:
                precision = precision_score(y[test_idx], y_pred)
                recall = recall_score(y[test_idx], y_pred)
                f1 = f1_score(y[test_idx], y_pred)
            else:
                precision = precision_score(y[test_idx], y_pred, average='weighted')
                recall = recall_score(y[test_idx], y_pred, average='weighted')
                f1 = f1_score(y[test_idx], y_pred, average='weighted')
            
            # Store results
            baseline_results.append({
                'Model': model_name,
                'Task': task,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })
    
    # Create DataFrame
    baseline_df = pd.DataFrame(baseline_results)
    
    # Save to CSV
    output_path = os.path.join(
        config.output_dir,
        f"transformer_baseline_comparison_{config.tissue}.csv"
    )
    baseline_df.to_csv(output_path, index=False)
    logging.info(f"Saved baseline model comparison to {output_path}")
    
    return baseline_df


# ===== VISUALIZATION FUNCTIONS =====
def visualize_cross_modal_network(cross_modal_pairs, attention_data, config):
    """
    Visualize cross-modal network of spectral-metabolite connections.
    
    Args:
        cross_modal_pairs: DataFrame of top spectral-metabolite pairs
        attention_data: Dictionary containing attention data
        config: Configuration object
    """
    # Extract top pairs
    top_n = min(50, len(cross_modal_pairs))  # Limit to 50 for readability
    top_pairs = cross_modal_pairs.head(top_n)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    spectral_nodes = set(top_pairs['Spectral_Feature'])
    metabolite_nodes = set(top_pairs['Metabolite_Feature'])
    
    for node in spectral_nodes:
        G.add_node(node, type='spectral')
    
    for node in metabolite_nodes:
        G.add_node(node, type='metabolite')
    
    # Add edges with weights
    for _, row in top_pairs.iterrows():
        G.add_edge(
            row['Spectral_Feature'],
            row['Metabolite_Feature'],
            weight=row['Attention_Score']
        )
    
    # Create positions (spectral on left, metabolite on right)
    pos = {}
    
    # Position spectral nodes
    for i, node in enumerate(spectral_nodes):
        pos[node] = (-2, (i - len(spectral_nodes)/2) * 1.0)
    
    # Position metabolite nodes
    for i, node in enumerate(metabolite_nodes):
        pos[node] = (2, (i - len(metabolite_nodes)/2) * 1.0)
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[n for n in G.nodes if G.nodes[n]['type'] == 'spectral'],
        node_color='skyblue',
        node_size=300,
        label='Spectral'
    )
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[n for n in G.nodes if G.nodes[n]['type'] == 'metabolite'],
        node_color='lightgreen',
        node_size=300,
        label='Metabolite'
    )
    
    # Draw edges with width based on attention score
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges]
    nx.draw_networkx_edges(
        G, pos,
        width=edge_weights,
        alpha=0.7,
        edge_color='gray'
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_family='sans-serif'
    )
    
    plt.title(f"Top {top_n} Cross-Modal Connections ({config.tissue.capitalize()} Tissue)")
    plt.legend()
    plt.axis('off')
    
    # Save figure
    output_path = os.path.join(
        config.figure_dir,
        f"network_cross_modal_{config.tissue}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved cross-modal network visualization to {output_path}")


def visualize_attention_heatmap(attention_data, cross_modal_pairs, config):
    """
    Visualize attention heatmap of strongest spectral-metabolite connections.
    
    Args:
        attention_data: Dictionary containing attention data
        cross_modal_pairs: DataFrame of top spectral-metabolite pairs
        config: Configuration object
    """
    # Extract global attention matrix
    s2m_global = attention_data['s2m_global']
    
    # Extract top features
    top_n = min(25, len(cross_modal_pairs))  # Limit for readability
    top_pairs = cross_modal_pairs.head(top_n)
    
    # Get unique features
    unique_spectral = top_pairs['Spectral_Feature'].unique()
    unique_metabolite = top_pairs['Metabolite_Feature'].unique()
    
    # Get indices
    spectral_indices = [attention_data['spectral_features'].index(feat) for feat in unique_spectral]
    metabolite_indices = [attention_data['metabolite_features'].index(feat) for feat in unique_metabolite]
    
    # Extract submatrix
    submatrix = s2m_global[np.ix_(spectral_indices, metabolite_indices)]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        submatrix,
        xticklabels=unique_metabolite,
        yticklabels=unique_spectral,
        cmap='Blues',
        annot=True,
        fmt='.2f',
        linewidths=0.5
    )
    
    plt.title(f"Attention Heatmap ({config.tissue.capitalize()} Tissue)")
    plt.xlabel('Metabolite Features')
    plt.ylabel('Spectral Features')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Save figure
    output_path = os.path.join(
        config.figure_dir,
        f"heatmap_attention_{config.tissue}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved attention heatmap to {output_path}")


def visualize_temporal_evolution(temporal_df, config):
    """
    Visualize temporal evolution of attention patterns across days.
    
    Args:
        temporal_df: DataFrame containing temporal attention patterns
        config: Configuration object
    """
    if temporal_df.empty:
        logging.warning("Cannot create temporal visualization (empty data)")
        return
    
    # Get unique days
    days = sorted(temporal_df['Day'].unique())
    
    # Create subplots for each day
    fig, axes = plt.subplots(1, len(days), figsize=(16, 6), sharey=True)
    
    # If only one day, wrap axes in list
    if len(days) == 1:
        axes = [axes]
    
    for i, day in enumerate(days):
        # Get data for this day
        day_data = temporal_df[temporal_df['Day'] == day]
        
        # Get top pairs for visualization
        top_n = min(20, len(day_data))
        top_pairs = day_data.head(top_n)
        
        # Create bar plot
        sns.barplot(
            x='Attention_Score',
            y='Metabolite_Feature',
            hue='Spectral_Feature',
            data=top_pairs,
            ax=axes[i],
            palette='viridis'
        )
        
        axes[i].set_title(f"Day {day}")
        
        # Only show legend for the last subplot
        if i < len(days) - 1:
            axes[i].legend([])
        else:
            axes[i].legend(title='Spectral Feature', loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.suptitle(f"Temporal Evolution of Attention Patterns ({config.tissue.capitalize()} Tissue)", y=1.05)
    
    # Save figure
    output_path = os.path.join(
        config.figure_dir,
        f"temporal_attention_evolution_{config.tissue}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved temporal evolution visualization to {output_path}")


def visualize_genotype_differences(genotype_diff, attention_data, config):
    """
    Visualize genotype-specific attention patterns.
    
    Args:
        genotype_diff: Dictionary containing genotype attention data
        attention_data: Dictionary containing attention data
        config: Configuration object
    """
    genotype_df = genotype_diff['genotype_df']
    
    if genotype_df is None or genotype_df.empty:
        logging.warning("Cannot create genotype visualization (empty data)")
        return
    
    # Extract top differential pairs
    top_n = min(20, len(genotype_df))
    top_diff = genotype_df.head(top_n)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create a paired bar plot
    x = np.arange(len(top_diff))
    width = 0.35
    
    plt.bar(x - width/2, top_diff['G1_Attention'], width, label='G1 (Tolerant)')
    plt.bar(x + width/2, top_diff['G2_Attention'], width, label='G2 (Susceptible)')
    
    # Add labels and title
    plt.xlabel('Spectral-Metabolite Pair')
    plt.ylabel('Attention Score')
    plt.title(f"Genotype-Specific Attention Patterns ({config.tissue.capitalize()} Tissue)")
    
    # Create pair labels
    pair_labels = [f"{s[:10]}...{m[:10]}..." for s, m in 
                   zip(top_diff['Spectral_Feature'], top_diff['Metabolite_Feature'])]
    
    plt.xticks(x, pair_labels, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(
        config.figure_dir,
        f"genotype_differential_network_{config.tissue}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved genotype difference visualization to {output_path}")


def visualize_spectral_regions(feature_importance, attention_data, config):
    """
    Visualize important spectral wavelength regions.
    
    Args:
        feature_importance: DataFrame containing feature importance scores
        attention_data: Dictionary containing attention data
        config: Configuration object
    """
    # Extract spectral feature importance
    spectral_df = feature_importance[feature_importance['Feature_Type'] == 'Spectral'].copy()
    
    if spectral_df.empty:
        logging.warning("Cannot create spectral regions visualization (empty data)")
        return
    
    # Extract wavelength numbers and sort
    spectral_df['Wavelength'] = spectral_df['Feature'].apply(
        lambda x: int(x.split('_')[0][1:]) if x.startswith('W_') and x.split('_')[0][1:].isdigit() else 0
    )
    spectral_df = spectral_df.sort_values('Wavelength')
    
    # Filter for non-zero wavelengths
    spectral_df = spectral_df[spectral_df['Wavelength'] > 0]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot wavelength importance
    plt.scatter(
        spectral_df['Wavelength'],
        spectral_df['Importance_Scaled'],
        c=spectral_df['Importance_Scaled'],
        cmap='viridis',
        s=50,
        alpha=0.7
    )
    
    # Add smoothed trend line
    if len(spectral_df) > 5:  # Only if we have enough points
        from scipy.signal import savgol_filter
        x = spectral_df['Wavelength'].values
        y = spectral_df['Importance_Scaled'].values
        
        # Sort by x for smoothing
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Apply Savitzky-Golay filter with appropriate window length
        window_length = min(15, len(x) - (len(x) % 2) - 1)  # Must be odd and less than len(x)
        if window_length > 3:  # Minimum required window length
            y_smooth = savgol_filter(y_sorted, window_length, 3)
            plt.plot(x_sorted, y_smooth, 'r-', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Importance Score (Scaled)')
    plt.title(f"Key Spectral Regions ({config.tissue.capitalize()} Tissue)")
    
    # Add colorbar
    plt.colorbar(label='Importance Score')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(
        config.figure_dir,
        f"key_spectral_regions_{config.tissue}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved spectral regions visualization to {output_path}")


# ===== MAIN EXECUTION =====
def setup_logging(config):
    """Set up logging configuration"""
    log_file = os.path.join(
        config.logs_dir,
        f"transformer_{config.tissue}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Multi-Omic Transformer Implementation')
    parser.add_argument('--tissue', type=str, choices=['leaf', 'root'], default='leaf',
                        help='Tissue type to analyze (leaf or root)')
    args = parser.parse_args()
    
    # Initialize configuration
    config = TransformerConfig(tissue=args.tissue)
    
    # Setup logging
    setup_logging(config)
    
    # Record start time
    start_time = time.time()
    logging.info(f"Starting Multi-Omic Transformer Analysis for {config.tissue.capitalize()} Tissue")
    
    # Set random seeds for reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    try:
        # Load and preprocess data
        data_dict = load_and_preprocess_data(config)
        
        # Create dataloaders
        dataloaders = create_dataloaders(data_dict, config)
        
        # Initialize model
        spectral_dim = data_dict['spectral_data'].shape[1]
        metabolite_dim = data_dict['metabolite_data'].shape[1]
        
        model = MultiOmicTransformer(
            spectral_dim=spectral_dim,
            metabolite_dim=metabolite_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            num_tasks=3,
            num_classes=[2, 2, 3]  # Genotype, Treatment, Day
        ).to(device)
        
        logging.info(f"Initialized model with {spectral_dim} spectral features and {metabolite_dim} metabolite features")
        
        # Train model
        trained_model, history = train_model(model, dataloaders, config, device)
        
        # Evaluate model performance
        performance_df = evaluate_model_performance(trained_model, dataloaders, config, device)
        
        # Run baseline models
        baseline_df = run_baseline_models(data_dict, config)
        
        # Extract and analyze attention weights
        attention_data = extract_attention_weights(trained_model, dataloaders['test'], data_dict, config, device)
        
        # Analyze cross-modal pairs
        cross_modal_pairs = analyze_cross_modal_pairs(attention_data, config)
        
        # Analyze temporal attention
        temporal_df = analyze_temporal_attention(attention_data, config)
        
        # Calculate feature importance
        feature_importance = analyze_feature_importance(attention_data, config)
        
        # Analyze genotype differences
        genotype_diff = analyze_genotype_differences(attention_data, config)
        
        # Create visualizations
        visualize_cross_modal_network(cross_modal_pairs, attention_data, config)
        visualize_attention_heatmap(attention_data, cross_modal_pairs, config)
        visualize_temporal_evolution(temporal_df, config)
        visualize_genotype_differences(genotype_diff, attention_data, config)
        visualize_spectral_regions(feature_importance, attention_data, config)
        
        # Log completion
        end_time = time.time()
        runtime = end_time - start_time
        logging.info(f"Analysis completed in {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise
    
    return 0


if __name__ == "__main__":
    main()