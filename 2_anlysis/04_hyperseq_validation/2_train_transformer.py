# -*- coding: utf-8 -*-
"""
train_transformer_attn.py

Validation Script: Train Transformer on HyperSeq Data

This script trains a transformer model on HyperSeq data and performs permutation testing
to validate the attention mechanism.
"""

# ===== IMPORTS =====
import os
import sys
import time
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
import h5py
from tqdm import tqdm

# ===== CONFIGURATION =====
print("="*80)
print("Transformer Training for HyperSeq Validation - START")
print("="*80)

# --- Paths & Config ---
BASE_DIR = r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val/val"
MOFA_RESULTS_DIR = os.path.join(BASE_DIR, "mofa_results")
OUTPUT_DIR = os.path.join(BASE_DIR, "transformer_results")
RESULTS_SUBDIR = os.path.join(OUTPUT_DIR, "results")
CHECKPOINT_SUBDIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_SUBDIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_SUBDIR, exist_ok=True)
os.makedirs(CHECKPOINT_SUBDIR, exist_ok=True)
os.makedirs(LOG_SUBDIR, exist_ok=True)

ANALYSIS_PAIRING = "HyperSeq"
INPUT_FILES = {
    "HyperSeq": {
        "spectral": os.path.join(MOFA_RESULTS_DIR, "transformer_input_spectral_hyperseq.csv"),
        "metabolite": os.path.join(MOFA_RESULTS_DIR, "transformer_input_transcriptomics_hyperseq.csv")
    }
}

METADATA_COLS_BASE = ['original_cell_id', 'Batch', 'Grid', 'Cell Quantity', 'Cell_number_in_grid']
TARGET_COLS = ['Batch', 'Grid']

# Hyperparameters
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-5
VAL_SIZE = 0.20
TEST_SIZE = 0.20
NUM_WORKERS = 0
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERMUTATION_RUNS = 5000

# ===== LOGGING =====
def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"transformer_hyperseq_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s', '%H:%M:%S')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(log_filepath)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

logger = setup_logging(LOG_SUBDIR)
logger.info(f"Analysis Pairing: {ANALYSIS_PAIRING}, Device: {DEVICE}")

# ===== DATA & MODEL CLASSES =====
class PlantOmicsDataset(Dataset):
    def __init__(self, spectral_features, metabolite_features, targets):
        self.spectral_data = torch.tensor(spectral_features.values, dtype=torch.float32)
        self.metabolite_data = torch.tensor(metabolite_features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.long)

    def __len__(self):
        return len(self.spectral_data)

    def __getitem__(self, idx):
        return {
            'spectral': self.spectral_data[idx],
            'metabolite': self.metabolite_data[idx],
            'targets': self.targets[idx]
        }

def load_and_preprocess_data():
    logger.info("--- Starting Data Loading & Preprocessing ---")
    
    spectral_path = INPUT_FILES[ANALYSIS_PAIRING]['spectral']
    metabolite_path = INPUT_FILES[ANALYSIS_PAIRING]['metabolite']
    
    df_spectral_raw = pd.read_csv(spectral_path, na_values='NA')
    df_metabolite_raw = pd.read_csv(metabolite_path, na_values='NA')
    
    all_potential_meta_cols = ['Row_names'] + METADATA_COLS_BASE
    spectral_feature_names = [col for col in df_spectral_raw.columns if col not in all_potential_meta_cols]
    metabolite_feature_names = [col for col in df_metabolite_raw.columns if col not in all_potential_meta_cols]
    
    df_merged = pd.merge(df_spectral_raw, df_metabolite_raw, on='Row_names', how='inner', suffixes=('_spec', '_metab'))
    df_merged.set_index('Row_names', inplace=True)
    
    meta_cols_suffixed = [f"{col}_spec" for col in METADATA_COLS_BASE if f"{col}_spec" in df_merged.columns]
    metadata_df = df_merged[meta_cols_suffixed].copy()
    metadata_df.columns = [c.replace('_spec', '') for c in meta_cols_suffixed]
    
    features_spectral_df = df_merged[spectral_feature_names]
    features_metabolite_df = df_merged[metabolite_feature_names]
    
    targets_encoded = pd.DataFrame(index=metadata_df.index)
    label_encoders = {}
    num_classes = {}
    
    for col in TARGET_COLS:
        le = LabelEncoder()
        targets_encoded[col] = le.fit_transform(metadata_df[col])
        label_encoders[col] = le
        num_classes[col] = len(le.classes_)
    
    stratify_key = metadata_df['Batch']
    indices = metadata_df.index
    
    train_idx, temp_idx = train_test_split(indices, test_size=(VAL_SIZE + TEST_SIZE), random_state=RANDOM_SEED, stratify=stratify_key)
    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_idx, test_idx = train_test_split(temp_idx, test_size=relative_test_size, random_state=RANDOM_SEED, stratify=stratify_key.loc[temp_idx])
    
    X_train_spec = features_spectral_df.loc[train_idx]
    X_val_spec = features_spectral_df.loc[val_idx]
    X_test_spec = features_spectral_df.loc[test_idx]
    
    X_train_metab = features_metabolite_df.loc[train_idx]
    X_val_metab = features_metabolite_df.loc[val_idx]
    X_test_metab = features_metabolite_df.loc[test_idx]
    
    y_train = targets_encoded.loc[train_idx]
    y_val = targets_encoded.loc[val_idx]
    y_test = targets_encoded.loc[test_idx]
    
    test_metadata_df = metadata_df.loc[test_idx]
    
    scaler_spec = StandardScaler()
    X_train_spec_scaled = scaler_spec.fit_transform(X_train_spec)
    X_val_spec_scaled = scaler_spec.transform(X_val_spec)
    X_test_spec_scaled = scaler_spec.transform(X_test_spec)
    
    scaler_metab = StandardScaler()
    X_train_metab_scaled = scaler_metab.fit_transform(X_train_metab)
    X_val_metab_scaled = scaler_metab.transform(X_val_metab)
    X_test_metab_scaled = scaler_metab.transform(X_test_metab)
    
    X_train_spec_df = pd.DataFrame(X_train_spec_scaled, index=train_idx)
    X_val_spec_df = pd.DataFrame(X_val_spec_scaled, index=val_idx)
    X_test_spec_df = pd.DataFrame(X_test_spec_scaled, index=test_idx)
    
    X_train_metab_df = pd.DataFrame(X_train_metab_scaled, index=train_idx)
    X_val_metab_df = pd.DataFrame(X_val_metab_scaled, index=val_idx)
    X_test_metab_df = pd.DataFrame(X_test_metab_scaled, index=test_idx)
    
    train_dataset = PlantOmicsDataset(X_train_spec_df, X_train_metab_df, y_train)
    val_dataset = PlantOmicsDataset(X_val_spec_df, X_val_metab_df, y_val)
    test_dataset = PlantOmicsDataset(X_test_spec_df, X_test_metab_df, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, spectral_feature_names, metabolite_feature_names, test_metadata_df, num_classes, test_dataset

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn_1_to_2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_2_to_1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x1, x2):
        attn_output_1, attn_weights_1_to_2 = self.cross_attn_1_to_2(query=x1, key=x2, value=x2, average_attn_weights=False)
        x1 = self.norm1(x1 + self.dropout(attn_output_1))
        
        attn_output_2, attn_weights_2_to_1 = self.cross_attn_2_to_1(query=x2, key=x1, value=x1, average_attn_weights=False)
        x2 = self.norm2(x2 + self.dropout(attn_output_2))
        
        ffn_output1 = self.ffn(x1)
        x1 = self.norm3(x1 + self.dropout(ffn_output1))
        
        ffn_output2 = self.ffn(x2)
        x2 = self.norm3(x2 + self.dropout(ffn_output2))
        
        return x1, x2, attn_weights_1_to_2, attn_weights_2_to_1

class SimplifiedTransformer(nn.Module):
    def __init__(self, spectral_dim, metabolite_dim, hidden_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.spectral_feature_embedding = nn.Linear(1, hidden_dim)
        self.metabolite_feature_embedding = nn.Linear(1, hidden_dim)
        
        self.pos_encoding_spec = nn.Parameter(torch.randn(1, spectral_dim, hidden_dim) * 0.02)
        self.pos_encoding_metab = nn.Parameter(torch.randn(1, metabolite_dim, hidden_dim) * 0.02)
        
        self.embedding_norm_spec = nn.LayerNorm(hidden_dim)
        self.embedding_norm_metab = nn.LayerNorm(hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.output_heads = nn.ModuleDict()
        total_pooled_dim = hidden_dim * 2
        
        for task_name, n_class in num_classes.items():
            self.output_heads[task_name] = nn.Sequential(
                nn.LayerNorm(total_pooled_dim),
                nn.Linear(total_pooled_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_class)
            )
            
        self.attention_weights = {}

    def forward(self, spectral, metabolite):
        spec_emb = self.spectral_feature_embedding(spectral.unsqueeze(-1))
        metab_emb = self.metabolite_feature_embedding(metabolite.unsqueeze(-1))
        
        spec_emb += self.pos_encoding_spec
        metab_emb += self.pos_encoding_metab
        
        spec_emb = self.embedding_dropout(self.embedding_norm_spec(spec_emb))
        metab_emb = self.embedding_dropout(self.embedding_norm_metab(metab_emb))
        
        self.attention_weights = {}
        
        for i, layer in enumerate(self.cross_attention_layers):
            spec_emb, metab_emb, attn_1_to_2, attn_2_to_1 = layer(spec_emb, metab_emb)
            if i == len(self.cross_attention_layers) - 1:
                self.attention_weights = {'1_to_2': attn_1_to_2, '2_to_1': attn_2_to_1}
        
        spec_pooled = spec_emb.mean(dim=1)
        metab_pooled = metab_emb.mean(dim=1)
        
        combined_pooled = torch.cat([spec_pooled, metab_pooled], dim=1)
        
        return {task_name: head(combined_pooled) for task_name, head in self.output_heads.items()}

# ===== TRAINING FUNCTIONS =====
def train_one_epoch(model, dataloader, optimizer, criterion, device, target_cols):
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        spectral = batch['spectral'].to(device)
        metab = batch['metabolite'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        outputs = model(spectral, metab)
        
        loss = sum(criterion(outputs[task_name], targets[:, i]) for i, task_name in enumerate(target_cols))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, target_cols):
    model.eval()
    total_loss = 0.0
    all_preds = {task: [] for task in target_cols}
    all_targets = {task: [] for task in target_cols}
    raw_attention_batches = {'1_to_2': [], '2_to_1': []}
    
    with torch.no_grad():
        for batch in dataloader:
            spectral = batch['spectral'].to(device)
            metab = batch['metabolite'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(spectral, metab)
            
            loss = sum(criterion(outputs[task_name], targets[:, i]) for i, task_name in enumerate(target_cols))
            total_loss += loss.item()
            
            for i, task_name in enumerate(target_cols):
                preds = torch.argmax(outputs[task_name], dim=1)
                all_preds[task_name].extend(preds.cpu().numpy())
                all_targets[task_name].extend(targets[:, i].cpu().numpy())
            
            if hasattr(model, 'attention_weights') and model.attention_weights:
                if '1_to_2' in model.attention_weights:
                    raw_attention_batches['1_to_2'].append(model.attention_weights['1_to_2'].detach().cpu())
                if '2_to_1' in model.attention_weights:
                    raw_attention_batches['2_to_1'].append(model.attention_weights['2_to_1'].detach().cpu())
    
    avg_loss = total_loss / len(dataloader)
    metrics = {task: f1_score(all_targets[task], all_preds[task], average='macro', zero_division=0) for task in target_cols}
    
    final_attention = {}
    if raw_attention_batches['1_to_2']:
        final_attention['1_to_2'] = torch.cat(raw_attention_batches['1_to_2'], dim=0)
    if raw_attention_batches['2_to_1']:
        final_attention['2_to_1'] = torch.cat(raw_attention_batches['2_to_1'], dim=0)
        
    return avg_loss, metrics, final_attention

# ===== PERMUTATION TESTS =====
def run_permutation_test(model, test_dataset, metab_feature_names, target_gene='NEAT1', num_permutations=100):
    """
    Runs a permutation test by permuting the gene expression values within each sample.
    
    Args:
        model: Trained transformer model
        test_dataset: Test dataset 
        metab_feature_names: List of metabolite/gene feature names
        target_gene: Gene to test (default: 'NEAT1')
        num_permutations: Number of permutation runs (default: 100)
    
    Returns:
        dict: Results including observed attention, null distribution, and p-value
    """
    logger.info(f"--- Starting Permutation Test for gene '{target_gene}' ({num_permutations} runs) ---")
    model.eval()
    
    # Check if target gene exists
    try: 
        target_gene_idx = metab_feature_names.index(target_gene)
        logger.info(f"Target gene '{target_gene}' found at index {target_gene_idx}")
    except ValueError: 
        logger.error(f"Target gene '{target_gene}' not found in feature list.")
        logger.info(f"Available features: {metab_feature_names[:10]}...") 
        return None
    
    # Step 1: Get the OBSERVED attention score for comparison
    logger.info("Computing observed attention scores...")
    observed_attention_scores = []
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for batch in test_loader:
            spectral, metab = batch['spectral'].to(DEVICE), batch['metabolite'].to(DEVICE)
            _ = model(spectral, metab)
            if hasattr(model, 'attention_weights') and '1_to_2' in model.attention_weights:
                attn_weights = model.attention_weights['1_to_2'].detach().cpu()
                # Extract attention to target gene: [batch, heads, spectral_features, target_gene]
                attn_to_target_gene = attn_weights[:, :, :, target_gene_idx]
                # Average across heads and spectral features for each sample
                avg_attn_per_sample = attn_to_target_gene.mean(dim=(1, 2))
                observed_attention_scores.extend(avg_attn_per_sample.numpy())
    
    if not observed_attention_scores:
        logger.error("Failed to extract observed attention scores")
        return None
        
    observed_mean_attention = np.mean(observed_attention_scores)
    observed_std_attention = np.std(observed_attention_scores)
    logger.info(f"Observed attention to {target_gene}: {observed_mean_attention:.6f} ± {observed_std_attention:.6f}")
    
    # Step 2: Run the NULL permutation tests
    logger.info("Running permutation tests...")
    null_attention_scores = []
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    for _ in tqdm(range(num_permutations), desc="Permutation Runs"):
        permuted_attention_scores_run = []
        
        with torch.no_grad():
            for batch in test_loader:
                spectral, metab = batch['spectral'].to(DEVICE), batch['metabolite'].to(DEVICE)
                
                # Permute the gene expression data
                batch_size = metab.shape[0]
                metab_permuted = metab.clone()
                
                # Permute gene expression values within each sample
                for sample_idx in range(batch_size):
                    permuted_indices = torch.randperm(metab.shape[1], device=DEVICE)
                    metab_permuted[sample_idx] = metab[sample_idx][permuted_indices]
                
                # Run model on permuted data
                _ = model(spectral, metab_permuted)
                
                if hasattr(model, 'attention_weights') and '1_to_2' in model.attention_weights:
                    attn_weights = model.attention_weights['1_to_2'].detach().cpu()
                    # Note: After permutation, target_gene_idx now points to a random gene
                    attn_to_target_position = attn_weights[:, :, :, target_gene_idx]
                    avg_attn_per_sample = attn_to_target_position.mean(dim=(1, 2))
                    permuted_attention_scores_run.extend(avg_attn_per_sample.numpy())
        
        if permuted_attention_scores_run: 
            null_attention_scores.append(np.mean(permuted_attention_scores_run))
    
    # Step 3: Calculate p-value and statistics
    if not null_attention_scores:
        logger.error("Failed to generate null distribution")
        return None
    
    null_mean = np.mean(null_attention_scores)
    null_std = np.std(null_attention_scores)
    
    # Calculate one-tailed p-value (testing if observed > null)
    count_extreme = np.sum(np.array(null_attention_scores) >= observed_mean_attention)
    p_value = (count_extreme + 1) / (len(null_attention_scores) + 1)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(observed_attention_scores) - 1) * observed_std_attention**2 + 
                          (len(null_attention_scores) - 1) * null_std**2) / 
                         (len(observed_attention_scores) + len(null_attention_scores) - 2))
    cohens_d = (observed_mean_attention - null_mean) / pooled_std if pooled_std > 0 else 0
    
    # Log results
    logger.info("="*60)
    logger.info("PERMUTATION TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Target Gene: {target_gene}")
    logger.info(f"Observed Attention: {observed_mean_attention:.6f} ± {observed_std_attention:.6f}")
    logger.info(f"Null Distribution: {null_mean:.6f} ± {null_std:.6f}")
    logger.info(f"Difference: {observed_mean_attention - null_mean:.6f}")
    logger.info(f"Effect Size (Cohen's d): {cohens_d:.3f}")
    logger.info(f"P-value: {p_value:.4f}")
    logger.info(f"Significant at alpha=0.05: {'YES' if p_value < 0.05 else 'NO'}")
    logger.info(f"Permutations showing higher attention: {count_extreme}/{len(null_attention_scores)}")
    logger.info("="*60)
    
    return {
        'target_gene': target_gene,
        'observed_attention_mean': float(observed_mean_attention),
        'observed_attention_std': float(observed_std_attention),
        'null_distribution_mean': float(null_mean),
        'null_distribution_std': float(null_std),
        'null_distribution': [float(x) for x in null_attention_scores],
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'count_extreme': int(count_extreme),
        'total_permutations': len(null_attention_scores),
        'method': 'gene_expression_permutation'
    }

def run_positional_permutation_test(model, test_dataset, metab_feature_names, target_gene='NEAT1', num_permutations=100):
    """
    Alternative permutation test that randomizes which gene index corresponds to the target.
    
    This tests whether the attention to the specific gene position is meaningful
    compared to attention to random gene positions.
    """
    logger.info(f"--- Starting Positional Permutation Test for gene '{target_gene}' ---")
    model.eval()
    
    try: 
        target_gene_idx = metab_feature_names.index(target_gene)
    except ValueError: 
        logger.error(f"Target gene '{target_gene}' not found. Aborting test.")
        return None
    
    # Get observed attention to the true target gene
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    observed_attention_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            spectral, metab = batch['spectral'].to(DEVICE), batch['metabolite'].to(DEVICE)
            _ = model(spectral, metab)
            if hasattr(model, 'attention_weights') and '1_to_2' in model.attention_weights:
                attn_weights = model.attention_weights['1_to_2'].detach().cpu()
                attn_to_target_gene = attn_weights[:, :, :, target_gene_idx]
                avg_attn_per_sample = attn_to_target_gene.mean(dim=(1, 2))
                observed_attention_scores.extend(avg_attn_per_sample.numpy())
    
    observed_mean_attention = np.mean(observed_attention_scores)
    
    # Generate null distribution by testing attention to random gene positions
    null_attention_scores = []
    available_indices = list(range(len(metab_feature_names)))
    available_indices.remove(target_gene_idx)
    
    np.random.seed(RANDOM_SEED)
    
    for _ in tqdm(range(num_permutations), desc="Random Gene Position Tests"):
        # Pick a random gene index (excluding the true target)
        random_gene_idx = np.random.choice(available_indices)
        
        random_attention_scores = []
        with torch.no_grad():
            for batch in test_loader:
                spectral, metab = batch['spectral'].to(DEVICE), batch['metabolite'].to(DEVICE)
                _ = model(spectral, metab)
                if hasattr(model, 'attention_weights') and '1_to_2' in model.attention_weights:
                    attn_weights = model.attention_weights['1_to_2'].detach().cpu()
                    attn_to_random_gene = attn_weights[:, :, :, random_gene_idx]
                    avg_attn_per_sample = attn_to_random_gene.mean(dim=(1, 2))
                    random_attention_scores.extend(avg_attn_per_sample.numpy())
        
        null_attention_scores.append(np.mean(random_attention_scores))
    
    # Calculate p-value
    count_extreme = np.sum(np.array(null_attention_scores) >= observed_mean_attention)
    p_value = (count_extreme + 1) / (len(null_attention_scores) + 1)
    
    logger.info(f"Positional permutation test P-VALUE for '{target_gene}': {p_value:.4f}")
    
    return {
        'target_gene': target_gene,
        'observed_attention': float(observed_mean_attention),
        'null_distribution': [float(x) for x in null_attention_scores],
        'p_value': float(p_value),
        'count_extreme': int(count_extreme),
        'total_permutations': len(null_attention_scores),
        'method': 'random_gene_position'
    }

# ===== MAIN EXECUTION =====
def main():
    start_time_main = time.time()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    train_loader, val_loader, test_loader, spec_feat, metab_feat, test_meta, num_classes, test_dataset = load_and_preprocess_data()
    
    model = SimplifiedTransformer(spectral_dim=len(spec_feat), metabolite_dim=len(metab_feat), hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, num_classes=num_classes, dropout=DROPOUT).to(DEVICE)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    logger.info("--- Starting Training ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(CHECKPOINT_SUBDIR, ANALYSIS_PAIRING, "best_model_hyperseq.pth")
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, TARGET_COLS)
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, DEVICE, TARGET_COLS)
        val_f1_avg = np.mean(list(val_metrics.values()))
        
        logger.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Avg Val F1: {val_f1_avg:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break
            
    logger.info("--- Training Finished. Evaluating on Test Set... ---")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_metrics, test_attention = evaluate(model, test_loader, criterion, DEVICE, TARGET_COLS)
    logger.info(f"Test Loss: {test_loss:.4f} | Test F1 Scores: {test_metrics}")

    # ===== PERMUTATION TESTING =====
    target_gene_for_test = 'NEAT1'
    if target_gene_for_test in metab_feat:
        logger.info("Running permutation test...")
        
        # Run the permutation test
        permutation_results = run_permutation_test(
            model, test_dataset, metab_feat, 
            target_gene=target_gene_for_test, 
            num_permutations=PERMUTATION_RUNS
        )
        
        # Also run the positional test for comparison
        logger.info("Running positional permutation test...")
        positional_results = run_positional_permutation_test(
            model, test_dataset, metab_feat, 
            target_gene=target_gene_for_test, 
            num_permutations=PERMUTATION_RUNS
        )
        
        # Save results
        if permutation_results:
            perm_results = {
                'permutation_test': permutation_results,
                'positional_permutation_test': positional_results,
                'analysis_notes': {
                    'permutation_method': 'Permutes gene expression values within samples',
                    'positional_method': 'Tests attention to random gene positions',
                    'recommendation': 'Use permutation_test results for primary analysis'
                }
            }
            
            perm_outfile = os.path.join(RESULTS_SUBDIR, f"permutation_test_results_{ANALYSIS_PAIRING}.json")
            with open(perm_outfile, 'w') as f: 
                json.dump(perm_results, f, indent=4)
            logger.info(f"Permutation test results saved to: {perm_outfile}")
            
            # Log final summary
            logger.info("="*80)
            logger.info("PERMUTATION TEST SUMMARY")
            logger.info("="*80)
            if permutation_results:
                logger.info(f"Permutation P-value: {permutation_results['p_value']:.4f}")
                logger.info(f"Effect size: {permutation_results['cohens_d']:.3f}")
                logger.info(f"Significant: {'YES' if permutation_results['p_value'] < 0.05 else 'NO'}")
            if positional_results:
                logger.info(f"Positional P-value: {positional_results['p_value']:.4f}")
            logger.info("="*80)
        
    else:
        logger.warning(f"Target gene '{target_gene_for_test}' not in selected features. Skipping permutation test.")

    logger.info("--- Saving Final Model Outputs ---")
    h5_path = os.path.join(RESULTS_SUBDIR, f"raw_attention_data_{ANALYSIS_PAIRING}.h5")
    with h5py.File(h5_path, 'w') as f:
        if test_attention and '1_to_2' in test_attention: 
            f.create_dataset('attention_spec_to_metab', data=test_attention['1_to_2'].numpy(), compression="gzip")
        if test_attention and '2_to_1' in test_attention: 
            f.create_dataset('attention_metab_to_spec', data=test_attention['2_to_1'].numpy(), compression="gzip")
        f.create_dataset('spectral_feature_names', data=np.array(spec_feat, dtype=h5py.string_dtype(encoding='utf-8')))
        f.create_dataset('metabolite_feature_names', data=np.array(metab_feat, dtype=h5py.string_dtype(encoding='utf-8')))
    logger.info(f"Raw attention data saved to: {h5_path}")

    meta_path_feather = os.path.join(RESULTS_SUBDIR, f"raw_attention_metadata_{ANALYSIS_PAIRING}.feather")
    test_meta.reset_index().to_feather(meta_path_feather)
    logger.info(f"Test set metadata saved to: {meta_path_feather}")

    logger.info(f"Total Runtime: {(time.time() - start_time_main)/60:.2f} minutes")
    logger.info("="*80)
    logger.info("TRANSFORMER TRAINING COMPLETE!")
    logger.info("="*80)

if __name__ == '__main__':
    main()
