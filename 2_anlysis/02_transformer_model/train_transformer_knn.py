# -*- coding: utf-8 -*-
"""
Transformer Model for Multi-Omic Plant Stress Response Analysis

This script implements a Transformer-based model for analyzing multi-omic plant stress response data.
It takes spectral and metabolite feature sets as input and uses a Transformer architecture with 
cross-attention mechanisms to perform multi-task classification (Genotype, Treatment, Day) while
extracting cross-modal attention patterns between different feature types.

Key capabilities:
1. Multi-task classification on plant stress response variables
2. Cross-modal attention analysis between spectral and metabolite features
3. Feature importance calculation based on attention mechanisms
4. Comparative performance evaluation against baseline models (Random Forest, KNN)

The model architecture incorporates cross-attention layers that enable bidirectional
information flow between the different omics data types, providing insights into 
their interactions and relative importance in prediction tasks.
"""

# ===== IMPORTS =====
import os
import sys
import time
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ===== CONFIGURATION =====
# --- Script Info ---
SCRIPT_NAME = "transformer_multi_omic_v1"
VERSION = "1.0.0_Skeleton"

# --- Analysis Pairing ---
ANALYSIS_PAIRING = "Leaf"  # Options: "Leaf" or "Root"

# --- Paths ---
BASE_DIR = r"C:/Users/ms/Desktop/hyper"
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "transformer")
MOFA_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "mofa")
CODE_DIR = os.path.join(BASE_DIR, "analysis")

# Construct input file paths dynamically based on ANALYSIS_PAIRING
INPUT_FILES = {
    "Leaf": {
        "spectral": os.path.join(MOFA_OUTPUT_DIR, "transformer_input_leaf_spectral.csv"),
        "metabolite": os.path.join(MOFA_OUTPUT_DIR, "transformer_input_leaf_metabolite.csv"),
    },
    "Root": {
        "spectral": os.path.join(MOFA_OUTPUT_DIR, "transformer_input_root_spectral.csv"),
        "metabolite": os.path.join(MOFA_OUTPUT_DIR, "transformer_input_root_metabolite.csv"),
    }
}

# --- Data & Columns ---
METADATA_COLS = ['Row_names', 'Vac_id', 'Genotype', 'Entry', 'Tissue.type',
                 'Batch', 'Treatment', 'Replication', 'Day']
TARGET_COLS = ['Genotype', 'Treatment', 'Day']

# --- Model Hyperparameters ---
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
NUM_CLASSES = {
    'Genotype': 2,
    'Treatment': 2,
    'Day': 3
}

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-5

# --- Data Handling ---
VAL_SIZE = 0.15
TEST_SIZE = 0.15
NUM_WORKERS = 0
RANDOM_SEED = 42

# --- Target Encoding ---
ENCODING_MAPS = {
    'Genotype': {'G1': 0, 'G2': 1},
    'Treatment': {0: 0, 1: 1},
    'Day': {1: 0, 2: 1, 3: 2}
}

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOGGING =====
def setup_logging(log_dir: str, script_name: str, version: str) -> logging.Logger:
    """
    Sets up file and console logging.
    
    Args:
        log_dir: Directory to store log files
        script_name: Name of the script for log file naming
        version: Version string for log file naming
        
    Returns:
        Logger object configured for both file and console output
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{script_name}_{version}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    log_format = '%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    logger.info(f"Logging setup complete. Log file: {log_filepath}")
    return logger

# Initialize logger (call this early)
logger = setup_logging(os.path.join(OUTPUT_DIR, "logs"), SCRIPT_NAME, VERSION)

logger.info("="*60)
logger.info(f"Starting Script: {SCRIPT_NAME} v{VERSION}")
logger.info(f"Analysis Pairing: {ANALYSIS_PAIRING}")
logger.info(f"Output Directory: {OUTPUT_DIR}")
logger.info(f"Input Directory (MOFA Output): {MOFA_OUTPUT_DIR}")
logger.info(f"Using Device: {DEVICE}")
logger.info(f"Batch Size: {BATCH_SIZE}, Num Workers: {NUM_WORKERS}")
logger.info(f"Model Params: Hidden Dim={HIDDEN_DIM}, Heads={NUM_HEADS}, Layers={NUM_LAYERS}")
logger.info("="*60)


# ===== DATA LOADING & PREPROCESSING =====

class PlantOmicsDataset(Dataset):
    """PyTorch Dataset for paired spectral and metabolite data."""
    
    def __init__(self, spectral_features, metabolite_features, targets):
        """
        Args:
            spectral_features (pd.DataFrame): Scaled spectral features (samples x features).
            metabolite_features (pd.DataFrame): Scaled metabolite features (samples x features).
            targets (pd.DataFrame): Encoded target labels (samples x num_targets).
        """
        if not spectral_features.index.equals(metabolite_features.index) or \
           not spectral_features.index.equals(targets.index):
            raise ValueError("Indices of spectral, metabolite, and target dataframes do not match!")

        # Convert to tensors
        self.spectral_data = torch.tensor(spectral_features.values, dtype=torch.float32)
        self.metabolite_data = torch.tensor(metabolite_features.values, dtype=torch.float32)
        # Ensure targets are Long type for CrossEntropyLoss
        self.targets = torch.tensor(targets.values, dtype=torch.long)

        self.sample_ids = spectral_features.index.tolist()  # Keep track of sample IDs

    def __len__(self):
        return len(self.spectral_data)

    def __getitem__(self, idx):
        """Returns a dictionary for clarity."""
        return {
            'spectral': self.spectral_data[idx],
            'metabolite': self.metabolite_data[idx],
            'targets': self.targets[idx],  # Shape: (num_targets,)
            'sample_id': self.sample_ids[idx]  # Include sample ID if needed later
        }

def load_and_preprocess_data(config: dict) -> tuple:
    """
    Loads, aligns, preprocesses, splits, and scales the data for the specified pairing.
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        tuple: Contains dataloaders, feature names, sample IDs, scalers, encoders,
               scaled dataframes, target dataframes, and full metadata
    """
    pairing = config['ANALYSIS_PAIRING']
    logger.info(f"--- Starting Data Loading & Preprocessing for Pairing: {pairing} ---")

    # --- 1. Load Data ---
    spectral_path = config['INPUT_FILES'][pairing]['spectral']
    metabolite_path = config['INPUT_FILES'][pairing]['metabolite']
    try:
        logger.info(f"Loading spectral data from: {spectral_path}")
        df_spectral_raw = pd.read_csv(spectral_path, index_col='Row_names', na_values='NA')
        logger.info(f"Loaded spectral data shape: {df_spectral_raw.shape}")
        logger.info(f"First 5 spectral indices: {df_spectral_raw.index[:5].tolist()}")

        logger.info(f"Loading metabolite data from: {metabolite_path}")
        df_metabolite_raw = pd.read_csv(metabolite_path, index_col='Row_names', na_values='NA')
        logger.info(f"Loaded metabolite data shape: {df_metabolite_raw.shape}")
        logger.info(f"First 5 metabolite indices: {df_metabolite_raw.index[:5].tolist()}")

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        raise
    except KeyError as e:
        logger.error(f"Column 'Row_names' not found in one of the input files: {e}. Check CSV generation.")
        raise
    except Exception as e:
        logger.error(f"Error loading data files: {e}")
        traceback.print_exc()
        raise


    # --- 2. Align Data (using index) ---
    # Now the indices should be the actual Row_names
    common_indices = df_spectral_raw.index.intersection(df_metabolite_raw.index)
    if len(common_indices) == 0:
        logger.error("No common Row_names found between spectral and metabolite files after loading.")
        raise ValueError("Alignment failed: No common Row_names.")
    elif len(common_indices) < len(df_spectral_raw.index) or len(common_indices) < len(df_metabolite_raw.index):
        logger.warning(f"Found {len(common_indices)} common samples out of "
                       f"{len(df_spectral_raw.index)} (spectral) and {len(df_metabolite_raw.index)} (metabolite). "
                       f"Proceeding with common samples only.")
        # Keep only common samples
        df_spectral_raw = df_spectral_raw.loc[common_indices]
        df_metabolite_raw = df_metabolite_raw.loc[common_indices]
    else:
        logger.info(f"Data aligned by Row_names index. Number of samples: {len(common_indices)}")


    # --- 3. Identify Columns & Separate ---
    meta_cols = config['METADATA_COLS']
    target_cols = config['TARGET_COLS']

    # Ensure 'Row_names' is handled correctly as it's now the index
    meta_cols_to_extract = [col for col in meta_cols if col != 'Row_names' and col in df_spectral_raw.columns]

    # Check if all metadata columns exist (excluding Row_names which is index)
    if not all(col in df_spectral_raw.columns for col in meta_cols_to_extract) or \
       not all(col in df_metabolite_raw.columns for col in meta_cols_to_extract):
        logger.warning(f"Missing one or more metadata columns in input files (checked: {meta_cols_to_extract}). Check METADATA_COLS.")
        # Allow proceeding if some meta cols are missing, but targets must exist

    # Separate metadata (use one source, assumed identical after alignment) and features
    # -----> FIX 1: Rename 'metadata' to 'full_metadata_df' <-----
    full_metadata_df = df_spectral_raw[meta_cols_to_extract].copy()
    # Keep index named correctly (should be 'Row_names' from load)
    if full_metadata_df.index.name != 'Row_names':
        logger.warning(f"Index name of full_metadata_df is '{full_metadata_df.index.name}', expected 'Row_names'.")
    # full_metadata_df['Row_names'] = full_metadata_df.index # Optional: Add as column if needed elsewhere

    features_spectral_df = df_spectral_raw.drop(columns=meta_cols_to_extract, errors='ignore') # errors='ignore' in case some meta cols were missing
    spectral_feature_names = features_spectral_df.columns.tolist()
    logger.info(f"Identified {len(spectral_feature_names)} spectral features.")

    features_metabolite_df = df_metabolite_raw.drop(columns=meta_cols_to_extract, errors='ignore')
    metabolite_feature_names = features_metabolite_df.columns.tolist()
    logger.info(f"Identified {len(metabolite_feature_names)} metabolite features.")

    # --- 4. Target Encoding ---
    logger.info("Encoding target variables...")
    # -----> Use the renamed variable 'full_metadata_df' <-----
    targets_encoded = pd.DataFrame(index=full_metadata_df.index)
    label_encoders = {}
    
    # --- DEBUG START ---
    if 'Treatment' in full_metadata_df.columns:
        # Convert to string and strip whitespace just in case, then find unique
        unique_treatments = full_metadata_df['Treatment'].astype(str).str.strip().unique()
        logger.info(f"Unique values found in 'Treatment' column BEFORE encoding: {unique_treatments}")
    else:
         logger.warning("Debugging: 'Treatment' column not found in metadata just before encoding loop.")
    # --- DEBUG END ---
    
    # Make sure target columns exist in the extracted metadata
    missing_targets = [col for col in target_cols if col not in full_metadata_df.columns]
    if missing_targets:
         logger.error(f"Target columns {missing_targets} not found in extracted full_metadata_df.")
         raise ValueError(f"Missing target columns: {missing_targets}")

    for col in target_cols:
        if col in config['ENCODING_MAPS']:
            logger.info(f"  Applying predefined map for '{col}'.")
            targets_encoded[col] = full_metadata_df[col].map(config['ENCODING_MAPS'][col])
            le = LabelEncoder()
            # Ensure classes_ are set in the order corresponding to 0, 1, 2... encoding
            sorted_items = sorted(config['ENCODING_MAPS'][col].items(), key=lambda item: item[1])
            le.classes_ = np.array([item[0] for item in sorted_items])
            label_encoders[col] = le
        else:
            logger.warning(f"  Encoding map not found for '{col}'. Using LabelEncoder.")
            le = LabelEncoder()
            targets_encoded[col] = le.fit_transform(full_metadata_df[col])
            label_encoders[col] = le
            logger.info(f"    '{col}' classes: {le.classes_} -> {le.transform(le.classes_)}")
        if targets_encoded[col].isnull().any():
             nan_indices = targets_encoded[targets_encoded[col].isnull()].index.tolist()
             logger.error(f"NaN values found in encoded target '{col}' for samples: {nan_indices}. "
                          f"Original values: {full_metadata_df.loc[nan_indices, col].unique()}. "
                          f"Check ENCODING_MAPS and original data.")
             raise ValueError(f"Encoding failed for target '{col}'.")
    logger.info("Target encoding complete.")

    # --- 5. Stratified Train/Validation/Test Split ---
    logger.info("Performing stratified train/validation/test split...")
    # -----> Use the renamed variable 'full_metadata_df' <-----
    try:
        full_metadata_df['stratify_key'] = full_metadata_df[target_cols[0]].astype(str)
        for col in target_cols[1:]:
            full_metadata_df['stratify_key'] += '_' + full_metadata_df[col].astype(str)
    except KeyError as e:
         logger.error(f"Stratification key column missing in full_metadata_df: {e}")
         raise

    indices = full_metadata_df.index
    stratify_values = full_metadata_df['stratify_key']

    # ... (NaN check in stratify_values) ...
    if stratify_values.isnull().any():
        logger.warning("NaN values found in stratification key. Stratification might be affected.")

    # ... (train_test_split calls using indices, stratify_values) ...
    try:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=(config['VAL_SIZE'] + config['TEST_SIZE']),
            random_state=config['RANDOM_SEED'],
            stratify=stratify_values
        )
    except ValueError as e:
         logger.warning(f"Could not stratify fully (possibly due to small groups or NaN in key): {e}. Proceeding without full stratification guarantee.")
         train_idx, temp_idx = train_test_split(
            indices,
            test_size=(config['VAL_SIZE'] + config['TEST_SIZE']),
            random_state=config['RANDOM_SEED'] # Fallback to non-stratified split
         )

    relative_test_size = config['TEST_SIZE'] / (config['VAL_SIZE'] + config['TEST_SIZE'])
    temp_stratify_values = stratify_values.loc[temp_idx]
    # ... (NaN check in temp_stratify_values) ...
    if temp_stratify_values.isnull().any():
        logger.warning("NaN values found in stratification key for temp set. Val/Test stratification might be affected.")

    try:
         val_idx, test_idx = train_test_split(
             temp_idx,
             test_size=relative_test_size,
             random_state=config['RANDOM_SEED'],
             stratify=temp_stratify_values
         )
    except ValueError as e:
         logger.warning(f"Could not stratify Temp set fully (possibly due to small groups or NaN in key): {e}. Proceeding without full stratification guarantee.")
         val_idx, test_idx = train_test_split(
             temp_idx,
             test_size=relative_test_size,
             random_state=config['RANDOM_SEED'] # Fallback to non-stratified split
         )

    X_train_spec = features_spectral_df.loc[train_idx]
    X_val_spec = features_spectral_df.loc[val_idx]
    X_test_spec = features_spectral_df.loc[test_idx]

    X_train_metab = features_metabolite_df.loc[train_idx]
    X_val_metab = features_metabolite_df.loc[val_idx]
    X_test_metab = features_metabolite_df.loc[test_idx]

    # -----> Use targets_encoded which was created using full_metadata_df index <-----
    y_train = targets_encoded.loc[train_idx]
    y_val = targets_encoded.loc[val_idx]
    y_test = targets_encoded.loc[test_idx]

    train_meta_ids = train_idx.tolist()
    val_meta_ids = val_idx.tolist()
    test_meta_ids = test_idx.tolist()

    logger.info(f"Split sizes: Train={len(train_idx)}, Validation={len(val_idx)}, Test={len(test_idx)}")

    # --- NEW DEBUG BLOCK ---
    logger.info("--- Debugging X_train_metab BEFORE Scaling ---")
    # Check for NaNs
    nan_counts_metab = X_train_metab.isnull().sum()
    nan_cols_metab = nan_counts_metab[nan_counts_metab > 0]
    if not nan_cols_metab.empty:
        logger.warning(f"NaNs found in X_train_metab BEFORE scaling in {len(nan_cols_metab)} columns.")
        logger.warning(f"Columns with NaNs (and counts):\n{nan_cols_metab}")
    else:
        logger.info("No NaNs found in X_train_metab before scaling.")

    # Check for Infs and non-numeric types
    inf_cols_metab = []
    non_numeric_cols_metab = []
    for col in X_train_metab.columns:
        if pd.api.types.is_numeric_dtype(X_train_metab[col]):
            if np.isinf(X_train_metab[col]).any():
                inf_cols_metab.append(col)
        else:
            non_numeric_cols_metab.append(col)

    if non_numeric_cols_metab:
        logger.warning(f"Non-numeric dtypes found in X_train_metab BEFORE scaling in {len(non_numeric_cols_metab)} columns: {non_numeric_cols_metab}")
        logger.warning("  These columns cannot be checked for Infs or scaled directly.")

    if inf_cols_metab:
        logger.warning(f"Infs found in X_train_metab BEFORE scaling in {len(inf_cols_metab)} columns: {inf_cols_metab}")
    elif not non_numeric_cols_metab: # Only log 'No Infs' if no non-numeric columns were found either
        logger.info("No Infs found in numeric columns of X_train_metab before scaling.")

    logger.info("--- End Debugging X_train_metab ---")
    # --- END NEW DEBUG BLOCK ---

    # --- 6. Scaling (Fit on Train only) ---
    logger.info("Scaling features (fitting on training data only)...")
    scaler_spec = StandardScaler()
    X_train_spec_scaled = scaler_spec.fit_transform(X_train_spec)
    X_val_spec_scaled = scaler_spec.transform(X_val_spec)
    X_test_spec_scaled = scaler_spec.transform(X_test_spec)

    scaler_metab = StandardScaler()
    X_train_metab_scaled = scaler_metab.fit_transform(X_train_metab)
    X_val_metab_scaled = scaler_metab.transform(X_val_metab)
    X_test_metab_scaled = scaler_metab.transform(X_test_metab)

    # --- Add this check after scaling ---
    logger.info("Checking for NaNs/Infs in scaled training data...")
    if np.isnan(X_train_spec_scaled).any() or np.isinf(X_train_spec_scaled).any():
         logger.error("NaNs or Infs found in SCALED spectral training data!")
         raise ValueError("Bad values in scaled spectral data")
    if np.isnan(X_train_metab_scaled).any() or np.isinf(X_train_metab_scaled).any():
         logger.error("NaNs or Infs found in SCALED metabolite training data!")
         raise ValueError("Bad values in scaled metabolite data")
    logger.info("No NaNs/Infs found in scaled training data.")
    # --- End check ---

    X_train_spec_scaled_df = pd.DataFrame(X_train_spec_scaled, index=train_idx, columns=spectral_feature_names)
    X_val_spec_scaled_df = pd.DataFrame(X_val_spec_scaled, index=val_idx, columns=spectral_feature_names)
    X_test_spec_scaled_df = pd.DataFrame(X_test_spec_scaled, index=test_idx, columns=spectral_feature_names)

    X_train_metab_scaled_df = pd.DataFrame(X_train_metab_scaled, index=train_idx, columns=metabolite_feature_names)
    X_val_metab_scaled_df = pd.DataFrame(X_val_metab_scaled, index=val_idx, columns=metabolite_feature_names)
    X_test_metab_scaled_df = pd.DataFrame(X_test_metab_scaled, index=test_idx, columns=metabolite_feature_names)

    scalers = {'spectral': scaler_spec, 'metabolite': scaler_metab}
    logger.info("Feature scaling complete.")

    # --- 7. Create Datasets and DataLoaders ---
    logger.info("Creating PyTorch Datasets and DataLoaders...")
    train_dataset = PlantOmicsDataset(X_train_spec_scaled_df, X_train_metab_scaled_df, y_train)
    val_dataset = PlantOmicsDataset(X_val_spec_scaled_df, X_val_metab_scaled_df, y_val)
    test_dataset = PlantOmicsDataset(X_test_spec_scaled_df, X_test_metab_scaled_df, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True)
    logger.info("DataLoaders created.")

    logger.info("--- Data Loading & Preprocessing Finished ---")

    # -----> FIX 2: Add 'full_metadata_df' to the return statement <-----
    return (train_loader, val_loader, test_loader,
            spectral_feature_names, metabolite_feature_names,
            train_meta_ids, val_meta_ids, test_meta_ids,
            scalers, label_encoders,
            X_train_spec_scaled_df, X_val_spec_scaled_df, X_test_spec_scaled_df,
            X_train_metab_scaled_df, X_val_metab_scaled_df, X_test_metab_scaled_df,
            y_train, y_val, y_test, # Return encoded target splits
            full_metadata_df # <--- ADD THIS
            )


# ===== MODEL DEFINITION =====
class CrossAttentionLayer(nn.Module):
    """Implements cross-attention between two modalities."""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        # Attention from modality 1 (query) to modality 2 (key/value)
        self.cross_attn_1_to_2 = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Attention from modality 2 (query) to modality 1 (key/value)
        self.cross_attn_2_to_1 = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

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
        """
        Processes inputs through cross-attention mechanisms.
        
        Args:
            x1: Features from modality 1 (batch, seq_len=1, hidden_dim)
            x2: Features from modality 2 (batch, seq_len=1, hidden_dim)
            
        Returns:
            out1, out2: Updated features for each modality
            attn_weights_1_to_2: Attention weights from modality 1 to 2
            attn_weights_2_to_1: Attention weights from modality 2 to 1
        """
        # Cross-Attention: x1 attends to x2
        attn_output_1, attn_weights_1_to_2 = self.cross_attn_1_to_2(x1, x2, x2)
        x1 = self.norm1(x1 + self.dropout(attn_output_1))  # Residual connection + Norm

        # Cross-Attention: x2 attends to x1
        attn_output_2, attn_weights_2_to_1 = self.cross_attn_2_to_1(x2, x1, x1)
        x2 = self.norm2(x2 + self.dropout(attn_output_2))  # Residual connection + Norm

        # Apply FFN to both
        ffn_output1 = self.ffn(x1)
        x1 = self.norm3(x1 + self.dropout(ffn_output1))

        ffn_output2 = self.ffn(x2)
        x2 = self.norm3(x2 + self.dropout(ffn_output2))

        return x1, x2, attn_weights_1_to_2, attn_weights_2_to_1


class SimplifiedTransformer(nn.Module):
    """Simplified Transformer model for multi-omic classification."""
    
    def __init__(self, spectral_dim, metabolite_dim, hidden_dim, num_heads, 
                 num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes  # Dict: {'Genotype': 2, 'Treatment': 2, 'Day': 3}

        # Input Embedding layers (Linear projection)
        self.spectral_embedding = nn.Linear(spectral_dim, hidden_dim)
        self.metabolite_embedding = nn.Linear(metabolite_dim, hidden_dim)

        # Simple Positional Encoding (Learned parameter) - Add to embeddings
        self.pos_encoding_spec = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_encoding_metab = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.embedding_norm_spec = nn.LayerNorm(hidden_dim)
        self.embedding_norm_metab = nn.LayerNorm(hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # Stack of Cross-Attention Layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Multi-task Classification Heads
        self.output_heads = nn.ModuleDict()
        total_feature_dim = hidden_dim * 2  # Concatenate outputs of both modalities

        for task_name, n_class in num_classes.items():
            self.output_heads[task_name] = nn.Sequential(
                nn.LayerNorm(total_feature_dim),
                nn.Linear(total_feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_class)
            )

        self.attention_weights = []  # To store attention weights during forward pass

    def forward(self, spectral, metabolite):
        """
        Forward pass through the model.
        
        Args:
            spectral: Spectral input features (batch, spectral_dim)
            metabolite: Metabolite input features (batch, metabolite_dim)
            
        Returns:
            dict: Dictionary mapping task names to prediction outputs
        """
        # 1. Embed inputs
        spec_emb = self.spectral_embedding(spectral)
        metab_emb = self.metabolite_embedding(metabolite)

        # 2. Add positional encoding and reshape for attention
        spec_emb = spec_emb.unsqueeze(1) + self.pos_encoding_spec
        metab_emb = metab_emb.unsqueeze(1) + self.pos_encoding_metab

        # Apply Norm and Dropout after pos enc
        spec_emb = self.embedding_dropout(self.embedding_norm_spec(spec_emb))
        metab_emb = self.embedding_dropout(self.embedding_norm_metab(metab_emb))

        # 3. Pass through Cross-Attention Layers
        self.attention_weights = []  # Clear previous weights
        all_attn_1_to_2 = []
        all_attn_2_to_1 = []
        for i, layer in enumerate(self.cross_attention_layers):
            spec_emb, metab_emb, attn_1_to_2, attn_2_to_1 = layer(spec_emb, metab_emb)
            # Store attention weights (e.g., from the last layer for analysis)
            if i == len(self.cross_attention_layers) - 1:
                self.attention_weights = {'1_to_2': attn_1_to_2, '2_to_1': attn_2_to_1}
            # Optionally store all layer weights
            all_attn_1_to_2.append(attn_1_to_2)
            all_attn_2_to_1.append(attn_2_to_1)

        # 4. Prepare for Classification Head
        spec_out = spec_emb.squeeze(1)
        metab_out = metab_emb.squeeze(1)

        # Concatenate features from both modalities
        combined_features = torch.cat([spec_out, metab_out], dim=1)

        # 5. Multi-task Classification
        outputs = {}
        for task_name, head in self.output_heads.items():
            outputs[task_name] = head(combined_features)

        return outputs


# ===== TRAINING FUNCTIONS =====
def train_one_epoch(model, dataloader, optimizer, criterion, device, target_cols):
    """
    Trains the model for one epoch.
    
    Args:
        model: Neural network model to train
        dataloader: DataLoader containing training data
        optimizer: Optimizer for updating model weights
        criterion: Loss function
        device: Device to run computations on (CPU/GPU)
        target_cols: List of target column names
        
    Returns:
        tuple: (average_loss, metrics_dict) for the epoch
    """
    model.train()
    total_loss = 0.0
    all_preds = {task: [] for task in target_cols}
    all_targets = {task: [] for task in target_cols}

    for i, batch in enumerate(dataloader):
        # Move data to device
        spectral = batch['spectral'].to(device)
        metabolite = batch['metabolite'].to(device)
        targets = batch['targets'].to(device)  # Shape: (batch, num_targets)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(spectral, metabolite)  # Dict of task outputs

        # Calculate loss (sum across tasks)
        loss = 0
        for task_idx, task_name in enumerate(target_cols):
            task_output = outputs[task_name]
            task_target = targets[:, task_idx]  # Get the correct target column
            loss += criterion(task_output, task_target)

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()

        total_loss += loss.item()

        # Store predictions and targets for metric calculation later
        with torch.no_grad():
            for task_idx, task_name in enumerate(target_cols):
                task_output = outputs[task_name]
                task_target = targets[:, task_idx]
                preds = torch.argmax(task_output, dim=1)
                all_preds[task_name].extend(preds.cpu().numpy())
                all_targets[task_name].extend(task_target.cpu().numpy())

        # Log progress periodically
        if (i + 1) % (len(dataloader) // 5) == 0:  # Log ~5 times per epoch
            logger.info(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)

    # Calculate epoch metrics
    epoch_metrics = {}
    for task_name in target_cols:
        accuracy = accuracy_score(all_targets[task_name], all_preds[task_name])
        # Use macro avg for f1/precision/recall as classes might be imbalanced
        f1 = f1_score(all_targets[task_name], all_preds[task_name], average='macro', zero_division=0)
        precision = precision_score(all_targets[task_name], all_preds[task_name], average='macro', zero_division=0)
        recall = recall_score(all_targets[task_name], all_preds[task_name], average='macro', zero_division=0)
        epoch_metrics[task_name] = {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

    return avg_loss, epoch_metrics


def evaluate(model, dataloader, criterion, device, target_cols):
    """
    Evaluates the model on the validation or test set.
    
    Args:
        model: Neural network model to evaluate
        dataloader: DataLoader containing evaluation data
        criterion: Loss function
        device: Device to run computations on (CPU/GPU)
        target_cols: List of target column names
        
    Returns:
        tuple: (average_loss, metrics_dict, predictions_df, targets_df, attention_weights)
    """
    model.eval()
    total_loss = 0.0
    all_preds = {task: [] for task in target_cols}
    all_targets = {task: [] for task in target_cols}
    all_sample_ids = []
    raw_attention_weights = {'1_to_2': [], '2_to_1': []}  # For analysis

    with torch.no_grad():
        for batch in dataloader:
            spectral = batch['spectral'].to(device)
            metabolite = batch['metabolite'].to(device)
            targets = batch['targets'].to(device)
            sample_ids = batch['sample_id']  # List of IDs in the batch

            outputs = model(spectral, metabolite)

            loss = 0
            for task_idx, task_name in enumerate(target_cols):
                task_output = outputs[task_name]
                task_target = targets[:, task_idx]
                loss += criterion(task_output, task_target)

            total_loss += loss.item()

            for task_idx, task_name in enumerate(target_cols):
                task_output = outputs[task_name]
                task_target = targets[:, task_idx]
                preds = torch.argmax(task_output, dim=1)
                all_preds[task_name].extend(preds.cpu().numpy())
                all_targets[task_name].extend(task_target.cpu().numpy())

            all_sample_ids.extend(sample_ids)

            # Store attention weights from the last layer (if available)
            if hasattr(model, 'attention_weights') and model.attention_weights:
                # Detach and move to CPU to avoid memory leaks
                raw_attention_weights['1_to_2'].append(model.attention_weights['1_to_2'].detach().cpu())
                raw_attention_weights['2_to_1'].append(model.attention_weights['2_to_1'].detach().cpu())

    avg_loss = total_loss / len(dataloader)

    eval_metrics = {}
    for task_name in target_cols:
        accuracy = accuracy_score(all_targets[task_name], all_preds[task_name])
        f1 = f1_score(all_targets[task_name], all_preds[task_name], average='macro', zero_division=0)
        precision = precision_score(all_targets[task_name], all_preds[task_name], average='macro', zero_division=0)
        recall = recall_score(all_targets[task_name], all_preds[task_name], average='macro', zero_division=0)
        eval_metrics[task_name] = {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

    # Concatenate attention weights from all batches
    if raw_attention_weights['1_to_2']:
        # weights are (batch, num_heads, query_len, key_len) - here query/key len is 1
        # Concatenate along batch dimension (dim 0)
        final_attn_1_to_2 = torch.cat(raw_attention_weights['1_to_2'], dim=0)  # Shape (N_samples, H, 1, 1)
        final_attn_2_to_1 = torch.cat(raw_attention_weights['2_to_1'], dim=0)  # Shape (N_samples, H, 1, 1)
        final_attention = {'1_to_2': final_attn_1_to_2, '2_to_1': final_attn_2_to_1}
    else:
        final_attention = None

    # Return predictions/targets as dataframes for easier analysis
    preds_df = pd.DataFrame(all_preds, index=all_sample_ids)
    targets_df = pd.DataFrame(all_targets, index=all_sample_ids)

    return avg_loss, eval_metrics, preds_df, targets_df, final_attention

# ===== ANALYSIS FUNCTIONS =====
def analyze_attention_view_level(attention_weights, test_metadata, output_dir, pairing):
    """
    Analyzes view-level attention weights using provided test set metadata.
    
    Calculates average view-to-view scores per sample, merges with metadata,
    and performs group comparisons based on the metadata.

    Args:
        attention_weights (dict): {'1_to_2': tensor(N, H, 1, 1), '2_to_1': tensor(N, H, 1, 1)}
                                  Raw attention weights from the test set.
        test_metadata (pd.DataFrame): DataFrame with metadata variables for test samples
                                      (index=Row_names/sample_id), containing original labels.
        output_dir (str): Directory to save results.
        pairing (str): 'Leaf' or 'Root'.

    Returns:
        tuple: (view_attention_df, grouped_attn_df, stats_df)
               - view_attention_df (pd.DataFrame or None): Per-sample scores merged with metadata.
               - grouped_attn_df (pd.DataFrame or None): Grouped statistics.
               - stats_df (pd.DataFrame or None): Statistical test results.
               Returns (None, None, None) if analysis fails.
    """
    logger.info("--- Starting VIEW-LEVEL Attention Analysis (Using Passed Metadata) ---")
    view_attention_df = None
    grouped_attn = None
    stat_df = None

    # --- Input Validation ---
    if not isinstance(attention_weights, dict) or \
       '1_to_2' not in attention_weights or '2_to_1' not in attention_weights or \
       not isinstance(attention_weights['1_to_2'], torch.Tensor) or \
       not isinstance(attention_weights['2_to_1'], torch.Tensor) or \
       attention_weights['1_to_2'].numel() == 0 or attention_weights['2_to_1'].numel() == 0:
        logger.warning("Attention weights dictionary is invalid, incomplete, or contains empty tensors. Skipping view-level analysis.")
        return None, None, None

    if not isinstance(test_metadata, pd.DataFrame) or test_metadata.empty:
        logger.error("Invalid or empty test_metadata DataFrame provided. Cannot proceed.")
        return None, None, None

    n_samples_attn = attention_weights['1_to_2'].shape[0]
    n_samples_meta = len(test_metadata)
    if n_samples_attn != n_samples_meta:
        logger.error(f"Mismatch between attention weight samples ({n_samples_attn}) and metadata samples ({n_samples_meta}). Cannot proceed.")
        # Ideally, the main function should ensure alignment before calling.
        # If they don't match here, it indicates a problem upstream.
        return None, None, None
    else:
        logger.info(f"Attention weights and metadata aligned for {n_samples_attn} test samples.")

    # --- Calculate Average Attention Scores ---
    try:
        attn_1_to_2_raw = attention_weights['1_to_2'] # Shape (N, H, 1, 1)
        attn_2_to_1_raw = attention_weights['2_to_1'] # Shape (N, H, 1, 1)

        avg_attn_1_to_2_per_sample = attn_1_to_2_raw.mean(dim=1).squeeze().cpu().numpy() # Shape (N,)
        avg_attn_2_to_1_per_sample = attn_2_to_1_raw.mean(dim=1).squeeze().cpu().numpy() # Shape (N,)

        # Ensure numpy arrays are 1D even if N=1
        if avg_attn_1_to_2_per_sample.ndim == 0: avg_attn_1_to_2_per_sample = avg_attn_1_to_2_per_sample.reshape(1)
        if avg_attn_2_to_1_per_sample.ndim == 0: avg_attn_2_to_1_per_sample = avg_attn_2_to_1_per_sample.reshape(1)

        # Create a DataFrame for scores, using the index from test_metadata
        scores_df = pd.DataFrame({
            'AvgAttn_Spec_to_Metab': avg_attn_1_to_2_per_sample,
            'AvgAttn_Metab_to_Spec': avg_attn_2_to_1_per_sample
        }, index=test_metadata.index) # Use the index from the input metadata

        # --- Merge with Passed Metadata ---
        # test_metadata already contains the original labels and correct index
        view_attention_df = test_metadata.join(scores_df, how='inner') # Inner join is safest

        if view_attention_df.empty:
             logger.error("Joining calculated attention scores with test_metadata resulted in an empty DataFrame. Check index alignment.")
             return None, None, None

        # Save per-sample scores with metadata
        outfile_samples = os.path.join(output_dir, f"transformer_view_level_attention_{pairing}.csv")
        view_attention_df.to_csv(outfile_samples, index=True, index_label=view_attention_df.index.name or 'Row_names') # Save with index
        logger.info(f"Saved per-sample view-level attention scores (with metadata) to {outfile_samples}")

        # --- Grouped Analysis ---
        # Use columns directly from view_attention_df (which came from test_metadata)
        grouping_cols_present = [col for col in TARGET_COLS if col in view_attention_df.columns] # Use TARGET_COLS for potential groups
        if not grouping_cols_present:
            logger.warning("No target columns ('Genotype', 'Treatment', 'Day') found in the merged attention data. Skipping grouped analysis.")
        else:
            logger.info(f"Calculating grouped view-level attention statistics by {grouping_cols_present}...")
            grouped_attn = view_attention_df.groupby(grouping_cols_present).agg(
                Mean_Attn_S2M=('AvgAttn_Spec_to_Metab', 'mean'),
                Std_Attn_S2M=('AvgAttn_Spec_to_Metab', 'std'),
                Median_Attn_S2M=('AvgAttn_Spec_to_Metab', 'median'),
                Mean_Attn_M2S=('AvgAttn_Metab_to_Spec', 'mean'),
                Std_Attn_M2S=('AvgAttn_Metab_to_Spec', 'std'),
                Median_Attn_M2S=('AvgAttn_Metab_to_Spec', 'median'),
                N_Samples=('AvgAttn_Spec_to_Metab', 'count')
            ).reset_index()

            grouped_outfile = os.path.join(output_dir, f"transformer_grouped_view_attention_{pairing}.csv")
            grouped_attn.to_csv(grouped_outfile, index=False)
            logger.info(f"Saved grouped view-level attention scores to {grouped_outfile}")

        # --- Statistical Tests (Mann-Whitney U) ---
        logger.info("Performing statistical tests on view-level attention...")
        stat_results = []
        attention_vars = ['AvgAttn_Spec_to_Metab', 'AvgAttn_Metab_to_Spec']

        # (Keep the safe_mwu helper function as defined in your provided code)
        def safe_mwu(group1_vals, group2_vals, var_name, comp_name, g1_name, g2_name):
            # ... (implementation of safe_mwu from your code) ...
            # Remove NaNs before checking length and performing test
            group1_vals = group1_vals.dropna()
            group2_vals = group2_vals.dropna()

            if len(group1_vals) < 3 or len(group2_vals) < 3:
                # logger.debug(f"Skipping MWU for {comp_name} on {var_name}: Insufficient non-NaN samples ({len(group1_vals)} vs {len(group2_vals)}).")
                return None
            try:
                # Use nan_policy='omit' just in case, although we dropped NaNs above
                stat, p_val = mannwhitneyu(group1_vals, group2_vals, alternative='two-sided', nan_policy='omit')
                # Check for p-value of NaN which can happen if input arrays are constant
                if np.isnan(p_val):
                    # Check if arrays are identical or constant
                    if len(group1_vals.unique()) == 1 and len(group2_vals.unique()) == 1 and group1_vals.unique()[0] == group2_vals.unique()[0]:
                         logger.debug(f"MWU for {comp_name} on {var_name}: Both groups identical constant value.")
                         return {'Variable': var_name, 'Comparison': comp_name, 'Group1': g1_name, 'Group2': g2_name, 'Statistic': stat, 'P_value': 1.0, 'Note': 'Identical constant values'}
                    else:
                         logger.warning(f"MWU for {comp_name} on {var_name} resulted in NaN p-value (inputs might be constant or have issues). Stat={stat}")
                         return {'Variable': var_name, 'Comparison': comp_name, 'Group1': g1_name, 'Group2': g2_name, 'Statistic': stat, 'P_value': np.nan, 'Note': 'MWU NaN p-value'}
                return {'Variable': var_name, 'Comparison': comp_name, 'Group1': g1_name, 'Group2': g2_name, 'Statistic': stat, 'P_value': p_val}
            except ValueError as ve:
                 # Handle specific ValueError cases
                 if "identical" in str(ve).lower() or "same distribution" in str(ve).lower():
                     logger.debug(f"MWU ValueError (Identical) for {comp_name} on {var_name}: {ve}")
                     # Check if they are truly identical constants
                     if len(group1_vals.unique()) == 1 and len(group2_vals.unique()) == 1 and group1_vals.unique()[0] == group2_vals.unique()[0]:
                         return {'Variable': var_name, 'Comparison': comp_name, 'Group1': g1_name, 'Group2': g2_name, 'Statistic': np.nan, 'P_value': 1.0, 'Note': 'Identical constant values'}
                     else:
                         return {'Variable': var_name, 'Comparison': comp_name, 'Group1': g1_name, 'Group2': g2_name, 'Statistic': np.nan, 'P_value': 1.0, 'Note': 'Identical distribution reported'}
                 elif "continuity correction" in str(ve).lower():
                      logger.debug(f"MWU continuity correction error for {comp_name} on {var_name}: {ve}. This might indicate many ties.")
                      # Attempt to calculate anyway, but note the issue. P-value might be less reliable.
                      try:
                          stat, p_val = mannwhitneyu(group1_vals, group2_vals, alternative='two-sided', nan_policy='omit', use_continuity=False) # Try without correction
                          logger.warning(f"Retried MWU without continuity correction for {comp_name} on {var_name}. P-value={p_val}")
                          return {'Variable': var_name, 'Comparison': comp_name, 'Group1': g1_name, 'Group2': g2_name, 'Statistic': stat, 'P_value': p_val, 'Note': 'Continuity correction issue (many ties?)'}
                      except Exception as ve_retry:
                          logger.warning(f"Retry MWU failed after continuity error for {comp_name} on {var_name}: {ve_retry}. Setting p=NaN.")
                          return {'Variable': var_name, 'Comparison': comp_name, 'Group1': g1_name, 'Group2': g2_name, 'Statistic': np.nan, 'P_value': np.nan, 'Note': 'Continuity error & retry failed'}
                 else:
                     logger.warning(f"Unhandled Mann-Whitney U ValueError for {comp_name} on {var_name}: {ve}")
                     return {'Variable': var_name, 'Comparison': comp_name, 'Group1': g1_name, 'Group2': g2_name, 'Statistic': np.nan, 'P_value': np.nan, 'Note': 'MWU ValueError'}
            except Exception as e:
                 logger.error(f"Unexpected error during Mann-Whitney U for {comp_name} on {var_name}: {e}")
                 return {'Variable': var_name, 'Comparison': comp_name, 'Group1': g1_name, 'Group2': g2_name, 'Statistic': np.nan, 'P_value': np.nan, 'Note': 'MWU Unexpected Error'}

        # Perform comparisons using columns directly from view_attention_df
        for att_var in attention_vars:
            # Compare Genotypes (overall)
            if 'Genotype' in view_attention_df.columns:
                genotypes = sorted(view_attention_df['Genotype'].dropna().unique()) # Sort for consistency
                if len(genotypes) == 2:
                    g1_vals = view_attention_df[view_attention_df['Genotype'] == genotypes[0]][att_var]
                    g2_vals = view_attention_df[view_attention_df['Genotype'] == genotypes[1]][att_var]
                    result = safe_mwu(g1_vals, g2_vals, att_var, 'Genotype', genotypes[0], genotypes[1])
                    if result: stat_results.append(result)

            # Compare Treatments (overall)
            if 'Treatment' in view_attention_df.columns:
                # Use Treatment column directly (assuming it contains comparable values like 0/1 or 'T0'/'T1')
                # Convert to string just to be safe with potential mixed types (e.g., 0 and '0')
                view_attention_df['Treatment_Str'] = view_attention_df['Treatment'].astype(str)
                treatments = sorted(view_attention_df['Treatment_Str'].dropna().unique())
                if len(treatments) == 2:
                    t0_vals = view_attention_df[view_attention_df['Treatment_Str'] == treatments[0]][att_var]
                    t1_vals = view_attention_df[view_attention_df['Treatment_Str'] == treatments[1]][att_var]
                    result = safe_mwu(t0_vals, t1_vals, att_var, 'Treatment', treatments[0], treatments[1])
                    if result: stat_results.append(result)
                view_attention_df.drop(columns=['Treatment_Str'], inplace=True, errors='ignore') # Clean up

            # Compare Treatment within each Genotype/Day
            if all(col in view_attention_df.columns for col in ['Genotype', 'Day', 'Treatment']):
                 for day in sorted(view_attention_df['Day'].dropna().unique()):
                    for geno in sorted(view_attention_df['Genotype'].dropna().unique()):
                         subset = view_attention_df[(view_attention_df['Day'] == day) & (view_attention_df['Genotype'] == geno)].copy()
                         if subset.empty: continue
                         subset['Treatment_Str'] = subset['Treatment'].astype(str)
                         treatments_sub = sorted(subset['Treatment_Str'].dropna().unique())
                         if len(treatments_sub) == 2:
                             t0_vals_sub = subset[subset['Treatment_Str'] == treatments_sub[0]][att_var]
                             t1_vals_sub = subset[subset['Treatment_Str'] == treatments_sub[1]][att_var]
                             comp_name = f'Treatment_in_G{geno}_D{day}'
                             result = safe_mwu(t0_vals_sub, t1_vals_sub, att_var, comp_name, treatments_sub[0], treatments_sub[1])
                             if result: stat_results.append(result)

        # --- FDR Correction and Saving ---
        if stat_results:
            stat_df = pd.DataFrame(stat_results)
            valid_pvals = stat_df['P_value'].dropna()
            if not valid_pvals.empty:
                try:
                    reject, pvals_corrected, _, _ = multipletests(valid_pvals, alpha=0.05, method='fdr_bh')
                    stat_df.loc[valid_pvals.index, 'P_value_FDR'] = pvals_corrected
                    stat_df.loc[valid_pvals.index, 'Significant_FDR'] = reject
                    # Fill NaNs for rows that didn't have a valid p-value initially or for correction results
                    stat_df['P_value_FDR'] = stat_df['P_value_FDR'] # Already assigned to specific indices
                    stat_df['Significant_FDR'] = stat_df['Significant_FDR'].fillna(False) # Fill NaNs in the boolean column
                except Exception as e_fdr:
                    logger.error(f"Error applying FDR correction: {e_fdr}. Skipping FDR.", exc_info=True)
                    stat_df['P_value_FDR'] = np.nan
                    stat_df['Significant_FDR'] = False
            else:
                stat_df['P_value_FDR'] = np.nan
                stat_df['Significant_FDR'] = False

            stat_outfile = os.path.join(output_dir, f"transformer_view_attention_stats_{pairing}.csv")
            stat_df.sort_values(['Comparison', 'Variable', 'P_value_FDR'], inplace=True, na_position='last')
            stat_df.to_csv(stat_outfile, index=False)
            logger.info(f"Saved view-level attention statistical comparisons to {stat_outfile}")
        else:
             logger.info("No statistical tests performed or yielded valid results for view-level attention.")

    except Exception as e_attn:
        logger.error(f"Error during view-level attention analysis: {e_attn}", exc_info=True)
        # Ensure function returns None tuple on error
        return None, None, None

    logger.info("--- View-Level Attention Analysis Finished ---")
    # Return the potentially generated dataframes
    return view_attention_df, grouped_attn, stat_df

# ===== VISUALIZATION FUNCTIONS =====
def plot_attention_heatmap(attention_weights, output_dir, pairing): # Removed spectral_features, metabolite_features, num_samples
    """ Plots a basic attention distribution for debugging. """ # Updated docstring
    logger.info("--- Generating Basic Attention Distribution Plot ---") # Updated log message
    # Check dictionary and tensor validity
    if not isinstance(attention_weights, dict) or \
       '1_to_2' not in attention_weights or \
       not isinstance(attention_weights['1_to_2'], torch.Tensor) or \
       attention_weights['1_to_2'].numel() == 0:
        logger.warning("Cannot plot attention distribution, attention_weights['1_to_2'] invalid or empty.")
        return

    # Example: Visualize average attention (across heads) from spec to metab
    try:
        # Squeeze is safe here as input is (N, H, 1, 1) -> mean(dim=1) -> (N, 1, 1) -> squeeze -> (N,)
        attn_1_to_2 = attention_weights['1_to_2'].mean(dim=1).squeeze().cpu().numpy()
        # Check if squeeze resulted in 0-dim array (if N=1)
        if attn_1_to_2.ndim == 0:
            attn_1_to_2 = attn_1_to_2.reshape(1) # Ensure it's at least 1D

        logger.info("Plotting distribution of average attention scores (Spectral -> Metabolite).")

        # Plot simple distribution of attention scores
        plt.figure(figsize=(8, 5))
        sns.histplot(attn_1_to_2, bins=30, kde=True) # Directly use the numpy array
        plt.title(f'Distribution of Avg Attention Scores (Spec->Metab) - {pairing}')
        plt.xlabel('Average Attention Score (Spec->Metab)')
        plt.ylabel('Frequency')
        plt.tight_layout() # Adjust layout

        viz_dir = os.path.join(output_dir, "visualizations") # Ensure viz dir exists
        os.makedirs(viz_dir, exist_ok=True)
        outfile = os.path.join(viz_dir, f"debug_attn_distribution_{pairing}.png")

        plt.savefig(outfile)
        plt.close()
        logger.info(f"Saved debug attention distribution plot to {outfile}")

    except Exception as e:
         logger.error(f"Error plotting basic attention distribution: {e}")
         traceback.print_exc()

def plot_final_visualizations(results_dict, output_dir, pairing):
    """ Generates all final publication-style visualizations. """
    logger.info("--- Generating Final Visualizations (Placeholder) ---")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Example calls to placeholder plot functions:
    # plot_network(results_dict['cross_modal_pairs'], viz_dir, pairing)
    # plot_heatmap(results_dict['temporal_attention'], viz_dir, pairing)
    # ... other plots from alternative_plan5.txt Section 5 ...

    logger.warning("Final visualization functions are placeholders.")
    logger.info("--- Final Visualizations Finished (Placeholder) ---")


# ===== BASELINE MODELS =====
# Placeholder functions - Implement details in Phase 5
def run_baseline_models(X_train_spec, X_train_metab, y_train,
                        X_test_spec, X_test_metab, y_test,
                        target_cols, output_dir, pairing, random_seed):
    """ Trains and evaluates baseline models (RF, LogReg). """
    logger.info("--- Running Baseline Models ---")

    baseline_results = []

    # Combine features for baselines that don't handle multi-view inherently
    X_train_combined = pd.concat([X_train_spec, X_train_metab], axis=1)
    X_test_combined = pd.concat([X_test_spec, X_test_metab], axis=1)
    logger.info(f"Combined feature shape for baselines: Train={X_train_combined.shape}, Test={X_test_combined.shape}")

    # --- Random Forest ---
    try:
        logger.info("Training Random Forest...")
        from sklearn.ensemble import RandomForestClassifier
        rf_model = RandomForestClassifier(n_estimators=200, random_state=random_seed, n_jobs=-1, max_depth=20, min_samples_leaf=5)
        # Multi-output wrapper not strictly needed if fitting per task, but can be used
        # from sklearn.multioutput import MultiOutputClassifier
        # multi_rf = MultiOutputClassifier(rf_model, n_jobs=-1)
        # multi_rf.fit(X_train_combined, y_train)
        # rf_preds = multi_rf.predict(X_test_combined)
        # rf_preds_df = pd.DataFrame(rf_preds, index=y_test.index, columns=target_cols)

        # Fit per task for clearer metrics
        rf_preds_dict = {}
        for i, task_name in enumerate(target_cols):
             rf_task = RandomForestClassifier(n_estimators=200, random_state=random_seed, n_jobs=-1, max_depth=20, min_samples_leaf=5)
             rf_task.fit(X_train_combined, y_train[task_name])
             rf_preds_dict[task_name] = rf_task.predict(X_test_combined)

        logger.info("Evaluating Random Forest...")
        for task_name in target_cols:
            task_target = y_test[task_name]
            task_preds = rf_preds_dict[task_name]
            accuracy = accuracy_score(task_target, task_preds)
            f1 = f1_score(task_target, task_preds, average='macro', zero_division=0)
            baseline_results.append({'Model': 'RandomForest', 'Task': task_name, 'Metric': 'Accuracy', 'Score': accuracy})
            baseline_results.append({'Model': 'RandomForest', 'Task': task_name, 'Metric': 'F1_Macro', 'Score': f1})

    except Exception as e:
        logger.error(f"Error running RandomForest baseline: {e}")
        traceback.print_exc()


    # --- K-Nearest Neighbors (KNN) --- # Replaced Logistic Regression
    try:
        logger.info("Training K-Nearest Neighbors (KNN)...")
        from sklearn.neighbors import KNeighborsClassifier
        # Need to fit per task
        knn_preds_dict = {}
        for i, task_name in enumerate(target_cols):
            # Initialize KNN (using default k=5 for simplicity, could be tuned)
            knn_task = KNeighborsClassifier(n_neighbors=5, n_jobs=-1) # Use default n_neighbors=5
            knn_task.fit(X_train_combined, y_train[task_name])
            knn_preds_dict[task_name] = knn_task.predict(X_test_combined)

        logger.info("Evaluating K-Nearest Neighbors (KNN)...")
        for task_name in target_cols:
            task_target = y_test[task_name]
            task_preds = knn_preds_dict[task_name]
            accuracy = accuracy_score(task_target, task_preds)
            f1 = f1_score(task_target, task_preds, average='macro', zero_division=0)
            baseline_results.append({'Model': 'KNN', 'Task': task_name, 'Metric': 'Accuracy', 'Score': accuracy})
            baseline_results.append({'Model': 'KNN', 'Task': task_name, 'Metric': 'F1_Macro', 'Score': f1})

    except Exception as e:
        logger.error(f"Error running KNN baseline: {e}")
        traceback.print_exc()

    # --- Save Baseline Results ---
    if baseline_results:
        baseline_df = pd.DataFrame(baseline_results)
        outfile = os.path.join(output_dir, f"transformer_baseline_comparison_{pairing}.csv")
        baseline_df.to_csv(outfile, index=False)
        logger.info(f"Baseline comparison results saved to {outfile}")
    else:
        logger.warning("No baseline results generated.")

    logger.info("--- Baseline Models Finished ---")
    return baseline_df


# ===== MAIN EXECUTION =====
def main():
    """Main execution function that handles the training and evaluation pipeline."""
    start_time = time.time()
    logger.info("--- Starting Main Execution ---")

    # --- Set Seed ---
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- Load Data ---
    try:
        (train_loader, val_loader, test_loader,
         spectral_feature_names, metabolite_feature_names,
         train_ids, val_ids, test_ids,
         scalers, label_encoders,
         X_train_spec_scaled_df, X_val_spec_scaled_df, X_test_spec_scaled_df,
         X_train_metab_scaled_df, X_val_metab_scaled_df, X_test_metab_scaled_df,
         y_train_df, y_val_df, y_test_df,
         full_metadata_df
         ) = load_and_preprocess_data(config=globals())
    except Exception as e:
        logger.error(f"Failed during data loading. Exiting. Error: {e}", exc_info=True)
        return

    # --- Initialize Model, Loss, Optimizer ---
    logger.info("Initializing model, loss, and optimizer...")
    spectral_dim = len(spectral_feature_names)
    metabolite_dim = len(metabolite_feature_names)

    model = SimplifiedTransformer(
        spectral_dim=spectral_dim,
        metabolite_dim=metabolite_dim,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)

    logger.info(f"Model initialized:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Training Loop ---
    logger.info("--- Starting Training ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_start_time = time.time()

    # Create directory for model checkpoints
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints", ANALYSIS_PAIRING)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_model_{SCRIPT_NAME}_{ANALYSIS_PAIRING}.pth")

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{EPOCHS}")

        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, TARGET_COLS)
        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, DEVICE, TARGET_COLS)

        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        # Log metrics per task
        for task in TARGET_COLS:
            logger.info(f"  {task}: Train Acc={train_metrics[task]['accuracy']:.4f}, "
                       f"Val Acc={val_metrics[task]['accuracy']:.4f} | "
                       f"Train F1={train_metrics[task]['f1']:.4f}, "
                       f"Val F1={val_metrics[task]['f1']:.4f}")

        # --- Early Stopping & Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model state
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Validation loss improved. Saved best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"Validation loss did not improve for {epochs_no_improve} epochs.")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    training_duration = time.time() - training_start_time
    logger.info(f"--- Training Finished --- Duration: {training_duration:.2f} seconds")

    # --- Evaluation on Test Set ---
    logger.info("--- Evaluating on Test Set using Best Model ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}")
    else:
        logger.warning("Best model checkpoint not found. Evaluating with the last state.")

    test_loss, test_metrics, test_preds_df, test_targets_df, test_attention = evaluate(
        model, test_loader, criterion, DEVICE, TARGET_COLS
    )

    logger.info(f"Test Loss: {test_loss:.4f}")
    performance_data = []
    for task in TARGET_COLS:
        logger.info(f"  Test Metrics for {task}:")
        logger.info(f"    Accuracy: {test_metrics[task]['accuracy']:.4f}")
        logger.info(f"    F1 Macro: {test_metrics[task]['f1']:.4f}")
        logger.info(f"    Precision Macro: {test_metrics[task]['precision']:.4f}")
        logger.info(f"    Recall Macro: {test_metrics[task]['recall']:.4f}")
        performance_data.append({'Task': task, 'Metric': 'Accuracy', 'Score': test_metrics[task]['accuracy']})
        performance_data.append({'Task': task, 'Metric': 'F1_Macro', 'Score': test_metrics[task]['f1']})
        performance_data.append({'Task': task, 'Metric': 'Precision_Macro', 'Score': test_metrics[task]['precision']})
        performance_data.append({'Task': task, 'Metric': 'Recall_Macro', 'Score': test_metrics[task]['recall']})

    # Save test performance metrics
    performance_df = pd.DataFrame(performance_data)
    perf_outfile = os.path.join(OUTPUT_DIR, f"transformer_class_performance_{ANALYSIS_PAIRING}.csv")
    performance_df.to_csv(perf_outfile, index=False)
    logger.info(f"Transformer test performance metrics saved to {perf_outfile}")

    # --- Prepare Metadata for Analysis ---
    logger.info("Preparing test metadata subset...")
    test_metadata_subset = None
    test_results_df = None

    # Ensure the full metadata index is set correctly
    if 'full_metadata_df' not in locals() or full_metadata_df is None:
        logger.error("full_metadata_df not available. Cannot prepare test metadata.")
    else:
        if full_metadata_df.index.name != 'Row_names':
            if 'Row_names' in full_metadata_df.columns:
                logger.warning(f"Full metadata index was '{full_metadata_df.index.name}', "
                              f"setting 'Row_names' column as index.")
                try:
                    full_metadata_df = full_metadata_df.set_index('Row_names')
                except Exception as e_idx:
                    logger.error(f"Failed to set 'Row_names' as index: {e_idx}.")
                    full_metadata_df = None
            else:
                logger.error(f"Cannot find 'Row_names' column or index.")
                full_metadata_df = None

        # Select metadata only for the test samples using the test_ids (Row_names)
        if full_metadata_df is not None:
            if not test_ids:
                logger.error("test_ids (Row_names) not available for selecting test metadata.")
            else:
                try:
                    # Check if all test_ids are actually in the metadata index
                    missing_ids_in_meta = [idx for idx in test_ids if idx not in full_metadata_df.index]
                    valid_test_ids = [idx for idx in test_ids if idx in full_metadata_df.index]

                    if missing_ids_in_meta:
                        logger.warning(f"{len(missing_ids_in_meta)} out of {len(test_ids)} "
                                      f"test_ids not found in full_metadata_df index.")

                    if not valid_test_ids:
                        logger.error("No common test IDs found between test set and full metadata index.")
                    else:
                        test_metadata_subset = full_metadata_df.loc[valid_test_ids].copy()
                        logger.info(f"Created test metadata subset for {len(valid_test_ids)} common test samples.")

                        # --- Optional: Create the full results file ---
                        if test_metadata_subset is not None and not test_metadata_subset.empty:
                            # Align predictions and targets to the metadata subset index
                            common_pred_idx = test_metadata_subset.index.intersection(test_preds_df.index)
                            common_target_idx = test_metadata_subset.index.intersection(test_targets_df.index)

                            if len(common_pred_idx) < len(test_metadata_subset.index):
                                logger.warning(f"Prediction index missing "
                                              f"{len(test_metadata_subset.index) - len(common_pred_idx)} "
                                              f"samples from test metadata subset.")
                            if len(common_target_idx) < len(test_metadata_subset.index):
                                logger.warning(f"Target index missing "
                                              f"{len(test_metadata_subset.index) - len(common_target_idx)} "
                                              f"samples from test metadata subset.")

                            # Align DFs before joining
                            test_preds_df_aligned = test_preds_df.loc[common_pred_idx]
                            test_targets_df_aligned = test_targets_df.loc[common_target_idx]
                            test_metadata_subset_aligned = test_metadata_subset.loc[common_pred_idx]

                            test_preds_df_renamed = test_preds_df_aligned.rename(
                                columns={col: f"{col}_pred_encoded" for col in TARGET_COLS})
                            test_targets_df_renamed = test_targets_df_aligned.rename(
                                columns={col: f"{col}_true_encoded" for col in TARGET_COLS})

                            test_results_df = test_metadata_subset_aligned.join(test_preds_df_renamed, how='left')
                            test_results_df = test_results_df.join(test_targets_df_renamed, how='left')

                            # Add decoded labels
                            logger.info("Decoding labels for the full results file...")
                            for task_name in TARGET_COLS:
                                if task_name in label_encoders:
                                    encoder = label_encoders[task_name]
                                    pred_col_encoded = f"{task_name}_pred_encoded"
                                    true_col_encoded = f"{task_name}_true_encoded"
                                    pred_col_decoded = f"{task_name}_pred"
                                    true_col_decoded = f"{task_name}_true"

                                    # Decode predictions
                                    if pred_col_encoded in test_results_df.columns:
                                        valid_preds = test_results_df[pred_col_encoded].dropna()
                                        if not valid_preds.empty:
                                            try:
                                                decoded_preds = encoder.inverse_transform(valid_preds.astype(int))
                                                test_results_df[pred_col_decoded] = pd.Series(
                                                    decoded_preds, index=valid_preds.index)
                                            except Exception as e_dec_pred:
                                                logger.error(f"Error decoding predictions for '{task_name}': "
                                                           f"{e_dec_pred}", exc_info=True)
                                    # Get true decoded labels
                                    if true_col_decoded not in test_results_df.columns:
                                        if task_name in test_metadata_subset.columns:
                                            test_results_df[true_col_decoded] = test_metadata_subset[task_name]
                                        else:
                                            logger.warning(f"Original metadata column '{task_name}' "
                                                         f"not found for true decoded label.")
                                else:
                                    logger.warning(f"Label encoder not found for task '{task_name}'. Cannot decode.")

                            results_outfile = os.path.join(
                                OUTPUT_DIR, f"transformer_test_predictions_metadata_{ANALYSIS_PAIRING}.csv")
                            try:
                                test_results_df.to_csv(results_outfile, index=True, index_label='Row_names')
                                logger.info(f"Full test results with metadata saved to {results_outfile}")
                            except Exception as e_save:
                                logger.error(f"Failed to save full test results: {e_save}")

                except KeyError as e:
                    logger.error(f"Some test_ids (Row_names) not found in full_metadata_df index: {e}.")
                except Exception as e:
                    logger.error(f"Unexpected error selecting test metadata subset: {e}", exc_info=True)

    # --- Attention Analysis ---
    logger.info("Starting view-level attention analysis using test metadata...")

    # Check if test_metadata_subset is valid before passing
    if test_metadata_subset is None or test_metadata_subset.empty:
        logger.warning("Test metadata subset is missing or empty. Skipping view-level attention analysis.")
        view_level_attn_df, grouped_attn, stat_df = None, None, None
    else:
        if 'analyze_attention_view_level' in globals():
            view_level_attn_df, grouped_attn, stat_df = analyze_attention_view_level(
                test_attention,
                test_metadata_subset,
                OUTPUT_DIR,
                ANALYSIS_PAIRING
            )
        else:
            logger.error("Function 'analyze_attention_view_level' not found.")
            view_level_attn_df, grouped_attn, stat_df = None, None, None

    # --- Baseline Model Comparison ---
    logger.info("Running baseline model comparison...")
    baseline_comparison_df = run_baseline_models(
        X_train_spec=X_train_spec_scaled_df, X_train_metab=X_train_metab_scaled_df, y_train=y_train_df,
        X_test_spec=X_test_spec_scaled_df, X_test_metab=X_test_metab_scaled_df, y_test=y_test_df,
        target_cols=TARGET_COLS, output_dir=OUTPUT_DIR, pairing=ANALYSIS_PAIRING, random_seed=RANDOM_SEED
    )

    # --- Visualization ---
    plot_attention_heatmap(test_attention, OUTPUT_DIR, ANALYSIS_PAIRING)

    # Update final_results dictionary
    final_results = {
        'view_level_attention': view_level_attn_df,
        'grouped_view_attention': grouped_attn,
        'view_attention_stats': stat_df,
        'baseline_comparison': baseline_comparison_df,
        'test_results_with_metadata': test_results_df
    }
    plot_final_visualizations(final_results, OUTPUT_DIR, ANALYSIS_PAIRING)

    # --- End ---
    total_duration = time.time() - start_time
    logger.info(f"--- Main Execution Finished --- Total Duration: {total_duration / 60:.2f} minutes ---")
    logger.info("="*60)


# --- Entry Point ---
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred during main execution: {e}", exc_info=True)
        sys.exit(1)