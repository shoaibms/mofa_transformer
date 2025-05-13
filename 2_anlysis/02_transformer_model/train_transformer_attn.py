# -*- coding: utf-8 -*-
"""
Transformer Model for Multi-Omic Plant Stress Response Analysis

This script implements a Transformer model with feature-level cross-attention
for analyzing plant stress response using multiple omics data types. The model 
takes filtered feature sets (~50 features/view) selected by MOFA+ as input
and performs multi-task classification for genotype, treatment, and day.

Key components:
1. Data loading and preprocessing from MOFA+ outputs
2. Transformer with feature-level cross-attention mechanism
3. Multi-task learning for classification tasks
4. Extraction and analysis of cross-modal attention patterns
5. Baseline model comparison with traditional ML methods

Outputs include model performance metrics, feature importance scores,
cross-modal feature pairs, and raw attention tensors for further analysis.

Workflow:
1. Load preprocessed data (_50feat.csv files).
2. Optionally subset features further for testing.
3. Define Transformer model with per-feature embeddings and cross-attention.
4. Train the model for multi-task classification (Genotype, Treatment, Day).
5. Evaluate on the test set.
6. Extract FEATURE-LEVEL attention weights (Spec<->Metab).
7. Calculate basic cross-modal pairs and feature importance from attention.
8. **Save raw attention tensor, predictions, and full test metadata for offline analysis/visualization.**
9. Run baseline models for comparison.
"""

# ===== IMPORTS =====
import os
import sys
import time
import logging
import traceback
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
# Visualization imports removed as per request
# import matplotlib.pyplot as plt
# import seaborn as sns
import h5py # For saving large attention tensors efficiently

# Baselines (Keep for comparison)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline # Needed for baseline CV

# ===== CONFIGURATION =====

# --- Script Info ---
SCRIPT_NAME = "transformer_multi_omic_v3_attention"
VERSION = "1.2.2_AvgWeightsFix" # Version update

# --- Paths ---
BASE_DIR = r"C:/Users/ms/Desktop/hyper"
# Input files are from the mofa50 run
MOFA_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "mofa") # <<< POINT TO mofa50 SUBDIR
# Specific output directory for this version
TRANSFORMER_BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "transformer")
OUTPUT_DIR = os.path.join(TRANSFORMER_BASE_OUTPUT_DIR, "v3_feature_attention") # <<< NEW OUTPUT DIR
CHECKPOINT_SUBDIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_SUBDIR = os.path.join(OUTPUT_DIR, "logs")
RESULTS_SUBDIR = os.path.join(OUTPUT_DIR, "results") # For CSVs, attention data

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_SUBDIR, exist_ok=True)
os.makedirs(LOG_SUBDIR, exist_ok=True)
os.makedirs(RESULTS_SUBDIR, exist_ok=True)

# --- Analysis Pairing ---
# Choose 'Leaf' or 'Root' to determine which data pairing to analyze
ANALYSIS_PAIRING = "Root" # Or "Leaf" - CHANGE THIS MANUALLY

# --- Input Data Files --- (Using _50feat suffix)
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

# --- Feature Subsetting (Optional for Testing) ---
# Set to a small number (e.g., 20) to test run quickly. Set high (e.g., 1000) to use all available features.
MAX_FEATURES_PER_VIEW_RUNTIME = 1000 # Example for quick test run

# --- Data & Columns ---
METADATA_COLS = ['Row_names', 'Vac_id', 'Genotype', 'Entry', 'Tissue.type',
                 'Batch', 'Treatment', 'Replication', 'Day']
TARGET_COLS = ['Genotype', 'Treatment', 'Day']

# --- Model Hyperparameters ---
HIDDEN_DIM = 64
NUM_HEADS = 4 # Keep relatively low for memory
NUM_LAYERS = 2 # Keep shallow
DROPOUT = 0.1
NUM_CLASSES = {'Genotype': 2, 'Treatment': 2, 'Day': 3}

# --- Training Hyperparameters ---
LEARNING_RATE = 5e-5 # Can try slightly higher for smaller model/data
BATCH_SIZE = 16 # <<< Start VERY SMALL due to attention complexity & 6GB VRAM
EPOCHS = 150 # Train potentially longer, but rely on early stopping
EARLY_STOPPING_PATIENCE = 15 # Increase patience slightly
WEIGHT_DECAY = 1e-5

# --- Data Handling ---
VAL_SIZE = 0.15
TEST_SIZE = 0.15
NUM_WORKERS = 0 # <<< Set to 0 for Windows, especially with complex models/limited RAM, can increase later if needed
RANDOM_SEED = 42
ENCODING_MAPS = {'Genotype': {'G1': 0, 'G2': 1}, 'Treatment': {0: 0, 1: 1}, 'Day': {1: 0, 2: 1, 3: 2}}

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOGGING =====
def setup_logging(log_dir: str, script_name: str, version: str) -> logging.Logger:
    """Set up logging with file and console handlers."""
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

logger = setup_logging(LOG_SUBDIR, SCRIPT_NAME, VERSION)

logger.info("="*60)
logger.info(f"Starting Script: {SCRIPT_NAME} v{VERSION} (Feature-Level Attention)")
logger.info(f"Analysis Pairing: {ANALYSIS_PAIRING}")
logger.info(f"Output Directory: {OUTPUT_DIR}")
logger.info(f"Input Directory (MOFA Features): {MOFA_OUTPUT_DIR}")
logger.info(f"Using Device: {DEVICE}")
logger.info(f"Batch Size: {BATCH_SIZE}, Num Workers: {NUM_WORKERS}")
logger.info(f"Max Features Per View (Runtime): {MAX_FEATURES_PER_VIEW_RUNTIME}")
logger.info(f"Model Params: Hidden Dim={HIDDEN_DIM}, Heads={NUM_HEADS}, Layers={NUM_LAYERS}")
logger.info("="*60)

# ===== DATA LOADING & PREPROCESSING =====

class PlantOmicsDataset(Dataset):
    """PyTorch Dataset for paired spectral and metabolite data."""
    def __init__(self, spectral_features, metabolite_features, targets):
        if not spectral_features.index.equals(metabolite_features.index) or \
           not spectral_features.index.equals(targets.index):
            raise ValueError("Indices of spectral, metabolite, and target dataframes do not match!")
        self.spectral_data = torch.tensor(spectral_features.values, dtype=torch.float32)
        self.metabolite_data = torch.tensor(metabolite_features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.long)
        self.sample_ids = spectral_features.index.tolist()
        
    def __len__(self):
        return len(self.spectral_data)
    
    def __getitem__(self, idx):
        return {
            'spectral': self.spectral_data[idx],
            'metabolite': self.metabolite_data[idx],
            'targets': self.targets[idx],
            'sample_id': self.sample_ids[idx]
        }

def load_and_preprocess_data(config: dict) -> tuple:
    """
    Loads, aligns, preprocesses, and splits data for training.
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        Tuple containing data loaders, feature names, indices, scalers,
        encoders, dataframes, and test metadata
    """
    pairing = config['ANALYSIS_PAIRING']
    max_features = config['MAX_FEATURES_PER_VIEW_RUNTIME']
    logger.info(f"--- Starting Data Loading & Preprocessing ({pairing}, Max Feat={max_features}) ---")

    spectral_path = config['INPUT_FILES'][pairing]['spectral']
    metabolite_path = config['INPUT_FILES'][pairing]['metabolite']

    try:
        logger.info(f"Loading spectral data: {spectral_path}")
        df_spectral_raw = pd.read_csv(spectral_path, index_col='Row_names', na_values='NA')
        logger.info(f"Loading metabolite data: {metabolite_path}")
        df_metabolite_raw = pd.read_csv(metabolite_path, index_col='Row_names', na_values='NA')
    except Exception as e:
        logger.error(f"Error loading data files: {e}")
        raise

    # Align samples between datasets
    common_indices = df_spectral_raw.index.intersection(df_metabolite_raw.index)
    if len(common_indices) == 0:
        raise ValueError("Alignment failed: No common Row_names.")
    if len(common_indices) < len(df_spectral_raw.index) or len(common_indices) < len(df_metabolite_raw.index):
        logger.warning(f"Found {len(common_indices)} common samples. Proceeding with common samples only.")
    
    df_spectral_raw = df_spectral_raw.loc[common_indices]
    df_metabolite_raw = df_metabolite_raw.loc[common_indices]
    logger.info(f"Data aligned by Row_names index. Samples: {len(common_indices)}")

    # Extract metadata and features
    meta_cols = config['METADATA_COLS']
    target_cols = config['TARGET_COLS']
    meta_cols_to_extract = [col for col in meta_cols if col != 'Row_names' and col in df_spectral_raw.columns]

    # Store the full metadata for ALL common samples before splitting
    full_metadata_df = df_spectral_raw[meta_cols_to_extract].copy()
    
    # Keep index name consistent
    if full_metadata_df.index.name != 'Row_names':
        logger.warning(f"Renaming full_metadata_df index from '{full_metadata_df.index.name}' to 'Row_names'.")

    # Extract features
    features_spectral_df = df_spectral_raw.drop(columns=meta_cols_to_extract, errors='ignore')
    features_metabolite_df = df_metabolite_raw.drop(columns=meta_cols_to_extract, errors='ignore')

    # Optional Feature Subsetting
    if max_features < features_spectral_df.shape[1]:
        logger.info(f"Subsetting spectral features from {features_spectral_df.shape[1]} to {max_features}.")
        features_spectral_df = features_spectral_df.iloc[:, :max_features]
    if max_features < features_metabolite_df.shape[1]:
        logger.info(f"Subsetting metabolite features from {features_metabolite_df.shape[1]} to {max_features}.")
        features_metabolite_df = features_metabolite_df.iloc[:, :max_features]

    spectral_feature_names = features_spectral_df.columns.tolist()
    metabolite_feature_names = features_metabolite_df.columns.tolist()
    logger.info(f"Using {len(spectral_feature_names)} spectral features and {len(metabolite_feature_names)} metabolite features.")

    # Target Encoding
    targets_encoded = pd.DataFrame(index=full_metadata_df.index)
    label_encoders = {}
    missing_targets = [col for col in target_cols if col not in full_metadata_df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in metadata: {missing_targets}")

    for col in target_cols:
        if col in config['ENCODING_MAPS']:
            targets_encoded[col] = full_metadata_df[col].map(config['ENCODING_MAPS'][col])
            # Create LabelEncoder consistent with the map for inverse transform later
            le = LabelEncoder()
            sorted_items = sorted(config['ENCODING_MAPS'][col].items(), key=lambda item: item[1])
            le.classes_ = np.array([item[0] for item in sorted_items])
            label_encoders[col] = le
        else:
            logger.warning(f"Used LabelEncoder for target '{col}'.")
            le = LabelEncoder()
            targets_encoded[col] = le.fit_transform(full_metadata_df[col])
            label_encoders[col] = le
            
        if targets_encoded[col].isnull().any():
            nan_indices = targets_encoded[targets_encoded[col].isnull()].index.tolist()
            logger.error(f"NaN values found in encoded target '{col}' for samples: {nan_indices}. "
                         f"Original values: {full_metadata_df.loc[nan_indices, col].unique()}. Check mapping.")
            raise ValueError(f"Encoding failed for target '{col}'.")
    
    logger.info("Target encoding complete.")

    # Train/Val/Test Split (Stratified)
    try:
        full_metadata_df['stratify_key'] = full_metadata_df[target_cols[0]].astype(str)
        for col in target_cols[1:]:
            full_metadata_df['stratify_key'] += '_' + full_metadata_df[col].astype(str)
    except KeyError as e:
        logger.error(f"Stratification key column missing: {e}")
        raise

    indices = full_metadata_df.index
    stratify_values = full_metadata_df['stratify_key']
    if stratify_values.isnull().any():
        logger.warning("NaNs found in stratification key.")

    try:
        train_idx, temp_idx = train_test_split(
            indices, 
            test_size=(config['VAL_SIZE'] + config['TEST_SIZE']), 
            random_state=config['RANDOM_SEED'], 
            stratify=stratify_values
        )
    except ValueError:
        logger.warning("Stratification failed for train/temp split. Proceeding without stratification.")
        train_idx, temp_idx = train_test_split(
            indices, 
            test_size=(config['VAL_SIZE'] + config['TEST_SIZE']), 
            random_state=config['RANDOM_SEED']
        )
        
    relative_test_size = config['TEST_SIZE'] / (config['VAL_SIZE'] + config['TEST_SIZE'])
    temp_stratify_values = stratify_values.loc[temp_idx]
    
    try:
        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=relative_test_size, 
            random_state=config['RANDOM_SEED'], 
            stratify=temp_stratify_values
        )
    except ValueError:
        logger.warning("Stratification failed for val/test split. Proceeding without stratification.")
        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=relative_test_size, 
            random_state=config['RANDOM_SEED']
        )

    # Split feature and target datasets
    X_train_spec = features_spectral_df.loc[train_idx]
    X_val_spec = features_spectral_df.loc[val_idx]
    X_test_spec = features_spectral_df.loc[test_idx]
    
    X_train_metab = features_metabolite_df.loc[train_idx]
    X_val_metab = features_metabolite_df.loc[val_idx]
    X_test_metab = features_metabolite_df.loc[test_idx]
    
    y_train = targets_encoded.loc[train_idx]
    y_val = targets_encoded.loc[val_idx]
    y_test = targets_encoded.loc[test_idx]
    
    logger.info(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Get Test Metadata Subset
    test_metadata_df = full_metadata_df.loc[test_idx].copy()
    test_metadata_df = test_metadata_df.drop(columns=['stratify_key'], errors='ignore')
    logger.info(f"Extracted test metadata subset with shape: {test_metadata_df.shape}")

    # Convert potentially numeric columns before returning
    logger.info("Converting key metadata columns to numeric where appropriate...")
    numeric_cols_to_try = ['Entry', 'Treatment', 'Replication', 'Day']
    for col in numeric_cols_to_try:
        if col in test_metadata_df.columns:
            original_sum_possible = False
            original_sum = 0
            original_dtype = test_metadata_df[col].dtype

            # Check original sum only if it's already numeric
            if pd.api.types.is_numeric_dtype(original_dtype):
                try:
                    original_sum = test_metadata_df[col].sum()
                    original_sum_possible = True
                except TypeError:
                    logger.debug(f"Could not calculate original sum for numeric column '{col}' (dtype: {original_dtype}).")

            try:
                # Use errors='coerce'. If it fails, NaNs are introduced.
                converted_col = pd.to_numeric(test_metadata_df[col], errors='coerce')
                nans_introduced = converted_col.isnull().any() and not test_metadata_df[col].isnull().any()

                # If NaNs were introduced, or if original was int, convert to float for HDF5 compatibility
                if nans_introduced or pd.api.types.is_integer_dtype(original_dtype):
                    # Check if the converted column is actually numeric before attempting astype
                    if pd.api.types.is_numeric_dtype(converted_col.dtype):
                        if not pd.api.types.is_float_dtype(converted_col.dtype):
                            logger.info(f"Converting column '{col}' to float64 due to NaNs or original int type.")
                            test_metadata_df[col] = converted_col.astype(np.float64)
                        else:
                            # Already float, assign the converted column (which might have new NaNs)
                            test_metadata_df[col] = converted_col
                elif pd.api.types.is_numeric_dtype(converted_col.dtype):
                    # Conversion successful, no new NaNs, wasn't originally int -> assign converted
                    test_metadata_df[col] = converted_col

                # Log if sum changed significantly (potential issue indicator)
                # Check only if the column *is now* numeric and original sum was possible
                if pd.api.types.is_numeric_dtype(test_metadata_df[col].dtype) and original_sum_possible:
                    try:
                        new_sum = test_metadata_df[col].sum()
                        # Use a tolerance for float comparison, handle NaNs
                        if not np.isclose(original_sum, new_sum, rtol=1e-5, atol=1e-8, equal_nan=True):
                            logger.warning(f"Sum for column '{col}' changed after numeric conversion ({original_sum} -> {new_sum}). Check data.")
                    except TypeError:
                        logger.debug(f"Could not calculate new sum for column '{col}' after potential conversion.")

            except Exception as e_conv:
                logger.error(f"Failed to process column '{col}' for numeric conversion: {e_conv}. Leaving as is.")

    # Feature Scaling
    scaler_spec = StandardScaler()
    X_train_spec_scaled = scaler_spec.fit_transform(X_train_spec)
    X_val_spec_scaled = scaler_spec.transform(X_val_spec)
    X_test_spec_scaled = scaler_spec.transform(X_test_spec)
    
    scaler_metab = StandardScaler()
    X_train_metab_scaled = scaler_metab.fit_transform(X_train_metab)
    X_val_metab_scaled = scaler_metab.transform(X_val_metab)
    X_test_metab_scaled = scaler_metab.transform(X_test_metab)
    
    if np.isnan(X_train_spec_scaled).any() or np.isinf(X_train_spec_scaled).any():
        raise ValueError("Bad values in scaled spectral data")
    if np.isnan(X_train_metab_scaled).any() or np.isinf(X_train_metab_scaled).any():
        raise ValueError("Bad values in scaled metabolite data")
        
    # Convert scaled arrays back to dataframes with original indices and column names
    X_train_spec_scaled_df = pd.DataFrame(X_train_spec_scaled, index=train_idx, columns=spectral_feature_names)
    X_val_spec_scaled_df = pd.DataFrame(X_val_spec_scaled, index=val_idx, columns=spectral_feature_names)
    X_test_spec_scaled_df = pd.DataFrame(X_test_spec_scaled, index=test_idx, columns=spectral_feature_names)
    
    X_train_metab_scaled_df = pd.DataFrame(X_train_metab_scaled, index=train_idx, columns=metabolite_feature_names)
    X_val_metab_scaled_df = pd.DataFrame(X_val_metab_scaled, index=val_idx, columns=metabolite_feature_names)
    X_test_metab_scaled_df = pd.DataFrame(X_test_metab_scaled, index=test_idx, columns=metabolite_feature_names)
    
    scalers = {'spectral': scaler_spec, 'metabolite': scaler_metab}

    # Create datasets and dataloaders
    train_dataset = PlantOmicsDataset(X_train_spec_scaled_df, X_train_metab_scaled_df, y_train)
    val_dataset = PlantOmicsDataset(X_val_spec_scaled_df, X_val_metab_scaled_df, y_val)
    test_dataset = PlantOmicsDataset(X_test_spec_scaled_df, X_test_metab_scaled_df, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, 
                              num_workers=config['NUM_WORKERS'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, 
                            num_workers=config['NUM_WORKERS'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, 
                             num_workers=config['NUM_WORKERS'], pin_memory=True)
    
    logger.info("DataLoaders created.")
    logger.info("--- Data Loading & Preprocessing Finished ---")

    return (train_loader, val_loader, test_loader,
            spectral_feature_names, metabolite_feature_names,
            train_idx.tolist(), val_idx.tolist(), test_idx.tolist(),
            scalers, label_encoders,
            X_train_spec_scaled_df, X_val_spec_scaled_df, X_test_spec_scaled_df,
            X_train_metab_scaled_df, X_val_metab_scaled_df, X_test_metab_scaled_df,
            y_train, y_val, y_test,
            test_metadata_df)


# ===== MODEL DEFINITION =====
class CrossAttentionLayer(nn.Module):
    """Implements cross-attention between two modalities (sequences)."""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn_1_to_2 = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
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
        self.norm3 = nn.LayerNorm(hidden_dim)  # Shared norm after FFN

    def forward(self, x1, x2):
        """
        Process inputs through cross-attention in both directions.
        
        Args:
            x1: Features from modality 1 (batch, seq_len_1, hidden_dim)
            x2: Features from modality 2 (batch, seq_len_2, hidden_dim)
            
        Returns:
            out1, out2: Updated features for each modality
            attn_weights_1_to_2: Attention weights (batch, heads, seq_len_1, seq_len_2)
            attn_weights_2_to_1: Attention weights (batch, heads, seq_len_2, seq_len_1)
        """
        # Use average_attn_weights=False to get per-head weights
        attn_output_1, attn_weights_1_to_2 = self.cross_attn_1_to_2(
            query=x1, key=x2, value=x2,
            average_attn_weights=False
        )
        x1 = self.norm1(x1 + self.dropout(attn_output_1))

        attn_output_2, attn_weights_2_to_1 = self.cross_attn_2_to_1(
            query=x2, key=x1, value=x1,
            average_attn_weights=False
        )
        x2 = self.norm2(x2 + self.dropout(attn_output_2))

        # Feed-forward networks and residual connections
        ffn_output1 = self.ffn(x1)
        x1 = self.norm3(x1 + self.dropout(ffn_output1))
        
        ffn_output2 = self.ffn(x2)
        x2 = self.norm3(x2 + self.dropout(ffn_output2))

        return x1, x2, attn_weights_1_to_2, attn_weights_2_to_1


class SimplifiedTransformer(nn.Module):
    """Simplified Transformer model with feature-level cross-attention."""
    def __init__(self, spectral_dim, metabolite_dim, hidden_dim, num_heads, 
                 num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature embeddings (convert each feature to hidden dimension)
        self.spectral_feature_embedding = nn.Linear(1, hidden_dim)
        self.metabolite_feature_embedding = nn.Linear(1, hidden_dim)
        
        # Positional encodings (learnable)
        self.pos_encoding_spec = nn.Parameter(torch.randn(1, spectral_dim, hidden_dim) * 0.02)
        self.pos_encoding_metab = nn.Parameter(torch.randn(1, metabolite_dim, hidden_dim) * 0.02)
        
        # Layer normalization and dropout
        self.embedding_norm_spec = nn.LayerNorm(hidden_dim)
        self.embedding_norm_metab = nn.LayerNorm(hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Multi-task classification heads
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
        
        # Storage for attention weights
        self.attention_weights = {}

    def forward(self, spectral, metabolite):
        """
        Forward pass through the model.
        
        Args:
            spectral: Spectral features (batch, spectral_dim)
            metabolite: Metabolite features (batch, metabolite_dim)
            
        Returns:
            Dictionary of outputs for each task
        """
        # Reshape inputs for per-feature embeddings
        spec_reshaped = spectral.unsqueeze(-1)
        metab_reshaped = metabolite.unsqueeze(-1)
        
        # Embed each feature
        spec_emb = self.spectral_feature_embedding(spec_reshaped)
        metab_emb = self.metabolite_feature_embedding(metab_reshaped)
        
        # Add positional encodings
        spec_emb = spec_emb + self.pos_encoding_spec
        metab_emb = metab_emb + self.pos_encoding_metab
        
        # Apply normalization and dropout
        spec_emb = self.embedding_dropout(self.embedding_norm_spec(spec_emb))
        metab_emb = self.embedding_dropout(self.embedding_norm_metab(metab_emb))

        # Reset attention weights storage
        self.attention_weights = {}

        # Process through cross-attention layers
        for i, layer in enumerate(self.cross_attention_layers):
            spec_emb, metab_emb, attn_1_to_2, attn_2_to_1 = layer(spec_emb, metab_emb)
            if i == len(self.cross_attention_layers) - 1:
                # Store attention weights from final layer
                self.attention_weights = {'1_to_2': attn_1_to_2, '2_to_1': attn_2_to_1}

        # Global pooling across features
        spec_pooled = spec_emb.mean(dim=1)
        metab_pooled = metab_emb.mean(dim=1)
        
        # Concatenate pooled representations
        combined_pooled = torch.cat([spec_pooled, metab_pooled], dim=1)
        
        # Apply task-specific heads
        outputs = {}
        for task_name, head in self.output_heads.items():
            outputs[task_name] = head(combined_pooled)
            
        return outputs


# ===== TRAINING FUNCTIONS =====
def train_one_epoch(model, dataloader, optimizer, criterion, device, target_cols):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: The optimizer
        criterion: Loss function
        device: Device to train on (CPU/GPU)
        target_cols: List of target column names
        
    Returns:
        avg_loss: Average loss over the epoch
        epoch_metrics: Dictionary of metrics for each task
    """
    model.train()
    total_loss = 0.0
    all_preds = {task: [] for task in target_cols}
    all_targets = {task: [] for task in target_cols}
    
    for batch in dataloader:
        spectral = batch['spectral'].to(device)
        metab = batch['metabolite'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        outputs = model(spectral, metab)
        
        loss = 0
        for task_idx, task_name in enumerate(target_cols):
            loss += criterion(outputs[task_name], targets[:, task_idx])
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
        with torch.no_grad():
            for task_idx, task_name in enumerate(target_cols):
                preds = torch.argmax(outputs[task_name], dim=1)
                all_preds[task_name].extend(preds.cpu().numpy())
                all_targets[task_name].extend(targets[:, task_idx].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = {}
    
    for task_name in target_cols:
        y_true = all_targets[task_name]
        y_pred = all_preds[task_name]
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        epoch_metrics[task_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    return avg_loss, epoch_metrics


# ===== EVALUATION FUNCTION =====
def evaluate(model, dataloader, criterion, device, target_cols):
    """
    Evaluate the model and collect attention weights.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to evaluate on (CPU/GPU)
        target_cols: List of target column names
        
    Returns:
        avg_loss: Average loss over the dataset
        eval_metrics: Dictionary of metrics for each task
        preds_df: DataFrame of predictions
        targets_df: DataFrame of true targets
        final_attention: Dictionary of attention weights
    """
    model.eval()
    total_loss = 0.0
    all_preds = {task: [] for task in target_cols}
    all_targets = {task: [] for task in target_cols}
    all_sample_ids = []
    raw_attention_batches = {'1_to_2': [], '2_to_1': []}  # Store weights batch by batch on CPU

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            spectral = batch['spectral'].to(device)
            metab = batch['metabolite'].to(device)
            targets = batch['targets'].to(device)
            sample_ids = batch['sample_id']
            
            outputs = model(spectral, metab)
            
            loss = 0
            for task_idx, task_name in enumerate(target_cols):
                loss += criterion(outputs[task_name], targets[:, task_idx])
            
            total_loss += loss.item()
            
            for task_idx, task_name in enumerate(target_cols):
                preds = torch.argmax(outputs[task_name], dim=1)
                all_preds[task_name].extend(preds.cpu().numpy())
                all_targets[task_name].extend(targets[:, task_idx].cpu().numpy())
                
            all_sample_ids.extend(sample_ids)

            # Collect attention weights if available
            if hasattr(model, 'attention_weights') and model.attention_weights and '1_to_2' in model.attention_weights:
                attn_1_to_2_batch = model.attention_weights['1_to_2'].detach().cpu()
                logger.info(f"Evaluate Batch {batch_idx}: Raw S->M attention shape: {attn_1_to_2_batch.shape}")
                raw_attention_batches['1_to_2'].append(attn_1_to_2_batch)

                if '2_to_1' in model.attention_weights:
                    attn_2_to_1_batch = model.attention_weights['2_to_1'].detach().cpu()
                    logger.info(f"Evaluate Batch {batch_idx}: Raw M->S attention shape: {attn_2_to_1_batch.shape}")
                    raw_attention_batches['2_to_1'].append(attn_2_to_1_batch)

    # Calculate average loss and evaluation metrics
    avg_loss = total_loss / len(dataloader)
    eval_metrics = {}
    
    for task_name in target_cols:
        y_true = all_targets[task_name]
        y_pred = all_preds[task_name]
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        eval_metrics[task_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Concatenate attention weights from batches
    final_attention = None
    if raw_attention_batches['1_to_2']:
        logger.info(f"Evaluate: Attempting to concatenate {len(raw_attention_batches['1_to_2'])} S->M attention batches.")
        try:
            # Concatenate along batch dim (dim=0)
            final_attn_1_to_2 = torch.cat(raw_attention_batches['1_to_2'], dim=0)
            
            # Check shape after concatenation
            if len(final_attn_1_to_2.shape) != 4:
                logger.error(f"Evaluate: Concatenated S->M attention has wrong shape: {final_attn_1_to_2.shape}. Expected 4D.")
                final_attention = None
            else:
                final_attn_2_to_1 = None
                if raw_attention_batches['2_to_1']:
                    logger.info(f"Evaluate: Attempting to concatenate {len(raw_attention_batches['2_to_1'])} M->S attention batches.")
                    final_attn_2_to_1 = torch.cat(raw_attention_batches['2_to_1'], dim=0)
                    if len(final_attn_2_to_1.shape) != 4:
                        logger.error(f"Evaluate: Concatenated M->S attention has wrong shape: {final_attn_2_to_1.shape}. Expected 4D.")
                        final_attn_2_to_1 = None

                final_attention = {'1_to_2': final_attn_1_to_2}
                if final_attn_2_to_1 is not None:
                    final_attention['2_to_1'] = final_attn_2_to_1

                log_s2m_shape = final_attn_1_to_2.shape
                log_m2s_shape = final_attn_2_to_1.shape if final_attn_2_to_1 is not None else "N/A"
                logger.info(f"Concatenated final attention tensors. S->M shape: {log_s2m_shape}, M->S shape: {log_m2s_shape}")

        except Exception as e_cat:
            logger.error(f"Could not concatenate attention batches: {e_cat}. Attention analysis will be skipped.", exc_info=True)
            final_attention = None
    else:
        logger.warning("Evaluate: No attention weights collected in batches.")
        final_attention = None

    # Create dataframes from predictions and targets
    preds_df = pd.DataFrame(all_preds, index=all_sample_ids)
    targets_df = pd.DataFrame(all_targets, index=all_sample_ids)
    
    # Ensure index name is consistent
    preds_df.index.name = 'Row_names'
    targets_df.index.name = 'Row_names'
    
    return avg_loss, eval_metrics, preds_df, targets_df, final_attention


# ===== ANALYSIS FUNCTIONS =====
def analyze_attention_feature_level(attention_weights, spectral_feature_names, metabolite_feature_names,
                                   output_dir, pairing, top_k_pairs=200):
    """
    Analyzes feature-level attention weights, saves summaries to CSV,
    and saves tensors and feature names to HDF5.
    
    Args:
        attention_weights: Dictionary of attention weight tensors
        spectral_feature_names: List of spectral feature names
        metabolite_feature_names: List of metabolite feature names
        output_dir: Directory to save outputs
        pairing: Analysis pairing (e.g., "Leaf" or "Root")
        top_k_pairs: Number of top attention pairs to save
        
    Returns:
        cross_modal_pairs_df: DataFrame of cross-modal feature pairs
        feature_importance_df: DataFrame of feature importance scores
    """
    logger.info("--- Starting Feature-Level Attention Analysis (HDF5 Tensors/Features + Summary CSVs) ---")
    
    # Input validation
    if attention_weights is None or '1_to_2' not in attention_weights or \
       not isinstance(attention_weights['1_to_2'], torch.Tensor) or \
       attention_weights['1_to_2'].numel() == 0:
        logger.warning("Attention weights invalid/empty. Skipping HDF5/CSV summary saving.")
        return None, None

    cross_modal_pairs_df, feature_importance_df = None, None  # Initialize

    try:
        # Get the 4D tensor from evaluate()
        attn_s2m_raw = attention_weights['1_to_2']  # Shape: (N, H, N_spec, N_metab)

        # Dimension validation
        n_samples, n_heads, n_spec, n_metab = attn_s2m_raw.shape

        if n_spec != len(spectral_feature_names):
            logger.error("Spectral dimension mismatch.")
            return None, None
        if n_metab != len(metabolite_feature_names):
            logger.error("Metabolite dimension mismatch.")
            return None, None
            
        logger.info(f"Validated attention tensor shape for HDF5: {(n_samples, n_heads, n_spec, n_metab)}")

        # Calculate summaries (average over heads and samples)
        logger.info("Calculating attention summaries (pairs, importance)...")
        attn_s2m_avg_heads_samples = attn_s2m_raw.mean(dim=(0, 1)).numpy()  # Avg over samples & heads

        # 1. Cross-Modal Pairs
        pairs = []
        for i in range(n_spec):
            for j in range(n_metab):
                pairs.append({
                    'Spectral_Feature': spectral_feature_names[i],
                    'Metabolite_Feature': metabolite_feature_names[j],
                    'Mean_Attention_S2M_AvgHeadsSamples': attn_s2m_avg_heads_samples[i, j]
                })
        cross_modal_pairs_df = pd.DataFrame(pairs).sort_values('Mean_Attention_S2M_AvgHeadsSamples', ascending=False)
        logger.info(f"Calculated cross-modal pairs summary (Top {top_k_pairs} pairs).")

        # 2. Feature Importance
        spec_importance_mag = attn_s2m_avg_heads_samples.sum(axis=1)
        metab_importance_mag = attn_s2m_avg_heads_samples.sum(axis=0)
        importance_list = []
        for i, name in enumerate(spectral_feature_names):
            importance_list.append({
                'Feature': name,
                'View': 'Spectral',
                'Importance_Magnitude_S2M_AvgHeadsSamples': spec_importance_mag[i]
            })
        for j, name in enumerate(metabolite_feature_names):
            importance_list.append({
                'Feature': name,
                'View': 'Metabolite',
                'Importance_Magnitude_S2M_AvgHeadsSamples': metab_importance_mag[j]
            })
        feature_importance_df = pd.DataFrame(importance_list).sort_values('Importance_Magnitude_S2M_AvgHeadsSamples', ascending=False)
        logger.info(f"Calculated feature importance summary ({len(feature_importance_df)} features).")

        # Save summaries to CSV files
        pairs_outfile = os.path.join(output_dir, f"transformer_cross_modal_pairs_{pairing}.csv")
        importance_outfile = os.path.join(output_dir, f"transformer_feature_importance_{pairing}.csv")
        try:
            cross_modal_pairs_df.head(top_k_pairs).to_csv(pairs_outfile, index=False)
            logger.info(f"Saved cross-modal pairs summary CSV to: {pairs_outfile}")
            if not os.path.exists(pairs_outfile):
                logger.error(f"CSV file not found after saving: {pairs_outfile}")
        except Exception as e_csv:
            logger.error(f"Failed to save pairs CSV: {e_csv}", exc_info=True)
            
        try:
            feature_importance_df.to_csv(importance_outfile, index=False)
            logger.info(f"Saved feature importance summary CSV to: {importance_outfile}")
            if not os.path.exists(importance_outfile):
                logger.error(f"CSV file not found after saving: {importance_outfile}")
        except Exception as e_csv:
            logger.error(f"Failed to save importance CSV: {e_csv}", exc_info=True)

        # Save tensors and features to HDF5
        logger.info("Saving raw tensors and features to HDF5...")
        attention_output_path = os.path.join(output_dir, f"raw_attention_data_{pairing}.h5")
        try:
            with h5py.File(attention_output_path, 'w') as f:
                logger.info(f"Opened HDF5 file for writing: {attention_output_path}")

                # 1. Save attention tensors
                logger.info("Saving attention tensors...")
                try:
                    attn_s2m_np = attn_s2m_raw.numpy()
                    f.create_dataset('attention_spec_to_metab', data=attn_s2m_np, compression="gzip")
                    logger.info(f"Saved attention_spec_to_metab (Shape: {attn_s2m_np.shape}).")
                except Exception as e:
                    logger.error(f"Failed to save S->M attention: {e}", exc_info=True)

                # Save M->S attention if available
                if '2_to_1' in attention_weights and attention_weights['2_to_1'] is not None:
                    try:
                        # Get the M->S tensor
                        attn_m2s_raw = attention_weights['2_to_1']

                        # Validate M->S tensor shape
                        n_samples_m2s, n_heads_m2s, n_metab_m2s, n_spec_m2s = attn_m2s_raw.shape
                        valid_m2s_shape = True
                        if n_samples_m2s != n_samples:
                            logger.error(f"M->S Sample count ({n_samples_m2s}) mismatch with S->M ({n_samples}). Cannot save M->S.")
                            valid_m2s_shape = False
                        if n_metab_m2s != len(metabolite_feature_names):
                            logger.error(f"M->S Metabolite dim mismatch: Tensor={n_metab_m2s}, Features={len(metabolite_feature_names)}. Cannot save M->S.")
                            valid_m2s_shape = False
                        if n_spec_m2s != len(spectral_feature_names):
                            logger.error(f"M->S Spectral dim mismatch: Tensor={n_spec_m2s}, Features={len(spectral_feature_names)}. Cannot save M->S.")
                            valid_m2s_shape = False

                        if valid_m2s_shape:
                            attn_m2s_np = attn_m2s_raw.numpy()
                            f.create_dataset('attention_metab_to_spec', data=attn_m2s_np, compression="gzip")
                            logger.info(f"Saved attention_metab_to_spec (Shape: {attn_m2s_np.shape}).")
                        else:
                            logger.error("Skipping save of attention_metab_to_spec due to shape validation errors.")

                    except Exception as e:
                        logger.error(f"Failed to save M->S attention ('attention_metab_to_spec'): {e}", exc_info=True)
                else:
                    logger.warning("'2_to_1' (M->S) attention weights not found or None. Skipping save of 'attention_metab_to_spec'.")

                # 2. Save feature names
                logger.info("Saving feature names...")
                try:
                    f.create_dataset('spectral_feature_names', 
                                     data=np.array(spectral_feature_names, dtype=h5py.string_dtype(encoding='utf-8')))
                    f.create_dataset('metabolite_feature_names', 
                                     data=np.array(metabolite_feature_names, dtype=h5py.string_dtype(encoding='utf-8')))
                    logger.info("Saved feature name datasets.")
                except Exception as e:
                    logger.error(f"Failed to save feature names: {e}", exc_info=True)

            # Final verification
            logger.info(f"Successfully completed writing to HDF5 file: {attention_output_path}")
            if os.path.exists(attention_output_path):
                logger.info(f"Verified HDF5 file exists: {attention_output_path}")
            else:
                logger.error(f"HDF5 saving reported success, but file not found: {attention_output_path}")

        except OSError as e_os:
            logger.error(f"Failed to save data to HDF5 (OSError): {e_os}", exc_info=True)
        except Exception as e_h5:
            logger.error(f"Failed to save data to HDF5 (Other Error): {e_h5}", exc_info=True)

    except Exception as e_outer:
        logger.error(f"Error during overall feature-level attention analysis: {e_outer}", exc_info=True)
        return None, None

    logger.info("--- Feature-Level Attention Analysis Finished ---")
    # Return the calculated summary DataFrames
    return cross_modal_pairs_df, feature_importance_df


# ===== VISUALIZATION FUNCTIONS (REMOVED / COMMENTED OUT) =====
# def plot_attention_heatmap(...):
#     logger.info("--- Visualization skipped ---")
# def plot_final_visualizations(...):
#     logger.info("--- Visualization skipped ---")

# ===== BASELINE MODELS =====
def run_baseline_models(X_train_spec, X_train_metab, y_train, X_test_spec, X_test_metab, 
                        y_test, target_cols, output_dir, pairing, random_seed):
    """
    Train and evaluate baseline models for comparison.
    
    Args:
        X_train_spec: Training spectral features
        X_train_metab: Training metabolite features
        y_train: Training targets
        X_test_spec: Test spectral features
        X_test_metab: Test metabolite features
        y_test: Test targets
        target_cols: List of target column names
        output_dir: Directory to save results
        pairing: Analysis pairing (e.g., "Leaf" or "Root")
        random_seed: Random seed for reproducibility
        
    Returns:
        baseline_df: DataFrame with baseline model results
    """
    logger.info("--- Running Baseline Models ---")
    baseline_results = []
    
    # Combine features for baseline models
    X_train_combined = pd.concat([X_train_spec, X_train_metab], axis=1)
    X_test_combined = pd.concat([X_test_spec, X_test_metab], axis=1)
    logger.info(f"Combined feature shape for baselines: Train={X_train_combined.shape}, Test={X_test_combined.shape}")
    
    try:
        # Random Forest
        logger.info("Training Random Forest...")
        rf_preds_dict = {}
        for task_name in target_cols:
            rf_task = RandomForestClassifier(
                n_estimators=200, 
                random_state=random_seed,
                n_jobs=-1,
                max_depth=20,
                min_samples_leaf=5
            )
            rf_task.fit(X_train_combined, y_train[task_name])
            rf_preds_dict[task_name] = rf_task.predict(X_test_combined)
            
        logger.info("Evaluating Random Forest...")
        for task_name in target_cols:
            task_target = y_test[task_name]
            task_preds = rf_preds_dict[task_name]
            accuracy = accuracy_score(task_target, task_preds)
            f1 = f1_score(task_target, task_preds, average='macro', zero_division=0)
            baseline_results.append({
                'Model': 'RandomForest',
                'Task': task_name,
                'Metric': 'Accuracy',
                'Score': accuracy
            })
            baseline_results.append({
                'Model': 'RandomForest',
                'Task': task_name,
                'Metric': 'F1_Macro',
                'Score': f1
            })
    except Exception as e:
        logger.error(f"Error running RandomForest baseline: {e}")
        traceback.print_exc()
        
    try:
        # KNN
        logger.info("Training K-Nearest Neighbors (KNN)...")
        knn_preds_dict = {}
        n_neighbors_val = 5
        for task_name in target_cols:
            knn_task = KNeighborsClassifier(
                n_neighbors=n_neighbors_val,
                n_jobs=-1,
                weights='distance'
            )
            knn_task.fit(X_train_combined, y_train[task_name])
            knn_preds_dict[task_name] = knn_task.predict(X_test_combined)
            
        logger.info("Evaluating K-Nearest Neighbors...")
        for task_name in target_cols:
            task_target = y_test[task_name]
            task_preds = knn_preds_dict[task_name]
            accuracy = accuracy_score(task_target, task_preds)
            f1 = f1_score(task_target, task_preds, average='macro', zero_division=0)
            baseline_results.append({
                'Model': f'KNN (k={n_neighbors_val})',
                'Task': task_name,
                'Metric': 'Accuracy',
                'Score': accuracy
            })
            baseline_results.append({
                'Model': f'KNN (k={n_neighbors_val})',
                'Task': task_name,
                'Metric': 'F1_Macro',
                'Score': f1
            })
    except Exception as e:
        logger.error(f"Error running KNN baseline: {e}")
        traceback.print_exc()
        
    # Save baseline results
    if baseline_results:
        baseline_df = pd.DataFrame(baseline_results)
        outfile = os.path.join(output_dir, f"transformer_baseline_comparison_{pairing}.csv")
        baseline_df.to_csv(outfile, index=False)
        logger.info(f"Baseline comparison results saved to {outfile}")
    else:
        logger.warning("No baseline results generated.")
        baseline_df = None
        
    logger.info("--- Baseline Models Finished ---")
    return baseline_df


# ===== MAIN EXECUTION =====
def main():
    """Main execution function."""
    start_time = time.time()
    logger.info(f"--- Starting Main Execution ({SCRIPT_NAME} v{VERSION}) ---")
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        # Load and preprocess data
        (train_loader, val_loader, test_loader,
         spectral_feature_names, metabolite_feature_names,
         train_ids, val_ids, test_ids,
         scalers, label_encoders,
         X_train_spec_scaled_df, X_val_spec_scaled_df, X_test_spec_scaled_df,
         X_train_metab_scaled_df, X_val_metab_scaled_df, X_test_metab_scaled_df,
         y_train_df, y_val_df, y_test_df,
         test_metadata_df
         ) = load_and_preprocess_data(config=globals())
    except Exception as e:
        logger.error(f"Failed during data loading. Exiting. Error: {e}", exc_info=True)
        return

    # Initialize model, loss, and optimizer
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
    
    logger.info(f"Model initialized (Feature Dims: Spec={spectral_dim}, Metab={metabolite_dim})")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training phase
    logger.info("--- Starting Training ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_start_time = time.time()
    
    pairing_checkpoint_dir = os.path.join(CHECKPOINT_SUBDIR, ANALYSIS_PAIRING)
    os.makedirs(pairing_checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(pairing_checkpoint_dir, f"best_model_{SCRIPT_NAME}_{ANALYSIS_PAIRING}.pth")

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        
        # Train for one epoch
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, TARGET_COLS
        )
        
        # Validate
        val_loss, val_metrics, _, _, val_attention = evaluate(
            model, val_loader, criterion, DEVICE, TARGET_COLS
        )
        
        # Log metrics
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        for task in TARGET_COLS:
            logger.info(f"  {task}: Train Acc={train_metrics[task]['accuracy']:.4f}, "
                       f"Val Acc={val_metrics[task]['accuracy']:.4f} | "
                       f"Train F1={train_metrics[task]['f1']:.4f}, "
                       f"Val F1={val_metrics[task]['f1']:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Val loss improved. Saved best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            logger.info(f"Val loss did not improve for {epochs_no_improve} epochs.")
            
        # Early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break
            
    training_duration = time.time() - training_start_time
    logger.info(f"--- Training Finished --- Duration: {training_duration:.2f} seconds")

    # Evaluation phase
    logger.info("--- Evaluating on Test Set using Best Model ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}")
    else:
        logger.warning("Best model checkpoint not found. Evaluating with the last state.")
        
    # Evaluate on test set
    test_loss, test_metrics, test_preds_df, test_targets_df, test_attention = evaluate(
        model, test_loader, criterion, DEVICE, TARGET_COLS
    )
    
    # Log test metrics
    logger.info(f"Test Loss: {test_loss:.4f}")
    performance_data = []
    for task in TARGET_COLS:
        logger.info(f"  Test Metrics for {task}: "
                  f"Acc={test_metrics[task]['accuracy']:.4f}, "
                  f"F1={test_metrics[task]['f1']:.4f}, "
                  f"Prec={test_metrics[task]['precision']:.4f}, "
                  f"Rec={test_metrics[task]['recall']:.4f}")
        
        performance_data.append({
            'Task': task, 'Metric': 'Accuracy', 'Score': test_metrics[task]['accuracy']
        })
        performance_data.append({
            'Task': task, 'Metric': 'F1_Macro', 'Score': test_metrics[task]['f1']
        })
        performance_data.append({
            'Task': task, 'Metric': 'Precision_Macro', 'Score': test_metrics[task]['precision']
        })
        performance_data.append({
            'Task': task, 'Metric': 'Recall_Macro', 'Score': test_metrics[task]['recall']
        })
        
    # Save performance metrics
    performance_df = pd.DataFrame(performance_data)
    perf_outfile = os.path.join(RESULTS_SUBDIR, f"transformer_class_performance_{ANALYSIS_PAIRING}.csv")
    performance_df.to_csv(perf_outfile, index=False)
    logger.info(f"Transformer test performance metrics saved to {perf_outfile}")

    # Prepare data for saving
    logger.info("Preparing test predictions, metadata, and attention for saving...")
    test_preds_df_renamed = test_preds_df.rename(columns={col: f"{col}_pred_encoded" for col in TARGET_COLS})
    test_targets_df_renamed = test_targets_df.rename(columns={col: f"{col}_true_encoded" for col in TARGET_COLS})

    # Align metadata with predictions/targets
    test_metadata_aligned = None
    if isinstance(test_metadata_df, pd.DataFrame) and not test_metadata_df.empty:
        try:
            # Ensure index names match if necessary
            if test_metadata_df.index.name != test_preds_df_renamed.index.name:
                logger.warning(f"Renaming test_metadata_df index to match predictions: '{test_preds_df_renamed.index.name}'")
                test_metadata_df = test_metadata_df.rename_axis(test_preds_df_renamed.index.name)
                
            # Align using the index (Row_names)
            test_metadata_aligned = test_metadata_df.reindex(test_preds_df_renamed.index)
            if test_metadata_aligned.isnull().any(axis=None):
                logger.warning("NaNs found in aligned metadata. Check if all test samples had metadata.")
                
            logger.info(f"Successfully aligned test metadata ({test_metadata_aligned.shape}) with predictions/targets.")
        except Exception as e_reidx:
            logger.error(f"Failed to align metadata with predictions: {e_reidx}. Cannot proceed with saving combined results or validated attention.", exc_info=True)
            test_metadata_aligned = None
    else:
        logger.error("Test metadata DataFrame is invalid or empty. Cannot proceed.")
        test_metadata_aligned = None

    # Validate alignment before saving
    logger.info("--- Validating Alignment Before Saving ---")
    validation_passed = False
    if test_attention is not None and '1_to_2' in test_attention and \
       isinstance(test_attention['1_to_2'], torch.Tensor) and \
       test_metadata_aligned is not None:

        n_samples_tensor = test_attention['1_to_2'].shape[0]
        n_samples_metadata = len(test_metadata_aligned)

        if n_samples_tensor == n_samples_metadata:
            logger.info(f"Validation PASSED: Tensor samples ({n_samples_tensor}) match metadata rows ({n_samples_metadata}).")
            validation_passed = True
        else:
            logger.error(f"VALIDATION FAILED: Tensor samples ({n_samples_tensor}) DO NOT match metadata rows ({n_samples_metadata})!")
            logger.error("Cannot guarantee alignment between HDF5 tensors and metadata file. Aborting further saving steps that depend on this alignment.")
    else:
        logger.error("VALIDATION SKIPPED: Attention tensors or aligned metadata are missing/invalid.")
        if test_attention is None or '1_to_2' not in test_attention:
            logger.error("Reason: Attention data missing or invalid.")
        if test_metadata_aligned is None:
            logger.error("Reason: Aligned metadata is missing or invalid (check alignment process).")

    # Save combined predictions + metadata CSV
    if test_metadata_aligned is not None:
        logger.info("Saving test predictions and true labels with metadata...")
        
        # Include the Row_names index in the CSV
        test_results_combined_df = pd.concat([test_metadata_aligned, test_preds_df_renamed, test_targets_df_renamed], axis=1)
        results_outfile = os.path.join(RESULTS_SUBDIR, f"transformer_test_predictions_metadata_{ANALYSIS_PAIRING}.csv")
        
        try:
            test_results_combined_df.to_csv(results_outfile, index=True, index_label='Row_names')
            logger.info(f"Test predictions, true labels, and metadata saved to CSV: {results_outfile}")
        except Exception as e_csv:
            logger.error(f"Failed to save predictions+metadata CSV: {e_csv}", exc_info=True)
    else:
        logger.warning("Skipping save of combined predictions+metadata CSV due to missing/unaligned metadata.")

    # Save attention data and separate metadata (only if validation passed)
    if validation_passed:
        logger.info("--- Saving Attention Tensors (HDF5) and Separate Metadata File ---")

        # Save metadata separately
        if test_metadata_aligned.index.name != 'Row_names':
            test_metadata_aligned = test_metadata_aligned.rename_axis('Row_names')

        metadata_outfile_feather = os.path.join(RESULTS_SUBDIR, f"raw_attention_metadata_{ANALYSIS_PAIRING}.feather")
        try:
            # Feather requires index to be reset or non-named for standard saving
            test_metadata_aligned.reset_index().to_feather(metadata_outfile_feather)
            logger.info(f"Test metadata saved separately to Feather file: {metadata_outfile_feather}")
        except ImportError:
            logger.warning("`pyarrow` not installed. Falling back to saving metadata as CSV.")
            metadata_outfile_csv = os.path.join(RESULTS_SUBDIR, f"raw_attention_metadata_{ANALYSIS_PAIRING}.csv")
            try:
                test_metadata_aligned.to_csv(metadata_outfile_csv, index=True, index_label='Row_names')
                logger.info(f"Test metadata saved separately to CSV file: {metadata_outfile_csv}")
            except Exception as e_csv_meta:
                logger.error(f"Failed to save metadata to CSV: {e_csv_meta}", exc_info=True)
        except Exception as e_feather:
            logger.error(f"Failed to save metadata to Feather: {e_feather}", exc_info=True)

        # Call attention analysis
        logger.info("Calling feature-level attention analysis (HDF5/Summary CSV saving)...")
        cross_modal_df, feat_importance_df = analyze_attention_feature_level(
            attention_weights=test_attention,
            spectral_feature_names=spectral_feature_names,
            metabolite_feature_names=metabolite_feature_names,
            output_dir=RESULTS_SUBDIR,
            pairing=ANALYSIS_PAIRING
        )
    else:
        logger.warning("Skipping saving of HDF5 attention data and separate metadata due to validation failure or missing data.")

    # Run baseline models for comparison
    run_baseline_models(
        X_train_spec_scaled_df, X_train_metab_scaled_df, y_train_df,
        X_test_spec_scaled_df, X_test_metab_scaled_df, y_test_df,
        TARGET_COLS, RESULTS_SUBDIR, ANALYSIS_PAIRING, RANDOM_SEED
    )

    logger.info("--- Visualization Phase Skipped ---")

    # Finish execution
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