# -*- coding: utf-8 -*-
"""
SHAP Analysis for Multi-Omic Transformer with Feature Attention

This script performs SHAP (SHapley Additive exPlanations) analysis on a trained
multi-omic transformer model that uses feature-level attention. It evaluates
feature importance across different omics types (spectral and metabolite) and
prediction tasks (genotype, treatment, day).

The script uses GradientExplainer to calculate SHAP values and generates:
- SHAP importance CSV files per task/tissue pairing
- Basic SHAP summary bar plots
- Advanced visualizations:
  - SHAP clustermaps showing top features across tasks
  - Omics contribution stacked bar plots
  - Faceted top features bar plots showing importance by task

This implementation supports separate metadata and feature files and
applies concatenated input strategy for SHAP calculation.

Outputs:
- SHAP importance CSV files per task/pairing (in SHAP_DATA_DIR).
- Basic SHAP summary bar plots per task/pairing (in SHAP_PLOT_DIR).
- Advanced SHAP clustermaps per pairing (in SHAP_PLOT_DIR).
- Advanced Omics Contribution stacked bar plots per pairing (in SHAP_PLOT_DIR).
- Advanced Faceted Top Features bar plots per pairing (in SHAP_PLOT_DIR) - like Figure 4.
"""

# Standard library imports
import os
import sys
import time
import logging
import traceback
from datetime import datetime

# Third-party imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Check for required packages
try:
    import shap
except ImportError:
    print("ERROR: SHAP library not found. Please install using 'pip install shap'")
    sys.exit(1)

try:
    import pyarrow
except ImportError:
    pyarrow = None
    print("WARNING: `pyarrow` library not found. Will not be able to read .feather metadata files. Will attempt .csv fallback.")

# ===== CONFIGURATION =====

# --- Script Info ---
SCRIPT_NAME = "run_shap_gradient_v3_model_integrated_plots_v2"
VERSION = "1.2.0"

# --- Paths ---
BASE_DIR = r"C:/Users/ms/Desktop/hyper"
MOFA_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "mofa")
TRANSFORMER_V3_OUTPUT_DIR = os.path.join(
    BASE_DIR, "output", "transformer", "v3_feature_attention"
)
SHAP_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "transformer", "shap_analysis_ggl")

# Specific subdirs within SHAP_OUTPUT_DIR
SHAP_DATA_DIR = os.path.join(SHAP_OUTPUT_DIR, "importance_data")
SHAP_PLOT_DIR = os.path.join(SHAP_OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(SHAP_OUTPUT_DIR, "logs")

# Ensure output directories exist
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
os.makedirs(SHAP_DATA_DIR, exist_ok=True)
os.makedirs(SHAP_PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Model Checkpoint Paths (v3 model) ---
MODEL_PATHS = {
    "Leaf": os.path.join(
        TRANSFORMER_V3_OUTPUT_DIR,
        r"leaf/checkpoints/Leaf/best_model_transformer_multi_omic_v3_attention_Leaf.pth"
    ),
    "Root": os.path.join(
        TRANSFORMER_V3_OUTPUT_DIR,
        r"root/checkpoints/Root/best_model_transformer_multi_omic_v3_attention_Root.pth"
    )
}

# --- Input Feature Data Paths ---
FEATURE_INPUT_FILES = {
    "Leaf": {
        "spectral": os.path.join(MOFA_OUTPUT_DIR, "transformer_input_leaf_spectral.csv"),
        "metabolite": os.path.join(MOFA_OUTPUT_DIR, "transformer_input_leaf_metabolite.csv")
    },
    "Root": {
        "spectral": os.path.join(MOFA_OUTPUT_DIR, "transformer_input_root_spectral.csv"),
        "metabolite": os.path.join(MOFA_OUTPUT_DIR, "transformer_input_root_metabolite.csv")
    }
}

# --- Input Metadata File Paths ---
METADATA_INPUT_FILES = {
    "Leaf": os.path.join(
        TRANSFORMER_V3_OUTPUT_DIR, 
        r"leaf/results/raw_attention_metadata_Leaf.feather"
    ),
    "Root": os.path.join(
        TRANSFORMER_V3_OUTPUT_DIR,
        r"root/results/raw_attention_metadata_Root.feather"
    )
}
METADATA_CSV_FALLBACK_FILES = {
    "Leaf": os.path.join(
        TRANSFORMER_V3_OUTPUT_DIR,
        r"leaf/results/raw_attention_metadata_Leaf.csv"
    ),
    "Root": os.path.join(
        TRANSFORMER_V3_OUTPUT_DIR,
        r"root/results/raw_attention_metadata_Root.csv"
    )
}

# --- Data & Columns ---
KNOWN_METADATA_COLS_IN_FEATURES = [
    'Vac_id', 'Genotype', 'Entry', 'Tissue.type', 
    'Batch', 'Treatment', 'Replication', 'Day'
]
TARGET_COLS = ['Genotype', 'Treatment', 'Day']
METADATA_INDEX_COL = 'Row_names'

# --- Model Hyperparameters ---
HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
NUM_CLASSES = {'Genotype': 2, 'Treatment': 2, 'Day': 3}

# --- Preprocessing Parameters ---
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42
ENCODING_MAPS = {
    'Genotype': {'G1': 0, 'G2': 1},
    'Treatment': {0: 0, 1: 1},
    'Day': {1: 0, 2: 1, 3: 2}
}

# --- SHAP Parameters ---
SHAP_EXPLAINER = shap.GradientExplainer
SHAP_BACKGROUND_SAMPLES = 100
SHAP_INSTANCE_SAMPLES = 200
SHAP_MAX_DISPLAY = 20  # For basic bar plot

# --- Visualization Parameters ---
FIG_DPI = 300
TOP_N_FEATURES_HEATMAP = 50  # For clustermap
TOP_M_FEATURES_PER_TASK = 15  # For faceted bar plot
FIG_SIZE_WIDE = (12, 8)
FIG_SIZE_TALL = (10, 12)
FIG_SIZE_SQUARE = (10, 10)
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FEATURE_NAME_MAX_LENGTH = 40  # Truncate feature names in plots
OMICS_PALETTE = {"Spectral": "skyblue", "Metabolite": "lightcoral"}
CLUSTERMAP_CMAP = 'viridis'  # Colormap for clustermap

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== LOGGING =====
def setup_logging(log_dir: str, script_name: str, version: str) -> logging.Logger:
    """Set up logging for the script execution."""
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


logger = setup_logging(LOG_DIR, SCRIPT_NAME, VERSION)

logger.info("="*60)
logger.info(f"Starting SHAP Analysis Script: {SCRIPT_NAME} v{VERSION}")
logger.info(f"Analyzing v3 Feature Attention Model using GradientExplainer")
logger.info(f"Includes Integrated Advanced Visualizations")
logger.info(f"Output Directory: {SHAP_OUTPUT_DIR}")
logger.info(f"Using Device: {DEVICE}")
logger.info(f"SHAP Params: Background Samples={SHAP_BACKGROUND_SAMPLES}, "
           f"Instance Samples={SHAP_INSTANCE_SAMPLES}")
logger.info(f"Plot Params: Top N (Clustermap)={TOP_N_FEATURES_HEATMAP}, "
           f"Top M (Faceted)={TOP_M_FEATURES_PER_TASK}")
logger.info("="*60)


# ===== MODEL DEFINITION =====
class CrossAttentionLayer(nn.Module):
    """Cross attention layer for interaction between different omics types."""
    
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
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x1, x2):
        attn_output_1, attn_weights_1_to_2 = self.cross_attn_1_to_2(
            query=x1, key=x2, value=x2, average_attn_weights=False
        )
        x1 = self.norm1(x1 + self.dropout(attn_output_1))
        
        attn_output_2, attn_weights_2_to_1 = self.cross_attn_2_to_1(
            query=x2, key=x1, value=x1, average_attn_weights=False
        )
        x2 = self.norm2(x2 + self.dropout(attn_output_2))
        
        ffn_output1 = self.ffn(x1)
        x1 = self.norm3(x1 + self.dropout(ffn_output1))
        ffn_output2 = self.ffn(x2)
        x2 = self.norm3(x2 + self.dropout(ffn_output2))
        
        return x1, x2, attn_weights_1_to_2, attn_weights_2_to_1


class SimplifiedTransformer(nn.Module):
    """Multi-omic transformer model with feature-level attention."""
    
    def __init__(self, spectral_dim, metabolite_dim, hidden_dim, num_heads, 
                num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Feature embeddings
        self.spectral_feature_embedding = nn.Linear(1, hidden_dim)
        self.metabolite_feature_embedding = nn.Linear(1, hidden_dim)
        
        # Positional encodings
        self.pos_encoding_spec = nn.Parameter(
            torch.randn(1, spectral_dim, hidden_dim) * 0.02
        )
        self.pos_encoding_metab = nn.Parameter(
            torch.randn(1, metabolite_dim, hidden_dim) * 0.02
        )
        
        # Normalization and dropout
        self.embedding_norm_spec = nn.LayerNorm(hidden_dim)
        self.embedding_norm_metab = nn.LayerNorm(hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList(
            [CrossAttentionLayer(hidden_dim, num_heads, dropout) 
             for _ in range(num_layers)]
        )
        
        # Output heads for each task
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
        # Reshape inputs and apply feature embeddings
        spec_reshaped = spectral.unsqueeze(-1)
        metab_reshaped = metabolite.unsqueeze(-1)
        spec_emb = self.spectral_feature_embedding(spec_reshaped)
        metab_emb = self.metabolite_feature_embedding(metab_reshaped)
        
        # Add positional encodings
        spec_emb = spec_emb + self.pos_encoding_spec
        metab_emb = metab_emb + self.pos_encoding_metab
        
        # Apply normalization and dropout
        spec_emb = self.embedding_dropout(self.embedding_norm_spec(spec_emb))
        metab_emb = self.embedding_dropout(self.embedding_norm_metab(metab_emb))
        
        # Process through cross-attention layers
        self.attention_weights = {}
        last_attn_1_to_2, last_attn_2_to_1 = None, None
        
        for i, layer in enumerate(self.cross_attention_layers):
            spec_emb, metab_emb, attn_1_to_2, attn_2_to_1 = layer(spec_emb, metab_emb)
            if i == len(self.cross_attention_layers) - 1:
                self.attention_weights = {'1_to_2': attn_1_to_2, '2_to_1': attn_2_to_1}
        
        # Global pooling and task-specific predictions
        spec_pooled = spec_emb.mean(dim=1)
        metab_pooled = metab_emb.mean(dim=1)
        combined_pooled = torch.cat([spec_pooled, metab_pooled], dim=1)
        
        outputs = {}
        for task_name, head in self.output_heads.items():
            outputs[task_name] = head(combined_pooled)
            
        return outputs


# ===== DATA LOADING & PREPROCESSING =====
def load_metadata_file(feather_path: str, csv_fallback_path: str, 
                       index_col: str, target_cols: list) -> pd.DataFrame or None:
    """Load metadata from feather file with CSV fallback option."""
    metadata_df = None
    
    # Try loading from feather first
    if pyarrow and os.path.exists(feather_path):
        logger.info(f"Attempting to load metadata from Feather: {feather_path}")
        try:
            metadata_df = pd.read_feather(feather_path)
            if index_col in metadata_df.columns:
                metadata_df.set_index(index_col, inplace=True)
                logger.info(f"  Loaded metadata from Feather, shape: {metadata_df.shape}. "
                           f"Index set to '{index_col}'.")
            else:
                logger.error(f"Index column '{index_col}' not found in Feather file.")
                raise ValueError(f"Index column '{index_col}' not found in Feather file.")
                
            missing_targets = [col for col in target_cols if col not in metadata_df.columns]
            if missing_targets:
                logger.error(f"Metadata file {feather_path} missing targets: {missing_targets}")
                raise ValueError(f"Missing target columns: {missing_targets}")
                
            return metadata_df
        except Exception as e_feather:
            logger.warning(f"  Failed load from Feather {feather_path}: {e_feather}. Trying CSV.")
            metadata_df = None
    
    # Try loading from CSV as fallback
    if metadata_df is None and os.path.exists(csv_fallback_path):
        logger.info(f"Attempting to load metadata from CSV: {csv_fallback_path}")
        try:
            metadata_df = pd.read_csv(csv_fallback_path, index_col=index_col)
            if metadata_df.index.name == index_col:
                logger.info(f"  Loaded metadata from CSV, shape: {metadata_df.shape}. "
                           f"Index set to '{index_col}'.")
            else:
                logger.error(f"Failed to set index to '{index_col}' from CSV.")
                raise ValueError(f"Failed index set CSV '{index_col}'.")
                
            missing_targets = [col for col in target_cols if col not in metadata_df.columns]
            if missing_targets:
                logger.error(f"Metadata file {csv_fallback_path} missing targets: {missing_targets}")
                raise ValueError(f"Missing target columns: {missing_targets}")
                
            return metadata_df
        except Exception as e_csv:
            logger.error(f"  Failed to load metadata from CSV {csv_fallback_path}: {e_csv}")
            return None
    
    if metadata_df is None:
        logger.error(f"Metadata file not found or failed load. Checked Feather: "
                    f"{feather_path}, CSV: {csv_fallback_path}")
        
    return None


def load_and_preprocess_for_shap(config: dict, pairing: str) -> tuple:
    """Load and preprocess data for SHAP analysis."""
    logger.info(f"--- Starting Data Loading & Preprocessing for SHAP ({pairing}) ---")
    
    spectral_path = config['FEATURE_INPUT_FILES'][pairing]['spectral']
    metabolite_path = config['FEATURE_INPUT_FILES'][pairing]['metabolite']
    metadata_path = config['METADATA_INPUT_FILES'][pairing]
    metadata_csv_fallback = config['METADATA_CSV_FALLBACK_FILES'][pairing]
    index_col = config['METADATA_INDEX_COL']
    target_cols = config['TARGET_COLS']
    known_meta_in_features = config['KNOWN_METADATA_COLS_IN_FEATURES']

    try:
        # Load raw data files
        logger.info(f"Loading spectral data (may include metadata): {spectral_path}")
        spectral_df_raw = pd.read_csv(spectral_path, index_col=index_col, na_values='NA')
        
        logger.info(f"Loading metabolite data (may include metadata): {metabolite_path}")
        metabolite_df_raw = pd.read_csv(metabolite_path, index_col=index_col, na_values='NA')

        # Extract feature names
        spectral_feature_names = spectral_df_raw.columns.difference(
            known_meta_in_features, sort=False).tolist()
        metabolite_feature_names = metabolite_df_raw.columns.difference(
            known_meta_in_features, sort=False).tolist()
        
        if not spectral_feature_names:
            raise ValueError("No spectral features identified after excluding metadata.")
        if not metabolite_feature_names:
            raise ValueError("No metabolite features identified after excluding metadata.")

        # Extract features
        features_spectral_df = spectral_df_raw[spectral_feature_names]
        features_metabolite_df = metabolite_df_raw[metabolite_feature_names]
        
        logger.info(f"Identified {len(spectral_feature_names)} spectral features "
                   f"(e.g., '{spectral_feature_names[0]}').")
        logger.info(f"Identified {len(metabolite_feature_names)} metabolite features "
                   f"(e.g., '{metabolite_feature_names[0]}').")

        # Load metadata
        metadata_df = load_metadata_file(
            metadata_path, metadata_csv_fallback, index_col, target_cols
        )
        if metadata_df is None:
            raise FileNotFoundError(f"Metadata load failed for {pairing}.")
            
        logger.info(f"Feature DFs and Metadata loaded. Spec feats: "
                   f"{len(spectral_feature_names)}, Metab feats: "
                   f"{len(metabolite_feature_names)}, Meta rows: {len(metadata_df)}")

        # Align indices
        common_indices = features_spectral_df.index.intersection(
            features_metabolite_df.index).intersection(metadata_df.index)
        
        n_spectral = len(features_spectral_df)
        n_metab = len(features_metabolite_df)
        n_meta = len(metadata_df)
        n_common = len(common_indices)
        
        if n_common == 0:
            raise ValueError("Alignment failed: No common Row_names.")
        elif n_common < n_spectral or n_common < n_metab or n_common < n_meta:
            logger.warning(f"Found {n_common} common samples (Spec:{n_spectral}, "
                          f"Metab:{n_metab}, Meta:{n_meta}). Aligning.")
            
        features_spectral_df = features_spectral_df.loc[common_indices]
        features_metabolite_df = features_metabolite_df.loc[common_indices]
        metadata_df = metadata_df.loc[common_indices]
        
        logger.info(f"Data successfully aligned by index '{index_col}'. Samples: {n_common}")

        # Encode targets
        targets_encoded = pd.DataFrame(index=metadata_df.index)
        label_encoders = {}
        
        for col in target_cols:
            if col in config['ENCODING_MAPS']:
                targets_encoded[col] = metadata_df[col].map(config['ENCODING_MAPS'][col])
                if targets_encoded[col].isnull().any():
                    missing_vals = metadata_df.loc[targets_encoded[col].isnull(), col].unique()
                    raise ValueError(f"Encoding NaN for '{col}'. Unmapped: {missing_vals}")
            else:
                logger.warning(f"Using LabelEncoder for '{col}'.")
                le = LabelEncoder()
                targets_encoded[col] = le.fit_transform(metadata_df[col])
                label_encoders[col] = le

        # Perform train/test split
        logger.info(f"Performing train/test split using RANDOM_SEED={config['RANDOM_SEED']}")
        
        # Create a combined stratification key
        stratify_key_col = '_stratify_key_'
        metadata_df[stratify_key_col] = metadata_df[target_cols[0]].astype(str)
        for col in target_cols[1:]:
            metadata_df[stratify_key_col] += '_' + metadata_df[col].astype(str)
            
        indices = metadata_df.index
        stratify_values = metadata_df[stratify_key_col]
        
        if config['TEST_SIZE'] <= 0 or config['TEST_SIZE'] >= 1:
            raise ValueError("TEST_SIZE must be > 0 and < 1")
            
        try:
            train_idx, test_idx = train_test_split(
                indices, 
                test_size=config['TEST_SIZE'],
                random_state=config['RANDOM_SEED'],
                stratify=stratify_values
            )
        except ValueError as e:
            logger.warning(f"Stratification failed: {e}. No stratify.")
            train_idx, test_idx = train_test_split(
                indices,
                test_size=config['TEST_SIZE'],
                random_state=config['RANDOM_SEED']
            )
            
        metadata_df.drop(columns=[stratify_key_col], inplace=True, errors='ignore')

        # Split features for train/test sets
        X_train_spec = features_spectral_df.loc[train_idx]
        X_test_spec = features_spectral_df.loc[test_idx]
        X_train_metab = features_metabolite_df.loc[train_idx]
        X_test_metab = features_metabolite_df.loc[test_idx]
        
        logger.info(f"Split sizes for SHAP: Train={len(train_idx)}, Test={len(test_idx)}")
        
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError("Train or Test set is empty.")

        # Scale features
        logger.info("Scaling features (fit on train)...")
        scaler_spec = StandardScaler()
        scaler_metab = StandardScaler()
        
        X_train_spec_scaled = scaler_spec.fit_transform(X_train_spec)
        X_test_spec_scaled = scaler_spec.transform(X_test_spec)
        X_train_metab_scaled = scaler_metab.fit_transform(X_train_metab)
        X_test_metab_scaled = scaler_metab.transform(X_test_metab)
        
        if np.isnan(X_train_spec_scaled).any() or np.isinf(X_train_spec_scaled).any():
            raise ValueError("Bad values in scaled spec train")
            
        if np.isnan(X_train_metab_scaled).any() or np.isinf(X_train_metab_scaled).any():
            raise ValueError("Bad values in scaled metab train")

        # Convert to tensors
        logger.info("Converting to Tensors...")
        X_train_spec_tensor = torch.tensor(X_train_spec_scaled, dtype=torch.float32)
        X_train_metab_tensor = torch.tensor(X_train_metab_scaled, dtype=torch.float32)
        X_test_spec_tensor = torch.tensor(X_test_spec_scaled, dtype=torch.float32)
        X_test_metab_tensor = torch.tensor(X_test_metab_scaled, dtype=torch.float32)
        
        logger.info("--- Data Loading & Preprocessing for SHAP Finished ---")
        
        return (
            X_train_spec_tensor, X_train_metab_tensor,
            X_test_spec_tensor, X_test_metab_tensor,
            spectral_feature_names, metabolite_feature_names
        )
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}", exc_info=True)
        raise


# ===== MODEL LOADING =====
def load_trained_model(model_path: str, spectral_dim: int, metabolite_dim: int, 
                      config: dict, device: torch.device) -> SimplifiedTransformer:
    """Load trained model from checkpoint file."""
    logger.info(f"Loading trained model state from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
    model = SimplifiedTransformer(
        spectral_dim=spectral_dim,
        metabolite_dim=metabolite_dim,
        hidden_dim=config['HIDDEN_DIM'],
        num_heads=config['NUM_HEADS'],
        num_layers=config['NUM_LAYERS'],
        num_classes=config['NUM_CLASSES'],
        dropout=config['DROPOUT']
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model loaded and set to eval mode.")
        return model
    except Exception as e:
        logger.error(f"Error loading model state dict: {e}", exc_info=True)
        raise


# ===== SHAP WRAPPER MODEL (Concatenated Input) =====
class ShapModelWrapper(nn.Module):
    """Wrapper around the transformer model for use with SHAP explainer.
    
    Handles splitting concatenated input into spectral and metabolite components.
    """
    
    def __init__(self, original_model, task_name, spectral_dim, metabolite_dim):
        super().__init__()
        self.original_model = original_model
        self.task_name = task_name
        self.spectral_dim = spectral_dim
        self.metabolite_dim = metabolite_dim
        self.device = next(original_model.parameters()).device
        self.num_classes = original_model.num_classes[task_name]
        
    def forward(self, combined_input):
        # Convert numpy array to tensor if needed
        if isinstance(combined_input, np.ndarray):
            combined_input = torch.from_numpy(combined_input).float()
            
        # Move input to correct device
        combined_input = combined_input.to(self.device)
        
        # Split combined input into spectral and metabolite components
        split_sizes = [self.spectral_dim, self.metabolite_dim]
        try:
            spectral, metabolite = torch.split(combined_input, split_sizes, dim=1)
        except Exception as e_split:
            logger.error(f"Error splitting SHAP input in wrapper ({self.task_name}): "
                        f"{e_split}. Shape: {combined_input.shape}, "
                        f"Splits: {split_sizes}. Returning zeros.")
            batch_size = combined_input.shape[0]
            return torch.zeros((batch_size, self.num_classes), device=self.device)
            
        # Forward pass through original model
        outputs = self.original_model(spectral, metabolite)
        return outputs[self.task_name]


# ===== SHAP CALCULATION (GradientExplainer, Concatenated Input) =====
def calculate_shap_values(model, X_train_spec, X_train_metab, X_test_spec, X_test_metab,
                          spectral_feature_names, metabolite_feature_names,
                          target_cols, config, device):
    """Calculate SHAP values using GradientExplainer with concatenated input approach."""
    logger.info(f"--- Starting SHAP Value Calculation (Using "
               f"{config['SHAP_EXPLAINER'].__name__} & Concatenated Input) ---")
    
    spectral_dim = len(spectral_feature_names)
    metabolite_dim = len(metabolite_feature_names)
    total_features = spectral_dim + metabolite_dim

    # Prepare background data
    bg_samples = config['SHAP_BACKGROUND_SAMPLES']
    n_train = X_train_spec.shape[0]
    
    if n_train == 0:
        raise ValueError("Training set empty.")
        
    if bg_samples >= n_train:
        logger.warning(f"BG samples ({bg_samples}) >= train ({n_train}). Using all.")
        bg_indices = np.arange(n_train)
        bg_samples = n_train
    else:
        logger.info(f"Selecting {bg_samples} BG samples.")
        bg_indices = np.random.choice(n_train, bg_samples, replace=False)
        
    background_spec = X_train_spec[bg_indices]
    background_metab = X_train_metab[bg_indices]
    background_combined = torch.cat((background_spec, background_metab), dim=1)
    logger.info(f"BG shape (Concat): {background_combined.shape}")

    # Prepare instance data
    instance_samples = config['SHAP_INSTANCE_SAMPLES']
    n_test = X_test_spec.shape[0]
    
    if n_test == 0:
        raise ValueError("Test set empty.")
        
    if instance_samples is None or instance_samples >= n_test:
        logger.info(f"Using all {n_test} test samples.")
        instance_indices = np.arange(n_test)
        instance_samples = n_test
    else:
        logger.info(f"Selecting {instance_samples} instance samples.")
        instance_indices = np.random.choice(n_test, instance_samples, replace=False)
        
    instance_spec = X_test_spec[instance_indices]
    instance_metab = X_test_metab[instance_indices]
    instance_combined = torch.cat((instance_spec, instance_metab), dim=1)
    logger.info(f"Instance shape (Concat): {instance_combined.shape}")

    # Calculate SHAP values for each task
    all_shap_values_combined = {}
    ExplainerClass = config['SHAP_EXPLAINER']

    for task_name in target_cols:
        logger.info(f"--- Calculating SHAP values for Task: {task_name} ---")
        start_task_time = time.time()
        
        # Create task-specific model wrapper
        shap_wrapper_model = ShapModelWrapper(
            model, task_name, spectral_dim, metabolite_dim
        ).to(device)
        shap_wrapper_model.eval()
        background_combined_dev = background_combined.to(device)

        # Initialize SHAP explainer
        try:
            explainer = ExplainerClass(shap_wrapper_model, background_combined_dev)
            logger.info(f"SHAP {ExplainerClass.__name__} instantiated for {task_name}.")
        except Exception as e_explain:
            logger.error(f"Failed to instantiate {ExplainerClass.__name__}: {e_explain}", 
                        exc_info=True)
            all_shap_values_combined[task_name] = None
            continue

        # Calculate SHAP values
        instance_combined_dev = instance_combined.to(device)
        logger.info(f"Calculating SHAP values for {instance_combined_dev.shape[0]} instances...")
        try:
            shap_values_raw = explainer.shap_values(instance_combined_dev)

            # Log SHAP output structure
            if isinstance(shap_values_raw, list):
                shapes_info_list = [f"{i}:{getattr(e, 'shape', 'N/A')}" 
                                   for i, e in enumerate(shap_values_raw)]
                types_info_list = [f"{i}:{type(e)}" 
                                  for i, e in enumerate(shap_values_raw)]
                logger.info(f"SHAP returned list (len={len(shap_values_raw)}). "
                           f"Shapes: [{'; '.join(shapes_info_list)}]. "
                           f"Types: [{'; '.join(types_info_list)}]")
            elif isinstance(shap_values_raw, np.ndarray):
                logger.info(f"SHAP returned ndarray shape {shap_values_raw.shape}.")
            else:
                logger.info(f"SHAP returned type: {type(shap_values_raw)}")

            # Validate and normalize SHAP output format
            num_expected_classes = shap_wrapper_model.num_classes
            n_samples_expected = instance_combined_dev.shape[0]
            n_features_expected = instance_combined_dev.shape[1]

            shap_values_list_np = []
            
            # Handle 3D array output
            if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
                n_samples_out, n_features_out, n_classes_out = shap_values_raw.shape
                logger.info(f"SHAP returned 3D NumPy array shape: "
                           f"{(n_samples_out, n_features_out, n_classes_out)}. "
                           f"Converting to list of 2D arrays.")
                           
                if (n_samples_out != n_samples_expected or 
                    n_features_out != n_features_expected or 
                    n_classes_out != num_expected_classes):
                    logger.error(f"SHAP output array dimension mismatch! "
                                f"Got ({n_samples_out}, {n_features_out}, {n_classes_out}), "
                                f"expected ({n_samples_expected}, {n_features_expected}, "
                                f"{num_expected_classes}).")
                    raise ValueError(f"Dimension mismatch in SHAP output array for task {task_name}")
                    
                for i in range(n_classes_out):
                    shap_values_list_np.append(shap_values_raw[:, :, i])
                    
                logger.info(f"Converted 3D array into list of {len(shap_values_list_np)} arrays.")

            # Handle list output
            elif isinstance(shap_values_raw, list) and len(shap_values_raw) == num_expected_classes:
                logger.info("SHAP returned a list as expected. Converting elements to NumPy CPU arrays.")
                shap_values_list_np = [(v.cpu().numpy() if isinstance(v, torch.Tensor) else v) 
                                     for v in shap_values_raw]
                                     
                for i, arr in enumerate(shap_values_list_np):
                    if not isinstance(arr, np.ndarray) or arr.shape != (n_samples_expected, n_features_expected):
                        raise ValueError(f"List element {i} has incorrect shape/type: "
                                       f"{type(arr)}, {getattr(arr, 'shape', 'N/A')}")
            else:
                raise TypeError(f"Unexpected SHAP output format for {task_name}. "
                              f"Got type {type(shap_values_raw)} with unexpected structure/dims.")

            # Validate final output structure
            all_valid_structure = True
            if not isinstance(shap_values_list_np, list) or len(shap_values_list_np) != num_expected_classes:
                all_valid_structure = False
                logger.error(f"Validation failed: result is not a list of length {num_expected_classes}.")
            else:
                for i, arr in enumerate(shap_values_list_np):
                    if not isinstance(arr, np.ndarray) or arr.shape != (n_samples_expected, n_features_expected):
                        logger.error(f"Validation failed: List element {i} has incorrect shape/type: "
                                   f"{type(arr)}, {getattr(arr, 'shape', 'N/A')}. "
                                   f"Expected ({n_samples_expected}, {n_features_expected}).")
                        all_valid_structure = False
                        break
                        
            if not all_valid_structure:
                raise ValueError(f"Validation failed after attempting format conversion for {task_name}.")

            all_shap_values_combined[task_name] = shap_values_list_np
            logger.info(f"SHAP values calculated and formatted correctly for {task_name}.")

        except Exception as e_shap:
            logger.error(f"SHAP calc failed for {task_name}: {e_shap}", exc_info=True)
            all_shap_values_combined[task_name] = None
            continue

        end_task_time = time.time()
        logger.info(f"--- SHAP Task {task_name} finished. "
                   f"Duration: {end_task_time - start_task_time:.2f} sec ---")

    logger.info("--- SHAP Value Calculation Finished ---")
    return all_shap_values_combined, instance_spec.cpu(), instance_metab.cpu()


# ===== BASIC SHAP PLOTTING & SAVING =====
# (Same as before - returns dict of DataFrames)
def plot_and_save_basic_shap(all_shap_values_combined, instance_spec_cpu, instance_metab_cpu,
                             spectral_features, metabolite_features, target_cols,
                             pairing, config):
    logger.info(f"--- Generating Basic SHAP Plots & Saving Importance ({pairing}) ---")
    data_dir = config['SHAP_DATA_DIR']; plot_dir = config['SHAP_PLOT_DIR']; max_display = config['SHAP_MAX_DISPLAY']
    all_feature_names = spectral_features + metabolite_features
    num_spectral_features = len(spectral_features); num_metabolite_features = len(metabolite_features); num_total_features = len(all_feature_names)

    instance_spec_np = instance_spec_cpu.numpy() if isinstance(instance_spec_cpu, torch.Tensor) else instance_spec_cpu
    instance_metab_np = instance_metab_cpu.numpy() if isinstance(instance_metab_cpu, torch.Tensor) else instance_metab_cpu
    try: instance_features_combined_np = np.hstack((instance_spec_np, instance_metab_np)); instance_df = pd.DataFrame(instance_features_combined_np, columns=all_feature_names); num_instances_data = instance_df.shape[0]
    except ValueError as e: logger.error(f"Error creating instance DataFrame for {pairing}: {e}", exc_info=True); return {}

    aggregated_importance_dict = {}
    for task_name in target_cols:
        logger.info(f"Processing SHAP results for Task: {task_name}")
        if task_name not in all_shap_values_combined or all_shap_values_combined[task_name] is None: logger.warning(f"No valid SHAP values found for task {task_name}. Skipping."); continue
        shap_values_list = all_shap_values_combined[task_name]
        num_classes = len(shap_values_list)
        valid_structure = isinstance(shap_values_list, list) and num_classes > 0 and all(isinstance(a, np.ndarray) and len(a.shape) == 2 and a.shape[0] == num_instances_data and a.shape[1] == num_total_features for a in shap_values_list)
        if not valid_structure: logger.error(f"Invalid SHAP structure for {task_name}. Skipping."); continue

        plot_class_names = [f'Class {i}' for i in range(num_classes)]
        try:
            plt.figure(); shap.summary_plot(shap_values_list, instance_df, plot_type="bar", feature_names=all_feature_names, max_display=max_display, show=False, class_names=plot_class_names)
            plt.title(f'SHAP Importance (Bar) - {pairing} - Task: {task_name}'); plot_filename_bar = os.path.join(plot_dir, f"shap_summary_bar_{pairing}_{task_name}.png")
            plt.tight_layout(); plt.savefig(plot_filename_bar, dpi=config['FIG_DPI'], bbox_inches='tight'); plt.close(); logger.info(f"Saved SHAP Bar plot: {plot_filename_bar}")
        except Exception as e_plot_bar: logger.error(f"Error SHAP Bar plot {task_name}: {e_plot_bar}", exc_info=True); plt.close()

        try:
            all_classes_shap_np = np.array(shap_values_list); mean_abs_shap = np.mean(np.abs(all_classes_shap_np), axis=(0, 1))
            importance_df = pd.DataFrame({'Feature': all_feature_names, 'MeanAbsoluteShap': mean_abs_shap})
            feature_types = ['Spectral'] * num_spectral_features + ['Metabolite'] * num_metabolite_features
            if len(feature_types) == len(importance_df): importance_df['FeatureType'] = feature_types
            else: logger.warning(f"Feature type list length mismatch. Skipping adding FeatureType.")
            importance_df = importance_df.sort_values(by='MeanAbsoluteShap', ascending=False).reset_index(drop=True)
            importance_df['Task'] = task_name; importance_df['Pairing'] = pairing
            aggregated_importance_dict[task_name] = importance_df
            csv_filename = os.path.join(data_dir, f"shap_importance_{pairing}_{task_name}.csv")
            importance_df.to_csv(csv_filename, index=False); logger.info(f"Saved aggregated SHAP importance: {csv_filename}")
        except Exception as e_agg: logger.error(f"Error aggregating/saving SHAP importance {task_name}: {e_agg}", exc_info=True)

    logger.info(f"--- Basic SHAP Plotting and Saving Finished ({pairing}) ---")
    return aggregated_importance_dict


# ===== ADVANCED VISUALIZATION FUNCTIONS (Integrated) =====

# --- Helper for Advanced Plots ---
def truncate_feature_names(feature_names, max_length=FEATURE_NAME_MAX_LENGTH):
    """Truncate feature names for plotting."""
    return [name[:max_length-3] + '...' if len(name) > max_length else name for name in feature_names]

# --- SHAP Clustermap ---
def plot_shap_clustermap(shap_data: pd.DataFrame, pairing: str, config: dict):
    # (Same as version 1.1.0)
    logger.info(f"--- Generating SHAP Clustermap for {pairing} ---")
    output_dir = config['SHAP_PLOT_DIR']
    top_n = config['TOP_N_FEATURES_HEATMAP']
    cmap = config['CLUSTERMAP_CMAP']
    figsize = config['FIG_SIZE_SQUARE']

    if shap_data is None or shap_data.empty: logger.warning(f"No SHAP data provided for {pairing}, skipping clustermap."); return

    try:
        if 'Task' not in shap_data.columns: logger.error("Combined SHAP data missing 'Task' column. Cannot create clustermap."); return
        if 'FeatureType' not in shap_data.columns: logger.warning("Combined SHAP data missing 'FeatureType'. Row colors missing."); shap_data['FeatureType'] = 'Unknown'

        idx_max = shap_data.groupby('Feature')['MeanAbsoluteShap'].idxmax()
        top_features_df = shap_data.loc[idx_max]
        top_features = top_features_df.nlargest(top_n, 'MeanAbsoluteShap')['Feature'].tolist()
        if not top_features: logger.warning(f"Could not determine top {top_n} features for {pairing}."); return
        logger.info(f"Plotting top {len(top_features)} features based on max SHAP value across tasks.")

        plot_data = shap_data[shap_data['Feature'].isin(top_features)]
        pivot_table = plot_data.pivot_table(index='Feature', columns='Task', values='MeanAbsoluteShap', fill_value=0)

        feature_types = plot_data[['Feature', 'FeatureType']].drop_duplicates().set_index('Feature')
        feature_types = feature_types.reindex(pivot_table.index)
        row_colors = feature_types['FeatureType'].map(config['OMICS_PALETTE']).fillna('grey').rename('Omics Type')

        g = sns.clustermap(pivot_table, cmap=cmap, figsize=figsize,
                           row_colors=row_colors if not row_colors.empty else None,
                           linewidths=0.5, linecolor='lightgray', dendrogram_ratio=(.2, .1),
                           cbar_pos=(0.02, 0.8, 0.03, 0.18), z_score=0, annot=False)

        title_text = f'Top {len(top_features)} Feature SHAP Importance ({pairing})'
        g.ax_heatmap.set_title(title_text, fontsize=config['FONT_SIZE_TITLE'], pad=20)
        g.ax_heatmap.set_xlabel("Prediction Task", fontsize=config['FONT_SIZE_LABEL'])
        g.ax_heatmap.set_ylabel("Feature", fontsize=config['FONT_SIZE_LABEL'])
        g.ax_heatmap.tick_params(axis='x', labelsize=config['FONT_SIZE_TICK'])
        g.ax_heatmap.tick_params(axis='y', labelsize=max(6, config['FONT_SIZE_TICK'] - 2))

        try:
            handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in config['OMICS_PALETTE'].values() if color != 'grey'] # Exclude grey if added
            labels = [k for k,v in config['OMICS_PALETTE'].items() if v != 'grey']
            if 'Unknown' in feature_types['FeatureType'].unique(): # Add Unknown if present
                 handles.append(plt.Rectangle((0,0),1,1, color='grey'))
                 labels.append('Unknown')
            if handles: # Only add legend if there are omics types
                 g.fig.legend(handles=handles, labels=labels, title='Omics Type',
                              bbox_to_anchor=(0.02, 0.02), loc='lower left', frameon=False)
        except Exception as e_legend: logger.warning(f"Could not create/position row color legend: {e_legend}")

        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0)
        fpath = os.path.join(output_dir, f"shap_clustermap_{pairing}_top{top_n}.png")
        plt.savefig(fpath, dpi=config['FIG_DPI'], bbox_inches='tight'); plt.close(g.fig)
        logger.info(f"Saved SHAP clustermap: {fpath}")
    except Exception as e: logger.error(f"Error generating SHAP clustermap for {pairing}: {e}", exc_info=True); plt.close()


# --- Omics Contribution Stacked Bar ---
def plot_omics_contribution_stacked_bar(shap_data: pd.DataFrame, pairing: str, config: dict):
    """Generate stacked bar plot showing contribution of each omics type to prediction tasks."""
    logger.info(f"--- Generating Omics Contribution Stacked Bar Plot for {pairing} ---")
    
    output_dir = config['SHAP_PLOT_DIR']
    palette = config['OMICS_PALETTE'].copy()  # Use a copy to potentially add 'Unknown'

    if shap_data is None or shap_data.empty:
        logger.warning(f"No SHAP data provided for {pairing}, skipping contribution plot.")
        return

    try:
        # Validate required columns
        if not all(c in shap_data.columns for c in ['Task', 'FeatureType', 'MeanAbsoluteShap']):
            logger.error("Missing required columns (Task, FeatureType, MeanAbsoluteShap). "
                        "Cannot create contribution plot.")
            return

        # Handle unknown feature types
        shap_data['FeatureType'] = shap_data['FeatureType'].fillna('Unknown')
        if 'Unknown' in shap_data['FeatureType'].unique() and 'Unknown' not in palette:
            palette['Unknown'] = 'grey'

        # Calculate contributions by omics type for each task
        contribution = shap_data.groupby(['Task', 'FeatureType'])['MeanAbsoluteShap'].sum().reset_index()
        total_shap_per_task = contribution.groupby('Task')['MeanAbsoluteShap'].sum().reset_index()
        total_shap_per_task = total_shap_per_task.rename(columns={'MeanAbsoluteShap': 'TotalShap'})
        
        contribution = pd.merge(contribution, total_shap_per_task, on='Task')
        contribution['Proportion'] = 0.0
        non_zero_mask = contribution['TotalShap'] != 0
        contribution.loc[non_zero_mask, 'Proportion'] = (
            contribution.loc[non_zero_mask, 'MeanAbsoluteShap'] / 
            contribution.loc[non_zero_mask, 'TotalShap']
        ) * 100

        # Create pivot table for plotting
        pivot_prop = contribution.pivot(
            index='Task',
            columns='FeatureType',
            values='Proportion'
        ).fillna(0)
        
        # Select columns in order matching palette
        plot_order = [ft for ft in palette.keys() if ft in pivot_prop.columns]
        pivot_prop = pivot_prop[plot_order]

        # Create stacked bar plot
        fig, ax = plt.subplots(figsize=config['FIG_SIZE_WIDE'])
        pivot_prop.plot(
            kind='bar',
            stacked=True,
            color=[palette[col] for col in pivot_prop.columns],
            ax=ax,
            width=0.8
        )

        # Set plot labels and appearance
        ax.set_xlabel("Prediction Task", fontsize=config['FONT_SIZE_LABEL'])
        ax.set_ylabel("Proportion of Total SHAP Importance (%)", fontsize=config['FONT_SIZE_LABEL'])
        ax.set_title(f"Relative Contribution of Omics Types to Task Prediction ({pairing})",
                    fontsize=config['FONT_SIZE_TITLE'])
        ax.tick_params(axis='x', labelsize=config['FONT_SIZE_TICK'], rotation=0)
        ax.tick_params(axis='y', labelsize=config['FONT_SIZE_TICK'])
        ax.legend(title='Omics Type', bbox_to_anchor=(1.02, 0), loc='lower left')
        ax.set_ylim(0, 100)
        
        plt.tight_layout(rect=[0, 0, 0.88, 1])

        # Save plot
        fpath = os.path.join(output_dir, f"shap_omics_contribution_stackedbar_{pairing}.png")
        plt.savefig(fpath, dpi=config['FIG_DPI'], bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved Omics Contribution plot: {fpath}")
        
    except Exception as e:
        logger.error(f"Error generating Omics Contribution plot for {pairing}: {e}",
                    exc_info=True)
        plt.close()


# --- Faceted Top Features Bar Plot ---
def plot_faceted_top_features(shap_data: pd.DataFrame, pairing: str, config: dict):
    """
    Creates a faceted bar plot showing the top M features for each task side by side.
    """
    logger.info(f"--- Generating Faceted Top Features Plot for {pairing} ---")
    
    output_dir = config['SHAP_PLOT_DIR']
    top_m = config['TOP_M_FEATURES_PER_TASK']
    palette = config['OMICS_PALETTE'].copy()
    figsize = config['FIG_SIZE_TALL']  # Use tall figure size

    if shap_data is None or shap_data.empty:
        logger.warning(f"No SHAP data provided for {pairing}, skipping faceted plot.")
        return

    try:
        # Validate required columns
        if not all(c in shap_data.columns for c in ['Task', 'Feature', 'MeanAbsoluteShap', 'FeatureType']):
            logger.error("Missing required columns for faceted plot. Skipping.")
            return

        # Handle unknown feature types
        shap_data['FeatureType'] = shap_data['FeatureType'].fillna('Unknown')
        if 'Unknown' in shap_data['FeatureType'].unique() and 'Unknown' not in palette:
            palette['Unknown'] = 'grey'

        # Get top M features *per task*
        task_top_features_list = []
        unique_tasks = shap_data['Task'].unique()
        
        for task in unique_tasks:
            task_data = shap_data[shap_data['Task'] == task]
            if not task_data.empty:
                top_m_df = task_data.nlargest(top_m, 'MeanAbsoluteShap')
                task_top_features_list.append(top_m_df)

        if not task_top_features_list:
            logger.warning(f"No top features found for any task in {pairing}. Skipping faceted plot.")
            return

        # Combine all task-specific top features
        top_features_df = pd.concat(task_top_features_list, ignore_index=True)

        # Create the faceted plot using catplot
        g = sns.catplot(
            data=top_features_df,
            x='MeanAbsoluteShap',
            y='Feature',
            hue='FeatureType',
            col='Task',
            kind='bar',
            height=figsize[1] / len(unique_tasks) * 1.5,  # Adjust height based on number of tasks
            aspect=1.0,  # Adjust aspect ratio
            palette=palette,
            sharey=False,  # Each facet has its own y-axis
            legend=False,  # Turn off automatic legend, add manually later
            # Attempt to order y-axis globally (might not work perfectly with sharey=False)
            order=top_features_df.sort_values('MeanAbsoluteShap', ascending=False)['Feature'].tolist()
        )

        # Adjust plot appearance
        g.set_titles(col_template="{col_name}", size=config['FONT_SIZE_LABEL'])
        g.set_axis_labels("Mean |SHAP value|", "Feature")
        # Adjust title position
        g.fig.suptitle(f'Top {top_m} Features per Task ({pairing})',
                      fontsize=config['FONT_SIZE_TITLE'], y=1.03)

        # Truncate long feature names on y-axis for each facet
        for ax in g.axes.flat:
            labels = ax.get_yticklabels()
            truncated_labels = truncate_feature_names(
                [label.get_text() for label in labels],
                config['FEATURE_NAME_MAX_LENGTH']
            )
            ax.set_yticklabels(truncated_labels, fontsize=config['FONT_SIZE_TICK'])
            ax.tick_params(axis='x', labelsize=config['FONT_SIZE_TICK'])

        # Add a single legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=color) 
                  for color in palette.values() if color != 'grey']
        labels = [k for k, v in palette.items() if v != 'grey']
        
        if 'Unknown' in top_features_df['FeatureType'].unique():
            handles.append(plt.Rectangle((0, 0), 1, 1, color='grey'))
            labels.append('Unknown')
            
        if handles:
            g.fig.legend(
                handles=handles,
                labels=labels,
                title='Omics Type',
                bbox_to_anchor=(1.05, 0),
                loc='lower left',
                frameon=False
            )

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make space for legend

        # Save plot
        fpath = os.path.join(output_dir, f"shap_faceted_top_features_{pairing}_top{top_m}.png")
        plt.savefig(fpath, dpi=config['FIG_DPI'], bbox_inches='tight')
        plt.close(g.fig)
        logger.info(f"Saved faceted top features plot: {fpath}")

    except Exception as e:
        logger.error(f"Error creating faceted top features plot for {pairing}: {e}", 
                    exc_info=True)
        plt.close()


# ===== MAIN EXECUTION =====
def main():
    """Main execution function for SHAP analysis and integrated visualization."""
    main_start_time = time.time()
    logger.info(f"--- Starting Main Execution ({SCRIPT_NAME} v{VERSION}) ---")

    # Define pairings to process
    pairings_to_process = ["Leaf", "Root"]

    # Loop through analysis pairings
    for pairing in pairings_to_process:
        logger.info(f"\n===== Processing SHAP Analysis & Plots for Pairing: {pairing} =====")
        pairing_start_time = time.time()
        aggregated_importance_all_tasks = {}  # To store results for advanced plots

        try:
            # 1. Load and preprocess data
            (X_train_spec, X_train_metab, X_test_spec, X_test_metab,
             spectral_features, metabolite_features) = load_and_preprocess_for_shap(
                 config=globals(), pairing=pairing
             )
            spectral_dim = len(spectral_features)
            metabolite_dim = len(metabolite_features)
            logger.info(f"Data loaded: Spec Dim={spectral_dim}, Metab Dim={metabolite_dim}")

            # 2. Load the trained model
            model_path = MODEL_PATHS.get(pairing)
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Model path invalid/not found for {pairing}: {model_path}. Skipping.")
                continue
                
            model = load_trained_model(
                model_path, spectral_dim, metabolite_dim, config=globals(), device=DEVICE
            )

            # 3. Calculate SHAP values
            all_shap_values, instance_spec_cpu, instance_metab_cpu = calculate_shap_values(
                model, X_train_spec, X_train_metab, X_test_spec, X_test_metab,
                spectral_features, metabolite_features,
                TARGET_COLS, config=globals(), device=DEVICE
            )

            # 4. Plot BASIC SHAP summaries and save/collect aggregated importance CSVs
            aggregated_importance_all_tasks = plot_and_save_basic_shap(
                all_shap_values, instance_spec_cpu, instance_metab_cpu,
                spectral_features, metabolite_features, TARGET_COLS,
                pairing, config=globals()
            )

            # 5. Generate ADVANCED Visualizations if importance data was generated
            if aggregated_importance_all_tasks:
                logger.info(f"--- Generating Advanced Visualizations for {pairing} ---")
                try:
                    # Combine the task-specific importance dataframes into one
                    combined_shap_df = pd.concat(
                        aggregated_importance_all_tasks.values(), ignore_index=True
                    )

                    # Validate combined dataframe has required columns
                    req_cols = ['Feature', 'MeanAbsoluteShap', 'FeatureType', 'Task', 'Pairing']
                    if not all(col in combined_shap_df.columns for col in req_cols):
                        logger.error("Combined DataFrame is missing essential columns. "
                                    "Skipping advanced plots.")
                    else:
                        # Call advanced plotting functions
                        plot_shap_clustermap(combined_shap_df, pairing, config=globals())
                        plot_omics_contribution_stacked_bar(combined_shap_df, pairing, config=globals())
                        plot_faceted_top_features(combined_shap_df, pairing, config=globals())

                except Exception as e_combine:
                    logger.error(f"Error combining importance data or generating advanced plots "
                                f"for {pairing}: {e_combine}", exc_info=True)
            else:
                logger.warning(f"Skipping advanced visualizations for {pairing} as no "
                             f"aggregated importance data was generated.")

        except FileNotFoundError as e:
            logger.error(f"Skipping analysis for {pairing} due to missing file: {e}")
        except ValueError as e:
            logger.error(f"Skipping analysis for {pairing} due to data/value error: {e}")
        except Exception as e:
            logger.error(f"Error during analysis phase for {pairing}: {e}", exc_info=True)

        pairing_end_time = time.time()
        logger.info(f"===== Finished Analysis & Plots for {pairing}. "
                   f"Duration: {(pairing_end_time - pairing_start_time):.2f} seconds =====")

    main_end_time = time.time()
    total_duration_min = (main_end_time - main_start_time) / 60
    logger.info(f"--- Main Execution Finished --- "
               f"Total Duration: {total_duration_min:.2f} minutes ---")
    logger.info(f"Outputs saved in: {SHAP_OUTPUT_DIR}")
    logger.info("="*60)


# --- Entry Point ---
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"A critical error occurred during the main execution workflow: {e}",
                      exc_info=True)
        sys.exit(1)