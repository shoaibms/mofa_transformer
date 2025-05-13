# -*- coding: utf-8 -*-
"""
Figure 4: SHAP Importance Clustermap Generator

This script generates clustermaps (heatmaps) visualizing the top 50 features based on 
SHAP importance values for Leaf and Root sample pairings. The script processes
pre-calculated SHAP importance data from CSV files, combines data across different
prediction tasks, and creates publication-ready visualizations.

Features:
- Loads SHAP importance data from CSV files for different tissue types and prediction tasks
- Generates clustermaps showing relative feature importance across prediction tasks
- Color-codes features by omics type (Spectral, Molecular feature, etc.)
- Saves high-resolution figures suitable for publication

Inputs: CSV files with SHAP importance data
Outputs: PNG visualization files with clustermaps
"""

# ===== IMPORTS =====
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the color definitions
from colour import COLORS

# ===== CONFIGURATION =====

# --- Script Info ---
SCRIPT_NAME = "Figure8_heatmap"
VERSION = "1.0.0"

# --- Paths ---
# Base directory structure
BASE_DIR = r"C:/Users/ms/Desktop/hyper"

# Input: Directory containing SHAP importance CSV data
SHAP_DATA_DIR = os.path.join(BASE_DIR, "output", "transformer", "shap_analysis_ggl", "importance_data")

# Output: Directory for saving the heatmap plots
OUTPUT_PLOT_DIR = os.path.join(BASE_DIR, "output", "transformer", "novility_plot")
LOG_DIR = os.path.join(OUTPUT_PLOT_DIR, "logs")

# Ensure output directories exist
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Data & Plotting Parameters ---
TARGET_COLS = ['Genotype', 'Treatment', 'Time point']
PAIRINGS_TO_PROCESS = ["Leaf", "Root"]

# Clustermap specific parameters
FIG_DPI = 300
TOP_N_FEATURES_HEATMAP = 50
FIG_SIZE_SQUARE = (10, 10)
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
OMICS_PALETTE = {
    "Spectral": COLORS.get('Spectral', 'skyblue'),
    "Molecular feature": COLORS.get('Metabolite', 'lightcoral'),
    "Unknown": COLORS.get('UnknownFeature', 'grey')
}
CLUSTERMAP_CMAP = 'BuGn'


# ===== LOGGING =====
def setup_logging(log_dir: str, script_name: str, version: str) -> logging.Logger:
    """Sets up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{script_name}_{version}_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Exception Hook
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
logger.info(f"Starting Heatmap Generation Script: {SCRIPT_NAME} v{VERSION}")
logger.info(f"Reading SHAP data from: {SHAP_DATA_DIR}")
logger.info(f"Saving clustermap plots to: {OUTPUT_PLOT_DIR}")
logger.info(f"Processing Pairings: {PAIRINGS_TO_PROCESS}")
logger.info(f"Top N Features for Heatmap: {TOP_N_FEATURES_HEATMAP}")
logger.info("="*60)


# ===== DATA LOADING FUNCTION =====
def load_combined_shap_data(pairing: str, shap_data_dir: str, target_cols: list) -> pd.DataFrame | None:
    """
    Loads and combines SHAP importance CSV files for a given pairing and tasks.
    
    Args:
        pairing: The tissue type (e.g., "Leaf" or "Root")
        shap_data_dir: Directory containing SHAP importance CSV files
        target_cols: List of prediction tasks to include
        
    Returns:
        Combined DataFrame of SHAP importance data or None if loading failed
    """
    logger.info(f"--- Loading SHAP importance data for pairing: {pairing} ---")
    all_task_dfs = []
    found_any = False
    for task_name in target_cols:
        # Handle the special case for 'Time point' in filenames
        csv_task_name = 'Day' if task_name == 'Time point' else task_name
        csv_filename = f"shap_importance_{pairing}_{csv_task_name}.csv"
        csv_filepath = os.path.join(shap_data_dir, csv_filename)

        if os.path.exists(csv_filepath):
            logger.info(f"  Loading file: {csv_filename}")
            try:
                task_df = pd.read_csv(csv_filepath)
                # Basic validation
                if not all(col in task_df.columns for col in [
                    'Feature', 'MeanAbsoluteShap', 'FeatureType', 'Task', 'Pairing'
                ]):
                    logger.warning(f"    File {csv_filename} is missing expected columns. Skipping.")
                    continue
                
                # Update any 'Day' task values to 'Time point'
                if task_df['Task'].iloc[0] == 'Day':
                    task_df['Task'] = 'Time point'
                
                # Validate after potential task name update
                if task_df['Task'].iloc[0] != task_name or task_df['Pairing'].iloc[0] != pairing:
                    logger.warning(
                        f"    File {csv_filename} has mismatched Task/Pairing metadata. Skipping."
                    )
                    continue

                # Update 'Metabolite' feature type to 'Molecular feature'
                task_df.loc[task_df['FeatureType'] == 'Metabolite', 'FeatureType'] = 'Molecular feature'

                all_task_dfs.append(task_df)
                found_any = True
            except Exception as e:
                logger.error(f"    Error loading or validating file {csv_filename}: {e}")
        else:
            logger.warning(f"  SHAP importance file not found: {csv_filename}")

    if not found_any:
        logger.error(
            f"No valid SHAP importance CSV files were found for pairing '{pairing}' "
            f"in {shap_data_dir}. Cannot generate heatmap."
        )
        return None

    if all_task_dfs:
        combined_shap_df = pd.concat(all_task_dfs, ignore_index=True)
        logger.info(
            f"Successfully loaded and combined data for {len(all_task_dfs)} tasks "
            f"for pairing '{pairing}'. Total rows: {len(combined_shap_df)}"
        )
        # Ensure FeatureType is present, fill if necessary
        if 'FeatureType' not in combined_shap_df.columns:
            logger.warning("  'FeatureType' column missing in loaded data. Filling with 'Unknown'.")
            combined_shap_df['FeatureType'] = 'Unknown'
        else:
            combined_shap_df['FeatureType'] = combined_shap_df['FeatureType'].fillna('Unknown')

        # Transform feature names
        combined_shap_df['Feature'] = combined_shap_df['Feature'].apply(
            lambda x: x.replace('N_Cluster_', 'N_').replace('P_Cluster_', 'P_') 
            if isinstance(x, str) else x
        )

        return combined_shap_df
    else:
        logger.error(f"Failed to load any SHAP data for pairing '{pairing}'.")
        return None


# ===== PLOTTING FUNCTION =====
def plot_shap_clustermap(shap_data: pd.DataFrame, pairing: str, config: dict):
    """
    Generates and saves a SHAP clustermap for the top N features.
    
    Args:
        shap_data: DataFrame containing SHAP importance data
        pairing: The tissue type (e.g., "Leaf" or "Root")
        config: Dictionary containing plotting parameters
    """
    logger.info(f"--- Generating SHAP Clustermap for {pairing} ---")

    output_dir = config['OUTPUT_PLOT_DIR']
    top_n = config['TOP_N_FEATURES_HEATMAP']
    cmap = config['CLUSTERMAP_CMAP']
    figsize = config['FIG_SIZE_SQUARE']
    palette = config['OMICS_PALETTE']
    
    # Determine the label based on pairing
    panel_label = "A)" if pairing == "Leaf" else "B)" if pairing == "Root" else ""

    if shap_data is None or shap_data.empty:
        logger.warning(f"No SHAP data provided for {pairing}, skipping clustermap.")
        return

    try:
        # Ensure required columns are present
        required_cols = ['Feature', 'MeanAbsoluteShap', 'Task', 'FeatureType']
        if not all(col in shap_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in shap_data.columns]
            logger.error(
                f"Combined SHAP data missing required columns: {missing}. Cannot create clustermap."
            )
            return

        # Ensure 'Unknown' is in palette if present in data
        if 'Unknown' in shap_data['FeatureType'].unique() and 'Unknown' not in palette:
            palette['Unknown'] = 'grey'

        # Find the top N features based on the maximum MeanAbsoluteShap across all tasks
        idx_max = shap_data.loc[shap_data.groupby('Feature')['MeanAbsoluteShap'].idxmax()]
        top_features = idx_max.nlargest(top_n, 'MeanAbsoluteShap')['Feature'].tolist()

        if not top_features:
            logger.warning(
                f"Could not determine top {top_n} features for {pairing}. "
                "The SHAP data might be empty or have zero importance values."
            )
            return
        logger.info(f"Plotting top {len(top_features)} features based on max SHAP value across tasks.")

        # Filter the original data to include only these top features
        plot_data = shap_data[shap_data['Feature'].isin(top_features)].copy()

        # Pivot the data for the heatmap
        pivot_table = plot_data.pivot_table(
            index='Feature', columns='Task', values='MeanAbsoluteShap', fill_value=0
        )

        # Create row colors based on FeatureType
        feature_types = plot_data[['Feature', 'FeatureType']].drop_duplicates().set_index('Feature')
        feature_types = feature_types.reindex(pivot_table.index)
        row_colors_series = feature_types['FeatureType'].map(palette).fillna('grey')
        row_colors_series.name = 'Omics Type'

        # Create the clustermap
        g = sns.clustermap(
            pivot_table,
            cmap=cmap,
            figsize=figsize,
            row_colors=row_colors_series if not row_colors_series.empty else None,
            linewidths=0.5,
            linecolor='lightgray',
            dendrogram_ratio=(.2, .1),
            cbar_pos=(0.02, 0.8, 0.03, 0.18),
            z_score=0,
            annot=False
        )

        # --- Customize Appearance ---
        title_text = f'{panel_label} Top {len(top_features)} Feature SHAP Importance ({pairing})'
        g.ax_heatmap.set_title(title_text, fontsize=config['FONT_SIZE_TITLE'], pad=20)
        g.ax_heatmap.set_xlabel("Prediction Task", fontsize=config['FONT_SIZE_LABEL'])
        g.ax_heatmap.set_ylabel("Feature", fontsize=config['FONT_SIZE_LABEL'])
        g.ax_heatmap.tick_params(axis='x', labelsize=config['FONT_SIZE_TICK'], rotation=0)
        g.ax_heatmap.tick_params(axis='y', labelsize=max(6, config['FONT_SIZE_TICK'] - 2))
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

        # --- Add Legend for Row Colors (Omics Type) ---
        try:
            present_types = feature_types['FeatureType'].unique()
            handles = [plt.Rectangle((0, 0), 1, 1, color=palette[ftype]) 
                      for ftype in present_types if ftype in palette]
            labels = [ftype for ftype in present_types if ftype in palette]

            if handles:
                g.fig.legend(
                    handles=handles, 
                    labels=labels, 
                    title='Omics Type',
                    bbox_to_anchor=(0.02, 0.02),
                    loc='lower left', 
                    frameon=False
                )
        except Exception as e_legend:
            logger.warning(f"Could not create/position row color legend: {e_legend}")

        # --- Save Figure ---
        fpath = os.path.join(output_dir, f"shap_clustermap_{pairing}_top{top_n}.png")
        plt.savefig(fpath, dpi=config['FIG_DPI'], bbox_inches='tight')
        plt.close(g.fig)
        logger.info(f"Saved SHAP clustermap: {fpath}")

    except Exception as e:
        logger.error(f"Error generating SHAP clustermap for {pairing}: {e}", exc_info=True)
        plt.close()


# ===== MAIN EXECUTION =====
def main():
    """Main execution function"""
    main_start_time = datetime.now()
    logger.info(f"--- Starting Main Execution ({SCRIPT_NAME} v{VERSION}) ---")

    # Prepare configuration dictionary to pass to plotting function
    plot_config = {
        'OUTPUT_PLOT_DIR': OUTPUT_PLOT_DIR,
        'TOP_N_FEATURES_HEATMAP': TOP_N_FEATURES_HEATMAP,
        'CLUSTERMAP_CMAP': CLUSTERMAP_CMAP,
        'FIG_SIZE_SQUARE': FIG_SIZE_SQUARE,
        'FONT_SIZE_TITLE': FONT_SIZE_TITLE,
        'FONT_SIZE_LABEL': FONT_SIZE_LABEL,
        'FONT_SIZE_TICK': FONT_SIZE_TICK,
        'OMICS_PALETTE': OMICS_PALETTE,
        'FIG_DPI': FIG_DPI,
    }

    # Loop through analysis pairings
    for pairing in PAIRINGS_TO_PROCESS:
        logger.info(f"\n===== Processing Heatmap for Pairing: {pairing} =====")
        pairing_start_time = datetime.now()

        # 1. Load the combined SHAP data for this pairing
        combined_shap_df = load_combined_shap_data(pairing, SHAP_DATA_DIR, TARGET_COLS)

        # 2. Generate the clustermap if data was loaded successfully
        if combined_shap_df is not None and not combined_shap_df.empty:
            plot_shap_clustermap(combined_shap_df, pairing, config=plot_config)
        else:
            logger.warning(f"Skipping heatmap generation for {pairing} due to data loading issues.")

        pairing_end_time = datetime.now()
        logger.info(
            f"===== Finished processing for {pairing}. "
            f"Duration: {pairing_end_time - pairing_start_time} ====="
        )

    main_end_time = datetime.now()
    total_duration = main_end_time - main_start_time
    logger.info(f"--- Main Execution Finished --- Total Duration: {total_duration} ---")
    logger.info(f"Clustermap plots saved in: {OUTPUT_PLOT_DIR}")
    logger.info("="*60)


# --- Entry Point ---
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
         logger.critical(
             f"A critical error occurred during the main execution workflow: {e}", 
             exc_info=True
         )
         sys.exit(1)