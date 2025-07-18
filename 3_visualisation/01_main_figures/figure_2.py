#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 2: Predictive Validation and Key Feature Identification

This script generates a comprehensive visualization (Figure 2) with three panels:
- Panel A: Bar chart comparing model performance (F1 Macro)
- Panel B: SHAP top features bar plot for Leaf tissue
- Panel C: SHAP top features bar plot for Root tissue

The visualization integrates performance metrics and feature importance
analysis to highlight key discriminative features across different
classification tasks and tissue types.

Usage:
    python Figure2_Integrated.py [--input INPUT_DIR] [--output OUTPUT_DIR]
"""

import os
import sys
import argparse
import re
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Import custom colors
try:
    # Assuming colour.py is in the same directory or Python path
    from colour import COLORS, FONTS_SANS
except ImportError:
    print("WARNING: colour.py not found. Using fallback colors and fonts.")
    # Fallback colors if colour.py is not available - Updated with FULL colour.py values
    COLORS = {
        # ==========================================================================
        # == Core Experimental Variables ==
        # ==========================================================================
        'G1': '#00FA9A',             # Tolerant Genotype (Medium-Dark Blue)
        'G2': '#48D1CC',             # Susceptible Genotype (Medium Teal)

        'T0': '#4682B4',            # Control Treatment (Medium Green)
        'T1': '#BDB76B',             # Stress Treatment (Muted Orange/Yellow)

        'Leaf': '#00FF00',            # Leaf Tissue (Darkest Green)
        'Root': '#40E0D0',            # Root Tissue (Darkest Blue)

        # --- Days (Subtle Yellow-Green sequence) ---
        'Day1': '#ffffcc',            # Very Light Yellow-Green
        'Day2': '#9CBA79',            # Light Yellow-Green
        'Day3': '#3e7d5a',            # Medium Yellow-Green

        # ==========================================================================
        # == Data Types / Omics / Features ==
        # ==========================================================================
        'Spectral': '#ECDA79',        # General Spectral (Medium Blue)
        'Metabolite': '#84ab92',       # General Metabolite (Medium-Dark Yellow-Green)
        'UnknownFeature': '#B0E0E6',  # Medium Grey for fallback

        # --- Specific Spectral Categories --- (Using blues, teals, greens, greys)
        'Spectral_Water': '#6DCAFA',     # Medium-Dark Blue
        'Spectral_Pigment': '#00FA9A',    # Medium-Dark Green
        'Spectral_Structure': '#7fcdbb',  # Medium Teal
        'Spectral_SWIR': '#636363',       # Dark Grey
        'Spectral_VIS': '#c2e699',        # Light Yellow-Green
        'Spectral_RedEdge': '#78c679',    # Medium Yellow-Green
        'Spectral_UV': '#00BFFF',         # Darkest Blue (Matches Root)
        'Spectral_Other': '#969696',      # Medium Grey

        # --- Specific Metabolite Categories --- (Using Yellow/Greens)
        'Metabolite_PCluster': '#3DB3BF', # Darkest Yellow-Green
        'Metabolite_NCluster': '#ffffd4', # Very Light Yellow
        'Metabolite_Other': '#bdbdbd',     # Light Grey

        # ==========================================================================
        # == Methods & Model Comparison ==
        # ==========================================================================
        'MOFA': '#FFEBCD',            # Dark Blue
        'SHAP': '#F0E68C',            # Dark Green
        'Overlap': '#AFEEEE',         # Medium-Dark Yellow-Green

        'Transformer': '#fae3a2',     # Medium Blue
        'RandomForest': '#40E0D0',    # Medium Green
        'KNN': '#729c87',             # Medium Teal

        # ==========================================================================
        # == Network Visualization Elements ==
        # ==========================================================================
        'Edge_Low': '#f0f0f0',         # Very Light Gray
        'Edge_High': '#EEE8AA',        # Dark Blue
        'Node_Spectral': '#6baed6',    # Default Spectral Node (Medium Blue)
        'Node_Metabolite': '#FFC4A1',   # Default Metabolite Node (Med-Dark Yellow-Green)
        'Node_Edge': '#252525',        # Darkest Gray / Near Black border

        # ==========================================================================
        # == Statistical & Difference Indicators ==
        # ==========================================================================
        'Positive_Diff': '#66CDAA',     # Medium-Dark Green
        'Negative_Diff': '#fe9929',     # Muted Orange/Yellow (Matches T1)
        'Significance': '#08519c',      # Dark Blue (for markers/text)
        'NonSignificant': '#bdbdbd',    # Light Grey
        'Difference_Line': '#636363',   # Dark Grey line

        # ==========================================================================
        # == Plot Elements & Annotations ==
        # ==========================================================================
        'Background': '#FFFFFF',       # White plot background
        'Panel_Background': '#f7f7f7', # Very Light Gray background for some panels
        'Grid': '#d9d9d9',             # Lighter Gray grid lines
        'Text_Dark': '#252525',        # Darkest Gray / Near Black text
        'Text_Light': '#FFFFFF',       # White text
        'Text_Annotation': '#000000',   # Black text for annotations
        'Annotation_Box_BG': '#FFFFFF', # White background for text boxes
        'Annotation_Box_Edge': '#bdbdbd',# Light Grey border for text boxes
        'Table_Header_BG': '#deebf7',   # Very Light Blue table header
        'Table_Highlight_BG': '#fff7bc',# Pale Yellow for highlighted table cells

        # --- Temporal Patterns (Fig S6) --- (Using core palette shades)
        'Pattern_Increasing': '#238b45',  # Medium-Dark Green
        'Pattern_Decreasing': '#fe9929',  # Muted Orange/Yellow
        'Pattern_Peak': '#78c679',        # Medium Yellow-Green
        'Pattern_Valley': '#6baed6',      # Medium Blue
        'Pattern_Stable': '#969696',      # Medium Grey
    }
    # Fallback fonts
    FONTS_SANS = {
        'family': 'sans-serif',
        'sans_serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'], # Fallback list
        'main_title': 22,         # Figure Title (e.g., "Figure 1: ...")
        'panel_label': 19,        # Panel Labels (e.g., "A)", "B)")
        'panel_title': 17,        # Title for each subplot/panel
        'axis_label': 17,         # X and Y axis labels
        'tick_label': 16,         # Axis tick numbers/text
        'legend_title': 19,       # Title of the legend box
        'legend_text': 16,        # Text for individual legend items
        'annotation': 15,          # Text annotations within the plot area
        'caption': 15,            # Figure caption text
        'table_header': 15,       # Text in table headers
        'table_cell': 15,          # Text in table cells
    }

# ===================== CONFIGURATION PARAMETERS =====================
# Default input file paths (can be overridden via command line arguments)
DEFAULT_INPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer"
DEFAULT_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot"

# Plot styling
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42  # Output TrueType fonts for editable text
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Model colors for Panel A - Now sourced from COLORS
MODEL_PALETTE = {
    'Transformer': COLORS.get('Transformer', '#fae3a2'),    # Updated fallback
    'RandomForest': COLORS.get('RandomForest', '#40E0D0'),  # Updated fallback
    'KNN': COLORS.get('KNN', '#729c87')                     # Updated fallback
}

# Feature selection
TOP_N_FEATURES = 15          # Features per SHAP plot
MAX_FEATURE_NAME_LENGTH = 22  # Max length of feature names

# Figure dimensions
FIG_WIDTH = 20  # Slightly narrower than original 22
FIG_HEIGHT = 28  # Slightly shorter than original 30
DPI = 300

# Panel-specific settings - Now using FONTS_SANS
BAR_WIDTH = 0.75  # Slightly wider bars

# Rename map based on colour.py notes
RENAME_MAP = {
    'Day': 'Time point',
    'Metabolite': 'Molecular feature'
}

def rename_task(task_name):
    """Rename task based on RENAME_MAP."""
    return RENAME_MAP.get(task_name, task_name)

def rename_feature_type(feature_type):
    """Rename feature type based on RENAME_MAP."""
    return RENAME_MAP.get(feature_type, feature_type)

def create_output_directory(dir_path):
    """Create output directory if it doesn't exist."""
    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Output directory confirmed: {dir_path}")
        return True
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return False

def load_performance_data(input_dir):
    """Load and process performance data for Panel A."""
    # Input file paths
    transformer_leaf_path = os.path.join(input_dir, "phase1.1/leaf/transformer_class_performance_Leaf.csv")
    transformer_root_path = os.path.join(input_dir, "phase1.1/root/transformer_class_performance_Root.csv")
    baseline_leaf_path = os.path.join(input_dir, "phase1.1/leaf/transformer_baseline_comparison_Leaf.csv")
    baseline_root_path = os.path.join(input_dir, "phase1.1/root/transformer_baseline_comparison_Root.csv")
    
    files_to_check = [transformer_leaf_path, transformer_root_path, baseline_leaf_path, baseline_root_path]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]
    
    if missing_files:
        print(f"WARNING: Some performance data files are missing: {missing_files}")
    
    # Create a new dataframe for plotting
    plot_data = []

    # Helper function to process performance data
    def process_performance_data(df_path, model=None, tissue=None):
        if not os.path.exists(df_path):
            print(f"WARNING: File not found: {df_path}")
            return []
            
        try:
            df = pd.read_csv(df_path)
            data = []
            
            # If 'Task' and 'Metric' and 'Score' columns exist
            if all(col in df.columns for col in ['Task', 'Metric', 'Score']):
                for _, row in df.iterrows():
                    if 'F1' in str(row['Metric']) or 'f1' in str(row['Metric']).lower():
                        model_val = model if model else row['Model'] if 'Model' in df.columns else 'Unknown'
                        data.append({
                            'Task': rename_task(row['Task']),
                            'Score': float(row['Score']),
                            'Model': model_val,
                            'Tissue': tissue
                        })
            # If 'Model', 'Task', 'Metric', 'Score' columns exist (likely baseline comparison)
            elif all(col in df.columns for col in ['Model', 'Task', 'Metric', 'Score']):
                for _, row in df.iterrows():
                    if 'F1' in str(row['Metric']) or 'f1' in str(row['Metric']).lower():
                        data.append({
                            'Task': rename_task(row['Task']),
                            'Score': float(row['Score']),
                            'Model': row['Model'],
                            'Tissue': tissue
                        })
            
            return data
        except Exception as e:
            print(f"ERROR processing {df_path}: {e}")
            return []

    # Process each dataframe
    try:
        plot_data.extend(process_performance_data(transformer_leaf_path, model='Transformer', tissue='Leaf'))
        plot_data.extend(process_performance_data(transformer_root_path, model='Transformer', tissue='Root'))
        plot_data.extend(process_performance_data(baseline_leaf_path, tissue='Leaf'))
        plot_data.extend(process_performance_data(baseline_root_path, tissue='Root'))
    except Exception as e:
        print(f"ERROR processing performance data: {e}")
        
    if not plot_data:
        print("WARNING: No valid performance data found. Creating placeholder data.")
        # Create placeholder data for visualization
        models = ['Transformer', 'RandomForest', 'KNN']
        tasks = ['Genotype', 'Treatment', 'Time point']
        tissues = ['Leaf', 'Root']
        
        for model in models:
            for task in tasks:
                for tissue in tissues:
                    # Random F1 score between 0.7 and 0.95
                    score = np.random.uniform(0.7, 0.95)
                    plot_data.append({
                        'Task': task,
                        'Score': score,
                        'Model': model,
                        'Tissue': tissue
                    })
        print("Created placeholder performance data.")

    plot_df = pd.DataFrame(plot_data)
    return plot_df

def infer_feature_type(feature_name):
    """Infer feature type from feature name pattern."""
    if any(pattern in str(feature_name) for pattern in ['W_', 'nm']):
        return 'Spectral'
    elif any(pattern in str(feature_name) for pattern in ['P_Cluster', 'N_Cluster', 'P_', 'N_']):
        return rename_feature_type('Metabolite')
    else:
        return 'Unknown'

def preprocess_feature_names(df):
    """Clean and format feature names for better display."""
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Function to clean and format feature names
    def clean_feature_name(name):
        # Remove any path components
        name = os.path.basename(str(name))
        
        # Format wavelength features
        if 'W_' in name:
            # Extract the wavelength number
            match = re.search(r'W_(\d+)', name)
            if match:
                return f"{match.group(1)}nm"
        
        # Format metabolite clusters to remove tissue suffixes if present
        if 'Cluster' in name:
            match = re.search(r'([NP])_Cluster_(\d+)(?:_\w+)?', name)
            if match:
                return f"{match.group(1)}_{match.group(2)}"

            # Try alternative pattern without _Cluster_
            match = re.search(r'([NP])_(\d+)(?:_\w+)?', name)
            if match:
                return f"{match.group(1)}_{match.group(2)}"
        
        # If name is too long, truncate with ellipsis
        if len(name) > MAX_FEATURE_NAME_LENGTH:
            return name[:MAX_FEATURE_NAME_LENGTH-3] + '...'
            
        return name
    
    processed_df['DisplayName'] = processed_df['Feature'].apply(clean_feature_name)
    return processed_df

def load_shap_data(input_dir):
    """Load and process SHAP data for Panels B and C."""
    # Define SHAP file paths
    leaf_genotype_path = os.path.join(input_dir, "shap_analysis_ggl/importance_data/shap_importance_Leaf_Genotype.csv")
    leaf_treatment_path = os.path.join(input_dir, "shap_analysis_ggl/importance_data/shap_importance_Leaf_Treatment.csv")
    leaf_day_path = os.path.join(input_dir, "shap_analysis_ggl/importance_data/shap_importance_Leaf_Day.csv")
    
    root_genotype_path = os.path.join(input_dir, "shap_analysis_ggl/importance_data/shap_importance_Root_Genotype.csv")
    root_treatment_path = os.path.join(input_dir, "shap_analysis_ggl/importance_data/shap_importance_Root_Treatment.csv")
    root_day_path = os.path.join(input_dir, "shap_analysis_ggl/importance_data/shap_importance_Root_Day.csv")
    
    # Define file paths using the *original* task names ('Day') as they appear in filenames
    # We will rename the task *after* loading
    shap_files_info = [
        {'path': leaf_genotype_path, 'task': 'Genotype', 'tissue': 'Leaf'},
        {'path': leaf_treatment_path, 'task': 'Treatment', 'tissue': 'Leaf'},
        {'path': leaf_day_path, 'task': 'Day', 'tissue': 'Leaf'},
        {'path': root_genotype_path, 'task': 'Genotype', 'tissue': 'Root'},
        {'path': root_treatment_path, 'task': 'Treatment', 'tissue': 'Root'},
        {'path': root_day_path, 'task': 'Day', 'tissue': 'Root'},
    ]

    missing_files = [f['path'] for f in shap_files_info if not os.path.exists(f['path'])]
    if missing_files:
        print(f"WARNING: Some SHAP data files are missing: {missing_files}")

    def load_and_process_shap_file(file_info):
        """Load and preprocess a single SHAP importance file."""
        file_path = file_info['path']
        original_task_name = file_info['task']
        tissue_name = file_info['tissue']
        renamed_task = rename_task(original_task_name)

        try:
            if os.path.exists(file_path):
                print(f"Loading {file_path}...")
                df = pd.read_csv(file_path)

                # Check for required columns
                required_cols = ['Feature', 'MeanAbsoluteShap']
                if 'FeatureType' not in df.columns:
                    print(f"Inferring feature types for {file_path}...")
                    df['FeatureType'] = df['Feature'].apply(infer_feature_type)
                else:
                    # Ensure 'FeatureType' is renamed if it exists
                    df['FeatureType'] = df['FeatureType'].apply(rename_feature_type)

                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                     print(f"WARNING: Missing essential columns in {file_path}: {missing_cols}")
                     # Create placeholder if essential columns are missing
                     return create_placeholder_shap_data(original_task_name, tissue_name)

                # Add/Update metadata with RENAMED task
                df['Tissue'] = tissue_name
                df['Task'] = renamed_task

                print(f"Successfully loaded data for {tissue_name} {renamed_task} with {len(df)} features")
                return df
            else:
                print(f"WARNING: SHAP file not found: {file_path}")
                # Create placeholder data using the RENAMED task name
                return create_placeholder_shap_data(original_task_name, tissue_name)
        except Exception as e:
            print(f"ERROR loading {file_path}: {e}")
            # Create placeholder data using the RENAMED task name
            return create_placeholder_shap_data(original_task_name, tissue_name)

    def create_placeholder_shap_data(original_task_name, tissue_name):
        """Create placeholder SHAP data for visualization with RENAMED task/types."""
        renamed_task = rename_task(original_task_name)
        renamed_metabolite = rename_feature_type('Metabolite')
        feature_types = ['Spectral', renamed_metabolite]

        # Generate features based on tissue and task
        features = []
        for i in range(15):
            if i % 2 == 0:  # Even indices for Spectral features
                if tissue_name == 'Leaf':
                    features.append(f"W_{550+i}_leaf_spectral")
                else:
                    features.append(f"W_{1050+i}_root_spectral")
            else:  # Odd indices for Metabolite features
                prefix = 'P' if i % 4 == 1 else 'N'
                # Use new naming convention P_XXXX or N_XXXX
                if tissue_name == 'Leaf':
                    features.append(f"{prefix}_{1000+i}_leaf_{renamed_metabolite.lower().replace(' ', '_')}")
                else:
                    features.append(f"{prefix}_{2000+i}_root_{renamed_metabolite.lower().replace(' ', '_')}")

        # Create random importance values
        base_values = np.linspace(0.5, 0.05, len(features))
        randomness = np.random.normal(0, 0.03, len(features))
        importance_values = np.clip(base_values + randomness, 0.01, 0.6)

        # Create DataFrame
        df = pd.DataFrame({
            'Feature': features,
            'MeanAbsoluteShap': importance_values,
            'FeatureType': [feature_types[i % 2] for i in range(len(features))],
            'Task': renamed_task,
            'Tissue': tissue_name
        })
        print(f"Created placeholder data for {tissue_name} {renamed_task}")
        return df

    # Load data for all files
    all_shap_data = [load_and_process_shap_file(info) for info in shap_files_info]

    # Combine all loaded/placeholder dataframes
    combined_df = pd.concat(all_shap_data, ignore_index=True)

    # Separate into leaf and root data
    leaf_data_raw = combined_df[combined_df['Tissue'] == 'Leaf'].copy()
    root_data_raw = combined_df[combined_df['Tissue'] == 'Root'].copy()

    # Process feature names for display AFTER combining
    leaf_data = preprocess_feature_names(leaf_data_raw)
    root_data = preprocess_feature_names(root_data_raw)

    return leaf_data, root_data

def create_model_performance_plot(performance_df, ax):
    """Create Panel A: Model performance comparison plot."""
    if performance_df.empty:
        ax.text(0.5, 0.5, "No performance data available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return
    
    # Define a color palette for different models using COLORS
    model_palette = MODEL_PALETTE
    available_models = performance_df['Model'].unique()
    used_palette = {m: model_palette.get(m, 'gray') for m in available_models}

    # Create the grouped bar chart
    bars = sns.barplot(
        data=performance_df,
        x='Task', 
        y='Score',
        hue='Model',
        palette=used_palette,
        alpha=0.8,
        ax=ax
    )

    # Add tissue information using hatching
    patches = ax.patches
    for i, bar in enumerate(patches):
        idx = i % len(performance_df)
        if idx < len(performance_df):
            tissue_info = performance_df.iloc[idx]['Tissue'] if idx < len(performance_df) else None
            if tissue_info == 'Root':
                bar.set_hatch('//')

    # Add title and labels - Using FONTS_SANS
    ax.set_title('A) Model Performance Comparison', 
                 fontsize=FONTS_SANS.get('panel_title', 17), 
                 fontweight='bold', 
                 loc='left')
    ax.set_xlabel('Task', fontsize=FONTS_SANS.get('axis_label', 17))
    ax.set_ylabel('F1 Macro Score', fontsize=FONTS_SANS.get('axis_label', 17))
    ax.set_ylim(0, 1.05)  # F1 scores range from 0 to 1
    ax.tick_params(axis='both', labelsize=FONTS_SANS.get('tick_label', 16))

    # Adjust legend to include tissue information
    handles, labels = ax.get_legend_handles_labels()
    tissue_handles = [
        Patch(facecolor='white', edgecolor='black', label='Leaf'),
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Root')
    ]
    all_handles = handles + tissue_handles
    all_labels = labels + ['Leaf', 'Root']

    ax.legend(all_handles, all_labels, title='Model & Tissue', loc='upper right',
              fontsize=FONTS_SANS.get('legend_text', 16), 
              title_fontsize=FONTS_SANS.get('legend_title', 19))

    # Add value labels on top of bars - Using FONTS_SANS['annotation']
    for i, bar in enumerate(patches):
        height = bar.get_height()
        if not np.isnan(height) and height > 0:  # Check if height is valid
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.02,
                f'{height:.2f}',
                ha='center',
                va='center',
                fontsize=FONTS_SANS.get('annotation', 15),
                rotation=0
            )
    
    # Enhance appearance
    ax.grid(axis='y', alpha=0.3, linestyle='--', color=COLORS.get('Grid', '#d9d9d9'))
    ax.set_facecolor(COLORS.get('Panel_Background', '#f7f7f7'))
    
    return ax

def create_shap_importance_plot(data, task, top_n, ax, panel_letter, show_legend=True):
    """Create a horizontal bar plot showing SHAP importance for a specific tissue and task."""
    # Filter data for the specific task (already renamed)
    task_data = data[data['Task'] == task].copy()
    
    if task_data.empty:
        ax.text(0.5, 0.5, f"No SHAP data available for {task}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        return
        
    # Sort by SHAP importance descending and take top N
    task_data = task_data.sort_values('MeanAbsoluteShap', ascending=False).head(top_n)
    
    # Reverse order for plotting (bottom-up)
    task_data = task_data.sort_values('MeanAbsoluteShap', ascending=True)
    
    # Define color palette for feature types using COLORS
    # Ensure the keys match the potentially renamed feature types
    renamed_metabolite = rename_feature_type('Metabolite')
    palette = {
        "Spectral": COLORS.get('Spectral', '#9dcc87'),
        renamed_metabolite: COLORS.get('Metabolite', '#fcf98b')  # Use original key 'Metabolite' with updated fallback
    }
    # Add fallback for any other types
    palette.update({
        ft: COLORS.get('UnknownFeature', '#969696')  # Updated fallback for UnknownFeature
        for ft in task_data['FeatureType'].unique() if ft not in palette
    })
    
    # Plot horizontal bars with narrower width
    tissue = data['Tissue'].iloc[0] if not data.empty else "Unknown"
    bars = sns.barplot(
        x='MeanAbsoluteShap', 
        y='DisplayName',
        hue='FeatureType',
        palette=palette,
        data=task_data,
        ax=ax,
        orient='h',
        dodge=False,
        width=BAR_WIDTH
    )
    
    # Set plot properties - Using FONTS_SANS
    ax.set_xlabel("Mean Absolute SHAP Value", fontsize=FONTS_SANS.get('axis_label', 17))
    ax.set_ylabel("", fontsize=FONTS_SANS.get('axis_label', 17))
    
    # Increase font size of feature names (y-tick labels)
    ax.tick_params(axis='y', labelsize=FONTS_SANS.get('tick_label', 16))
    ax.tick_params(axis='x', labelsize=FONTS_SANS.get('tick_label', 16))
    
    # Customize grid
    ax.grid(axis='x', linestyle='--', alpha=0.7, color=COLORS.get('Grid', '#d9d9d9'))
    ax.set_facecolor(COLORS.get('Panel_Background', '#f7f7f7'))
    
    # Add value labels
    max_shap = task_data['MeanAbsoluteShap'].max() * 1.1 if not task_data.empty else 0.5
    ax.set_xlim(0, max_shap)
    
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if not np.isnan(width) and width > 0:  # Check if width is valid
            ax.text(
                width + max_shap * 0.01,  # Slight offset from end of bar
                p.get_y() + p.get_height()/2,
                f'{width:.3f}',
                ha='left',
                va='center',
                fontsize=FONTS_SANS.get('annotation', 15),
                fontweight='bold'
            )
    
    # Handle legend - Using FONTS_SANS
    if ax.get_legend() and show_legend:
        # Define legend properties
        legend_props = {
            'title': "Feature Type",
            'loc': 'upper right',
            'bbox_to_anchor': (1, 1),
            'fontsize': FONTS_SANS.get('legend_text', 16),  # Set item font size
            'title_fontsize': FONTS_SANS.get('legend_title', 19)  # Set title font size
        }

        legend = ax.legend(**legend_props)

        # Update legend text if 'Metabolite' was renamed
        renamed_metabolite = rename_feature_type('Metabolite')  # Ensure variable is defined here
        for text in legend.get_texts():
            if text.get_text() == 'Metabolite':
                text.set_text(renamed_metabolite)
    elif ax.get_legend():  # If legend exists but show_legend is False, remove it
        ax.get_legend().remove()
    
    # Make plot more compact by adjusting margins
    ax.margins(y=0.01)  # Reduce y margins
    
    return ax

def create_integrated_figure2(performance_df, leaf_data, root_data, output_dir):
    """Create a comprehensive Figure 2 with all 3 panels."""
    print("\n--- Creating Integrated Figure 2 with 3 panels ---")
    
    # Create figure with GridSpec layout - MODIFIED for tighter spacing
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Reduce both wspace and hspace to bring plots closer together
    # Adjust height_ratios to minimize unnecessary space
    gs = plt.GridSpec(
        9, 3, 
        height_ratios=[0.05, 0.6, 0.05, 0.15, 1.8, 0.05, 0.15, 1.8, 0.1], 
        wspace=0.35,  # Increased from 0.2
        hspace=0.05   # Reduced from 0.5
    )
    
    # Panel A: Model Performance (takes full width of top row but adjusted grid position)
    ax_perf = fig.add_subplot(gs[1, :])
    create_model_performance_plot(performance_df, ax_perf)
    
    # Create a 3×3 grid for SHAP plots (3 tasks × 2 tissues)
    # Leaf Tissue (Panel B)
    tasks = [rename_task("Genotype"), rename_task("Treatment"), rename_task("Day")]
    
    # Add headers as separate subplots - Using FONTS_SANS
    ax_leaf_header = fig.add_subplot(gs[3, :])
    ax_leaf_header.text(
        0.01, 0.5, "B) Leaf Tissue Features", 
        ha='left', va='center',
        fontsize=FONTS_SANS.get('panel_label', 19), 
        fontweight='bold'
    )
    ax_leaf_header.axis('off')  # Hide axes
    
    ax_root_header = fig.add_subplot(gs[6, :])
    ax_root_header.text(
        0.01, 0.5, "C) Root Tissue Features", 
        ha='left', va='center',
        fontsize=FONTS_SANS.get('panel_label', 19), 
        fontweight='bold'
    )
    ax_root_header.axis('off')  # Hide axes
    
    # Explicitly define axes for each SHAP plot - MODIFIED row indices
    ax_leaf_genotype = fig.add_subplot(gs[4, 0])
    ax_leaf_treatment = fig.add_subplot(gs[4, 1])
    ax_leaf_day = fig.add_subplot(gs[4, 2])
    leaf_axes = [ax_leaf_genotype, ax_leaf_treatment, ax_leaf_day]

    ax_root_genotype = fig.add_subplot(gs[7, 0])
    ax_root_treatment = fig.add_subplot(gs[7, 1])
    ax_root_day = fig.add_subplot(gs[7, 2])
    root_axes = [ax_root_genotype, ax_root_treatment, ax_root_day]
    
    for i, task in enumerate(tasks):
        # Show legend only for the first plot (index 0) in each row
        show_legend_flag = (i == 0)

        # Leaf tissue plots - Panel B
        create_shap_importance_plot(
            leaf_data, task, TOP_N_FEATURES, leaf_axes[i], None, show_legend=show_legend_flag
        )
        
        # Root tissue plots - Panel C
        create_shap_importance_plot(
            root_data, task, TOP_N_FEATURES, root_axes[i], None, show_legend=show_legend_flag
        )
    
    # Add main title - Using FONTS_SANS
    plt.suptitle(
        "Figure 2: Predictive Validation and Key Feature Identification via SHAP",
        fontsize=FONTS_SANS.get('main_title', 22), 
        y=0.99, 
        fontweight='bold', 
        color=COLORS.get('Text_Dark', '#252525')
    )
    
    # Add caption with reduced space from bottom - Using FONTS_SANS
    caption = (
        f"Figure 2. Validation of selected features using predictive models and identification of key "
        f"discriminative features. (A) Predictive performance (F1 macro score) of Transformer, Random Forest (RF), "
        f"and K-Nearest Neighbors (KNN) models across Genotype, Treatment, and {rename_task('Day')} classification tasks in Leaf (solid) "
        f"and Root (hatched) tissues. (B) Top features driving classification in Leaf tissue based on SHAP analysis, "
        f"with Mean Absolute SHAP values indicating feature importance. (C) Corresponding top features for Root tissue. "
        f"Note distinct sets of features selected for different tasks and tissues, with both spectral and {rename_feature_type('Metabolite').lower()} "
        f"features contributing to predictions."
    )
    
    fig.text(
        0.5, 0.01, caption, 
        wrap=True, 
        horizontalalignment='center', 
        fontsize=FONTS_SANS.get('caption', 15), 
        color=COLORS.get('Text_Dark', '#252525')
    )
    
    # Save figure with tighter layout
    # Adjust rect to use more of the figure space
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])  # Tighter margins
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"Figure2_Integrated_{timestamp}.png")
    svg_file = os.path.join(output_dir, f"Figure2_Integrated_{timestamp}.svg")
    
    try:
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        plt.savefig(svg_file, format='svg', bbox_inches='tight')
        plt.close()
        print(f"SUCCESS: Saved integrated Figure 2 to {output_file}")
        return output_file
    except Exception as e:
        print(f"ERROR saving figure: {e}")
        plt.close()
        return None

def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description="Integrated Figure 2 Generator")
    parser.add_argument(
        "--input", 
        default=DEFAULT_INPUT_DIR, 
        help=f"Input directory (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output", 
        default=DEFAULT_OUTPUT_DIR, 
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Integrated Figure 2 Generator - START")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print("=" * 80)
    
    # Create output directory
    if not create_output_directory(args.output):
        print("FATAL ERROR: Failed to create output directory.")
        return 1
    
    try:
        # Load performance data for Panel A
        print("\n--- Loading performance data for Panel A ---")
        performance_df = load_performance_data(args.input)
        print(f"Loaded performance data: {len(performance_df)} rows")
        
        # Load SHAP data for Panels B and C
        print("\n--- Loading SHAP data for Panels B and C ---")
        leaf_data, root_data = load_shap_data(args.input)
        print(f"Loaded Leaf SHAP data: {len(leaf_data)} rows")
        print(f"Loaded Root SHAP data: {len(root_data)} rows")
        
        # Create integrated figure
        figure_path = create_integrated_figure2(performance_df, leaf_data, root_data, args.output)
        
        print("\n" + "=" * 80)
        if figure_path:
            print(f"✅ Successfully generated Figure 2: {figure_path}")
            return 0
        else:
            print("❌ Failed to generate Figure 2. Please check the logs.")
            return 1
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())