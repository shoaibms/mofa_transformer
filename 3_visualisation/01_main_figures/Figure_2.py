#!/usr/bin/env python
# -- coding: utf-8 --
"""
Figure 2 Generation Script: Predictive Validation and Key Feature Identification via SHAP

This script creates a comprehensive Figure 2 with 3 panels:
Panel A: Bar chart comparing model performance (F1 Macro)
Panel B: SHAP top features bar plot for Leaf tissue
Panel C: SHAP top features bar plot for Root tissue

The script automatically looks for necessary input files in the expected project locations.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib.patches import Patch
from datetime import datetime
import argparse
import traceback

# Import custom colors
try:
    # Assuming colour.py is in the same directory or Python path
    from colour import COLORS, FONTS_SANS
except ImportError:
    print("WARNING: colour.py not found. Using fallback colors and fonts.")
    # Fallback colors if external configuration is not available
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
        'Spectral': "#2EB6E0",        # General Spectral (Medium Blue)
        'Metabolite': "#5be08c",       # General Metabolite (Medium-Dark Yellow-Green)
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

        'Transformer': '#fae3a2',     # Beige/Yellow
        'RandomForest': '#40E0D0',    # Teal
        'KNN': '#808080',             # Gray (to match figure)

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
        'panel_label': 19,        # Panel Labels (e.g., "A", "B")
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
DEFAULT_OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\figure"

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

# Model colors for Panel A sourced from central configuration
MODEL_PALETTE = {
    'Transformer': COLORS.get('Transformer', '#fae3a2'),
    'RandomForest': COLORS.get('RandomForest', '#40E0D0'),
    'KNN': COLORS.get('KNN', '#808080')
}

# Feature selection
TOP_N_FEATURES = 15          # Features per SHAP plot
MAX_FEATURE_NAME_LENGTH = 22  # Max length of feature names

# Figure dimensions
FIG_WIDTH = 20
FIG_HEIGHT = 19
DPI = 300

# Panel-specific settings using standardized fonts
BAR_WIDTH = 0.5 # Adjustment for bar width

# Rename map for display consistency
RENAME_MAP = {
    'Day': 'Time Point',
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
    """Load and process final v3 performance data for Panel A.

    Figure 2 must be generated only from final analysis outputs.
    Missing files or malformed files should stop figure generation.
    """
    transformer_leaf_path = os.path.join(
        input_dir, "v3_feature_attention/results/transformer_class_performance_Leaf.csv"
    )
    transformer_root_path = os.path.join(
        input_dir, "v3_feature_attention/results/transformer_class_performance_Root.csv"
    )
    baseline_leaf_path = os.path.join(
        input_dir, "v3_feature_attention/results/transformer_baseline_comparison_Leaf.csv"
    )
    baseline_root_path = os.path.join(
        input_dir, "v3_feature_attention/results/transformer_baseline_comparison_Root.csv"
    )

    files_to_check = [
        transformer_leaf_path,
        transformer_root_path,
        baseline_leaf_path,
        baseline_root_path
    ]
    missing_files = [f for f in files_to_check if not os.path.exists(f)]

    if missing_files:
        raise FileNotFoundError(
            "Required final Figure 2 performance files are missing:\n" +
            "\n".join(missing_files)
        )

    def process_performance_data(df_path, model=None, tissue=None):
        df = pd.read_csv(df_path)

        required_cols = ['Task', 'Metric', 'Score']
        if model is None:
            required_cols = ['Model', 'Task', 'Metric', 'Score']

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {df_path}: {missing_cols}"
            )

        data = []
        f1_rows = df[df['Metric'].astype(str).str.lower().str.contains('f1')]

        if f1_rows.empty:
            raise ValueError(f"No F1 metric rows found in {df_path}")

        for _, row in f1_rows.iterrows():
            model_val = model if model is not None else row['Model']
            data.append({
                'Task': rename_task(row['Task']),
                'Score': float(row['Score']),
                'Model': model_val,
                'Tissue': tissue
            })

        return data

    plot_data = []
    plot_data.extend(process_performance_data(transformer_leaf_path, model='Transformer', tissue='Leaf'))
    plot_data.extend(process_performance_data(transformer_root_path, model='Transformer', tissue='Root'))
    plot_data.extend(process_performance_data(baseline_leaf_path, tissue='Leaf'))
    plot_data.extend(process_performance_data(baseline_root_path, tissue='Root'))

    if not plot_data:
        raise ValueError(
            "No valid performance data found. Figure 2 must not be generated from synthetic data."
        )

    plot_df = pd.DataFrame(plot_data)
    plot_df['Model'] = plot_df['Model'].replace('KNN (k=5)', 'KNN')

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
        raise FileNotFoundError(
            "Required SHAP files for Figure 2B–C are missing:\n" +
            "\n".join(missing_files)
        )

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
                    raise ValueError(
                        f"Missing essential columns in {file_path}: {missing_cols}"
                    )

                # Add/Update metadata with RENAMED task
                df['Tissue'] = tissue_name
                df['Task'] = renamed_task

                print(f"Successfully loaded data for {tissue_name} {renamed_task} with {len(df)} features")
                return df
            else:
                raise FileNotFoundError(f"Required SHAP file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading SHAP file {file_path}: {e}")

    # Load data for all files
    all_shap_data = [load_and_process_shap_file(info) for info in shap_files_info]

    # Combine all loaded dataframes
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

    # Prepare data for plotting
    performance_df['Model_Tissue'] = performance_df['Model'] + '_' + performance_df['Tissue']
    model_order = ['Transformer', 'RandomForest', 'KNN']
    tissue_order = ['Leaf', 'Root']
    hue_order = [f"{m}_{t}" for m in model_order for t in tissue_order]
    
    # Define a palette that colors bars by Model
    model_palette = MODEL_PALETTE
    full_palette = {hue: model_palette.get(hue.split('_')[0], 'gray') for hue in hue_order}

    # Create the grouped bar chart
    sns.barplot(
        data=performance_df,
        x='Task',
        y='Score',
        hue='Model_Tissue',
        hue_order=hue_order,
        palette=full_palette,
        alpha=0.98,
        ax=ax
    )

    # Apply hatching based on tissue type
    plotted_hues = [label.get_text() for label in ax.get_legend().get_texts()] if ax.get_legend() else []
    num_plotted_hues = len(plotted_hues)
    if num_plotted_hues > 0:
        for i, bar in enumerate(ax.patches):
            hue_label_for_bar = plotted_hues[i % num_plotted_hues]
            if '_Root' in hue_label_for_bar:
                bar.set_hatch('//')

    # Remove the default seaborn legend before creating the custom one
    if ax.get_legend():
        ax.get_legend().remove()

    # Create custom handles for the legend
    model_handles = [Patch(facecolor=model_palette.get(m, 'gray'), label=m) for m in model_order]
    tissue_handles = [
        Patch(facecolor='white', edgecolor='black', label='Leaf'),
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Root')
    ]
    all_handles = model_handles + tissue_handles

    # Place the legend above the plot in a single horizontal row
    ax.legend(handles=all_handles, title='',
              loc='lower center', bbox_to_anchor=(0.5, 1.10),
              ncol=len(all_handles),
              fontsize=FONTS_SANS.get('legend_text', 16),
              title_fontsize=FONTS_SANS.get('legend_title', 19),
              frameon=False, columnspacing=1.5, handletextpad=0.5)

    # Add title and labels
    ax.set_title('Model Performance Comparison', fontsize=FONTS_SANS.get('panel_title', 18), fontweight='bold', loc='center', pad=55)
    ax.text(-0.1, 1.1, 'A', transform=ax.transAxes,
            fontsize=FONTS_SANS.get('panel_label', 19), fontweight='bold', va='top', ha='left')
    ax.set_xlabel('Task', fontsize=FONTS_SANS.get('axis_label', 17))
    ax.set_ylabel('F1 Macro Score', fontsize=FONTS_SANS.get('axis_label', 17))
    ax.set_ylim(0, 1.2)
    ax.tick_params(axis='both', labelsize=FONTS_SANS.get('tick_label', 16))

    # Add value labels on top of bars
    for bar in ax.patches:
        height = bar.get_height()
        if not np.isnan(height) and height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=FONTS_SANS.get('annotation', 15),
                rotation=0
            )

    # Enhance appearance
    ax.grid(axis='y', alpha=0.5, linestyle='--', color=COLORS.get('Grid', '#d9d9d9'))
    ax.set_facecolor(COLORS.get('Panel_Background', '#f7f7f7'))
    
    return ax


def create_shap_importance_plot(data, task, top_n, ax, panel_letter, show_legend=True, show_ylabel=False, show_xlabel=True):
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
    renamed_metabolite = rename_feature_type('Metabolite')
    palette = {
        "Spectral": COLORS.get('Spectral', '#9dcc87'),
        renamed_metabolite: COLORS.get('Metabolite', "#2fb6ec")
    }
    palette.update({ft: COLORS.get('UnknownFeature', '#969696')
                    for ft in task_data['FeatureType'].unique() if ft not in palette})

    # Plot horizontal bars
    sns.barplot(
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

    # Set plot properties
    if show_xlabel:
        ax.set_xlabel("Mean Absolute SHAP Value", fontsize=FONTS_SANS.get('axis_label', 17))
    else:
        ax.set_xlabel("")
        
    if show_ylabel:
        ax.set_ylabel("Predictive Features", fontsize=FONTS_SANS.get('axis_label', 17))
    else:
        ax.set_ylabel("", fontsize=FONTS_SANS.get('axis_label', 17))
    ax.tick_params(axis='y', labelsize=FONTS_SANS.get('tick_label', 16))
    ax.tick_params(axis='x', labelsize=FONTS_SANS.get('tick_label', 16))

    # Customize grid and background
    ax.grid(axis='x', linestyle='--', alpha=0.7, color=COLORS.get('Grid', '#d9d9d9'))
    ax.set_facecolor(COLORS.get('Panel_Background', '#f7f7f7'))

    # Add value labels
    max_shap = task_data['MeanAbsoluteShap'].max() * 1.1 if not task_data.empty else 0.5
    ax.set_xlim(0, max_shap)

    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if not np.isnan(width) and width > 0:
            ax.text(
                width + max_shap * 0.01,
                p.get_y() + p.get_height()/2,
                f'{width:.3f}',
                ha='left',
                va='center',
                fontsize=FONTS_SANS.get('annotation', 15),
                fontweight='bold'
            )

    # Handle legend
    if ax.get_legend() and show_legend:
        legend_props = {
            'title': "Feature Type",
            'loc': 'upper right',
            'bbox_to_anchor': (1, 1),
            'fontsize': FONTS_SANS.get('legend_text', 16),
            'title_fontsize': FONTS_SANS.get('legend_title', 19)
        }
        legend = ax.legend(**legend_props)
        for text in legend.get_texts():
            if text.get_text() == 'Metabolite':
                text.set_text(renamed_metabolite)
    elif ax.get_legend():
        ax.get_legend().remove()

    ax.margins(y=0.01)
    return ax

def create_integrated_figure2(performance_df, leaf_data, root_data, output_dir):
    """Create a comprehensive Figure 2 with all 3 panels."""
    print("\n--- Creating Integrated Figure 2 with 3 panels ---")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    gs = plt.GridSpec(9, 3,
                     height_ratios=[0.05, 0.7, 0.05, 0.1, 1.2, 0.05, 0.1, 1.2, 0.1],
                     wspace=0.35,
                     hspace=0.4)

    # Panel A: Model Performance
    ax_perf = fig.add_subplot(gs[1, :])
    create_model_performance_plot(performance_df, ax_perf)

    # Panels B and C: SHAP plots
    tasks = [rename_task("Genotype"), rename_task("Treatment"), rename_task("Day")]

    ax_leaf_header = fig.add_subplot(gs[3, :])
    ax_leaf_header.text(0.5, 0.5, "Leaf Tissue Features", ha='center', va='center',
                      fontsize=FONTS_SANS.get('panel_label', 19), fontweight='bold')
    ax_leaf_header.text(0.01, 0.5, "B", ha='left', va='center',
                      fontsize=FONTS_SANS.get('panel_label', 19), fontweight='bold')
    ax_leaf_header.axis('off')

    ax_root_header = fig.add_subplot(gs[6, :])
    ax_root_header.text(0.5, 0.5, "Root Tissue Features", ha='center', va='center',
                      fontsize=FONTS_SANS.get('panel_label', 19), fontweight='bold')
    ax_root_header.text(0.01, 0.5, "C", ha='left', va='center',
                      fontsize=FONTS_SANS.get('panel_label', 19), fontweight='bold')
    ax_root_header.axis('off')

    leaf_axes = [fig.add_subplot(gs[4, i]) for i in range(3)]
    root_axes = [fig.add_subplot(gs[7, i]) for i in range(3)]

    for i, task in enumerate(tasks):
        show_legend_flag = (i == 0)
        show_ylabel_flag = (i == 0)
        show_xlabel_flag = (i == 1)
        leaf_axes[i].set_title(task, fontsize=FONTS_SANS.get('panel_title', 17))
        create_shap_importance_plot(leaf_data, task, TOP_N_FEATURES, leaf_axes[i], None, show_legend=show_legend_flag, show_ylabel=show_ylabel_flag, show_xlabel=show_xlabel_flag)
        
        root_axes[i].set_title(task, fontsize=FONTS_SANS.get('panel_title', 17))
        create_shap_importance_plot(root_data, task, TOP_N_FEATURES, root_axes[i], None, show_legend=show_legend_flag, show_ylabel=show_ylabel_flag, show_xlabel=show_xlabel_flag)

    # Adjust layout to make room for the legend on the right
    plt.tight_layout(rect=[0.01, 0.02, 0.90, 0.96])

    output_file = os.path.join(output_dir, "fig_2.png")
    svg_file = os.path.join(output_dir, "fig_2.svg")

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

def generate_table_s2(input_dir, output_dir):
    """Generate Supplementary Table S2 from final v3 Transformer output CSVs.

    This does NOT retrain the model. It only reads the final performance CSV
    files and writes a clean Table S2 (wide + long-format audit table) into
    the same output folder as Figure 2.

    Required files (under input_dir/v3_feature_attention/results/):
    - transformer_class_performance_Leaf.csv
    - transformer_class_performance_Root.csv
    - transformer_baseline_comparison_Leaf.csv
    - transformer_baseline_comparison_Root.csv
    """
    print("\n--- Generating Supplementary Table S2 ---")

    results_dir = os.path.join(input_dir, "v3_feature_attention", "results")

    output_wide = os.path.join(output_dir, "Supplementary_Table_S2_model_performance.csv")
    output_long = os.path.join(output_dir, "Supplementary_Table_S2_model_performance_long.csv")

    def read_transformer_file(path, tissue):
        """Read Transformer performance CSV and add model/tissue labels."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

        df = pd.read_csv(path)

        required = {"Task", "Metric", "Score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        df = df.copy()
        df["Tissue"] = tissue
        df["Model"] = "Transformer"

        return df[["Tissue", "Model", "Task", "Metric", "Score"]]

    def read_baseline_file(path, tissue):
        """Read baseline performance CSV and add tissue label."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

        df = pd.read_csv(path)

        required = {"Model", "Task", "Metric", "Score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        df = df.copy()
        df["Tissue"] = tissue

        return df[["Tissue", "Model", "Task", "Metric", "Score"]]

    files = {
        "transformer_leaf": os.path.join(results_dir, "transformer_class_performance_Leaf.csv"),
        "transformer_root": os.path.join(results_dir, "transformer_class_performance_Root.csv"),
        "baseline_leaf": os.path.join(results_dir, "transformer_baseline_comparison_Leaf.csv"),
        "baseline_root": os.path.join(results_dir, "transformer_baseline_comparison_Root.csv"),
    }

    combined = pd.concat(
        [
            read_transformer_file(files["transformer_leaf"], "Leaf"),
            read_transformer_file(files["transformer_root"], "Root"),
            read_baseline_file(files["baseline_leaf"], "Leaf"),
            read_baseline_file(files["baseline_root"], "Root"),
        ],
        ignore_index=True
    )

    # Harmonise labels for manuscript consistency
    combined["Model"] = combined["Model"].replace({
        "KNN (k=5)": "KNN"
    })

    combined["Task"] = combined["Task"].replace({
        "Day": "Time Point"
    })

    # Save full long-format audit table
    combined["Score"] = combined["Score"].astype(float).round(4)
    combined.to_csv(output_long, index=False)

    # Keep common metrics available for all models
    common_metrics = ["Accuracy", "F1_Macro"]
    common = combined[combined["Metric"].isin(common_metrics)].copy()

    # Generate manuscript-friendly wide table
    table_s2 = (
        common
        .pivot_table(
            index=["Tissue", "Task", "Model"],
            columns="Metric",
            values="Score",
            aggfunc="first"
        )
        .reset_index()
    )

    # Ensure expected columns exist
    for col in common_metrics:
        if col not in table_s2.columns:
            table_s2[col] = pd.NA

    table_s2 = table_s2[["Tissue", "Task", "Model", "Accuracy", "F1_Macro"]]

    # Sort cleanly
    tissue_order = {"Leaf": 0, "Root": 1}
    task_order = {"Genotype": 0, "Treatment": 1, "Time Point": 2}
    model_order = {"Transformer": 0, "RandomForest": 1, "KNN": 2}

    table_s2["_tissue_order"] = table_s2["Tissue"].map(tissue_order)
    table_s2["_task_order"] = table_s2["Task"].map(task_order)
    table_s2["_model_order"] = table_s2["Model"].map(model_order)

    table_s2 = (
        table_s2
        .sort_values(["_tissue_order", "_task_order", "_model_order"])
        .drop(columns=["_tissue_order", "_task_order", "_model_order"])
        .reset_index(drop=True)
    )

    table_s2["Accuracy"] = table_s2["Accuracy"].astype(float).round(4)
    table_s2["F1_Macro"] = table_s2["F1_Macro"].astype(float).round(4)

    table_s2.to_csv(output_wide, index=False)

    print("Supplementary Table S2 generated successfully.")
    print(f"Wide table: {output_wide}")
    print(f"Long audit table: {output_long}")

    return output_wide


def main():
    """Main function to run the visualization."""
    parser = argparse.ArgumentParser(description="Integrated Figure 2 Generator")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help=f"Input directory (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    print("=" * 80)
    print(f"Figure 2 Generator - Start")
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

        # Generate Supplementary Table S2 into the same output folder
        table_s2_path = None
        try:
            table_s2_path = generate_table_s2(args.input, args.output)
        except Exception as e:
            print(f"ERROR generating Supplementary Table S2: {e}")
            traceback.print_exc()

        print("\n" + "=" * 80)
        if figure_path:
            print(f"Successfully generated Figure 2: {figure_path}")
            if table_s2_path:
                print(f"Successfully generated Supplementary Table S2: {table_s2_path}")
            return 0
        else:
            print("Failed to generate Figure 2. Please check the logs.")
            return 1

    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
