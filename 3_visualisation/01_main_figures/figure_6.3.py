"""
Figure 6: Integrated Temporal Trajectories Visualization

This script generates visualizations of temporal trajectories for leaf and root data,
comparing spectral and metabolite features between tolerant (G1) and susceptible (G2)
genotypes under stress conditions. The visualization focuses on features identified as
important through SHAP analysis and attention mechanisms from transformer models.

The plot shows how key features evolve over time, highlighting differential responses
between genotypes under stress treatment.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Color Definitions (Muted Blue/Green/Yellow/Grey Focus)
COLORS = {
    # Core Experimental Variables
    'G1': '#0db84b',             # Tolerant Genotype (Medium-Dark Blue)
    'G2': '#25aac2',             # Susceptible Genotype (Medium Teal)
    'T0': '#4682B4',             # Control Treatment (Medium Green)
    'T1': '#BDB76B',             # Stress Treatment (Muted Orange/Yellow)
    'Leaf': '#00FF00',           # Leaf Tissue (Darkest Green)
    'Root': '#40E0D0',           # Root Tissue (Darkest Blue)

    # Days (Subtle Yellow-Green sequence)
    'Day1': '#ffffcc',           # Very Light Yellow-Green
    'Day2': '#9CBA79',           # Light Yellow-Green
    'Day3': '#3e7d5a',           # Medium Yellow-Green

    # Data Types / Omics / Features
    'Spectral': '#ECDA79',       # General Spectral (Medium Blue)
    'Metabolite': '#84ab92',     # General Metabolite (Medium-Dark Yellow-Green)
    'UnknownFeature': '#B0E0E6', # Medium Grey for fallback

    # Specific Spectral Categories
    'Spectral_Water': '#6DCAFA',     # Medium-Dark Blue
    'Spectral_Pigment': '#00FA9A',   # Medium-Dark Green
    'Spectral_Structure': '#7fcdbb', # Medium Teal
    'Spectral_SWIR': '#636363',      # Dark Grey
    'Spectral_VIS': '#c2e699',       # Light Yellow-Green
    'Spectral_RedEdge': '#78c679',   # Medium Yellow-Green
    'Spectral_UV': '#00BFFF',        # Darkest Blue (Matches Root)
    'Spectral_Other': '#969696',     # Medium Grey

    # Specific Metabolite Categories
    'Metabolite_PCluster': '#3DB3BF', # Darkest Yellow-Green
    'Metabolite_NCluster': '#ffffd4', # Very Light Yellow
    'Metabolite_Other': '#bdbdbd',    # Light Grey

    # Methods & Model Comparison
    'MOFA': '#FFEBCD',           # Dark Blue
    'SHAP': '#F0E68C',           # Dark Green
    'Overlap': '#AFEEEE',        # Medium-Dark Yellow-Green
    'Transformer': '#fae3a2',    # Medium Blue
    'RandomForest': '#40E0D0',   # Medium Green
    'KNN': '#729c87',            # Medium Teal

    # Network Visualization Elements
    'Edge_Low': '#f0f0f0',         # Very Light Gray
    'Edge_High': '#EEE8AA',        # Dark Blue
    'Node_Spectral': '#6baed6',    # Default Spectral Node (Medium Blue)
    'Node_Metabolite': '#FFC4A1',  # Default Metabolite Node (Med-Dark Yellow-Green)
    'Node_Edge': '#252525',        # Darkest Gray / Near Black border

    # Statistical & Difference Indicators
    'Positive_Diff': '#66CDAA',    # Medium-Dark Green
    'Negative_Diff': '#fe9929',    # Muted Orange/Yellow (Matches T1)
    'Significance': '#08519c',     # Dark Blue (for markers/text)
    'NonSignificant': '#bdbdbd',   # Light Grey
    'Difference_Line': '#636363',  # Dark Grey line

    # Plot Elements & Annotations
    'Background': '#FFFFFF',       # White plot background
    'Panel_Background': '#f7f7f7', # Very Light Gray background for panels
    'Grid': '#d9d9d9',             # Lighter Gray grid lines
    'Text_Dark': '#252525',        # Darkest Gray / Near Black text
    'Text_Light': '#FFFFFF',       # White text
    'Text_Annotation': '#000000',  # Black text for annotations
    'Annotation_Box_BG': '#FFFFFF', # White background for text boxes
    'Annotation_Box_Edge': '#bdbdbd', # Light Grey border for text boxes
    'Table_Header_BG': '#deebf7',   # Very Light Blue table header
    'Table_Highlight_BG': '#fff7bc', # Pale Yellow for highlighted table cells

    # Temporal Patterns (Fig S6)
    'Pattern_Increasing': '#238b45', # Medium-Dark Green
    'Pattern_Decreasing': '#fe9929', # Muted Orange/Yellow
    'Pattern_Peak': '#78c679',       # Medium Yellow-Green
    'Pattern_Valley': '#6baed6',     # Medium Blue
    'Pattern_Stable': '#969696',     # Medium Grey
}

# Font Scheme
FONTS_SANS = {
    'family': 'sans-serif',
    'sans_serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'main_title': 22,         # Figure Title (e.g., "Figure 1: ...")
    'panel_label': 19,        # Panel Labels (e.g., "A)", "B)")
    'panel_title': 17,        # Title for each subplot/panel
    'axis_label': 17,         # X and Y axis labels
    'tick_label': 16,         # Axis tick numbers/text
    'legend_title': 19,       # Title of the legend box
    'legend_text': 16,        # Text for individual legend items
    'annotation': 15,         # Text annotations within the plot area
    'caption': 15,            # Figure caption text
    'table_header': 15,       # Text in table headers
    'table_cell': 15,         # Text in table cells
}

# Output directory
output_dir = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot"
os.makedirs(output_dir, exist_ok=True)

# Define base paths
base_path = r"C:\Users\ms\Desktop\hyper"
mofa_path = os.path.join(base_path, "output", "mofa")
transformer_path = os.path.join(base_path, "output", "transformer")

def find_file(file_path):
    """Check if file exists and return path or None"""
    if os.path.exists(file_path):
        return file_path
    else:
        return None

# Define file paths - using transformer input files from MOFA output first, 
# fall back to original data files if needed
data_files = {
    'leaf_spectral': find_file(os.path.join(mofa_path, "transformer_input_leaf_spectral.csv")) or 
                     find_file(r"C:/Users/ms/Desktop/hyper/data/hyper_l_w_augmt.csv"),
    'leaf_metabolite': find_file(os.path.join(mofa_path, "transformer_input_leaf_metabolite.csv")) or 
                       find_file(r"C:/Users/ms/Desktop/hyper/data/n_p_l2_augmt.csv"),
    'root_spectral': find_file(os.path.join(mofa_path, "transformer_input_root_spectral.csv")) or 
                     find_file(r"C:/Users/ms/Desktop/hyper/data/hyper_r_w_augmt.csv"),
    'root_metabolite': find_file(os.path.join(mofa_path, "transformer_input_root_metabolite.csv")) or 
                       find_file(r"C:/Users/ms/Desktop/hyper/data/n_p_r2_augmt.csv")
}

# SHAP importance files
shap_files = {
    'leaf_genotype': find_file(os.path.join(transformer_path, "shap_analysis_ggl", 
                               "importance_data", "shap_importance_Leaf_Genotype.csv")),
    'leaf_treatment': find_file(os.path.join(transformer_path, "shap_analysis_ggl", 
                                "importance_data", "shap_importance_Leaf_Treatment.csv")),
    'leaf_day': find_file(os.path.join(transformer_path, "shap_analysis_ggl", 
                          "importance_data", "shap_importance_Leaf_Day.csv")),
    'root_genotype': find_file(os.path.join(transformer_path, "shap_analysis_ggl", 
                               "importance_data", "shap_importance_Root_Genotype.csv")),
    'root_treatment': find_file(os.path.join(transformer_path, "shap_analysis_ggl", 
                                "importance_data", "shap_importance_Root_Treatment.csv")),
    'root_day': find_file(os.path.join(transformer_path, "shap_analysis_ggl", 
                          "importance_data", "shap_importance_Root_Day.csv"))
}

# Attention data files
attention_files = {
    'leaf_pairs': find_file(os.path.join(transformer_path, "v3_feature_attention", 
                            "processed_attention_data_leaf", 
                            "processed_top_500_pairs_overall_Leaf.csv")),
    'root_pairs': find_file(os.path.join(transformer_path, "v3_feature_attention", 
                            "processed_attention_data_root", 
                            "processed_top_500_pairs_overall_Root.csv"))
}

# Load data
data = {}
metadata_cols = ['Row_names', 'Vac_id', 'Genotype', 'Entry', 'Tissue.type', 'Batch', 'Treatment', 'Replication', 'Day']

for key, filepath in data_files.items():
    if filepath:
        try:
            data[key] = pd.read_csv(filepath)
        except Exception as e:
            data[key] = None

# Load SHAP importance data
shap_data = {}
for key, filepath in shap_files.items():
    if filepath:
        try:
            shap_data[key] = pd.read_csv(filepath)
        except Exception as e:
            shap_data[key] = None

# Load attention data
attention_data = {}
for key, filepath in attention_files.items():
    if filepath:
        try:
            attention_data[key] = pd.read_csv(filepath)
        except Exception as e:
            attention_data[key] = None

# Function to select features from SHAP data
def get_shap_features(tissue, task, feature_type, n=2):
    """Get top SHAP features for a specific tissue, task, and feature type"""
    key = f"{tissue}_{task}"
    if key not in shap_data or shap_data[key] is None:
        return []
    
    df = shap_data[key]
    
    # Check if we have FeatureType column to filter by type
    if 'FeatureType' in df.columns:
        df = df[df['FeatureType'] == feature_type]
    else:
        # Try to infer feature type from name
        if feature_type == 'Spectral':
            df = df[df['Feature'].str.startswith('W_', na=False)]
        elif feature_type == 'Metabolite':
            df = df[df['Feature'].str.contains('Cluster', na=False)]
    
    # Sort by importance
    if 'MeanAbsoluteShap' in df.columns:
        df = df.sort_values('MeanAbsoluteShap', ascending=False)
    
    # Get top features
    features = df['Feature'].head(n).tolist()
    return features

# Function to select features from attention data
def get_attention_features(tissue, feature_type, n=2):
    """Get top features based on attention scores"""
    key = f"{tissue}_pairs"
    if key not in attention_data or attention_data[key] is None:
        return []
    
    df = attention_data[key]
    
    # Determine feature column based on type
    if feature_type.lower() == 'spectral':
        if 'Spectral_Feature' in df.columns:
            feature_col = 'Spectral_Feature'
        else:
            return []
    else:  # metabolite
        if 'Metabolite_Feature' in df.columns:
            feature_col = 'Metabolite_Feature'
        else:
            return []
    
    # Find attention score column
    attention_col = None
    for col in df.columns:
        if 'Attention' in col or 'attention' in col:
            attention_col = col
            break
    
    if attention_col is None:
        return []
    
    # Sort by attention score
    df = df.sort_values(attention_col, ascending=False)
    
    # Get unique top features 
    top_features = df[feature_col].unique()[:n].tolist()
    return top_features

# Features from network analysis in Figs 9-11
document_features = {
    'leaf_spectral': ['W_584', 'W_561', 'W_637'],
    'leaf_metabolite': ['N_1909', 'N_3029', 'P_0816'],
    'root_spectral': ['W_1092', 'W_1096', 'W_1072'],
    'root_metabolite': ['N_0512', 'N_1234', 'P_0816']
}

# Feature selection strategy:
# 1. Try SHAP-based selection for genotype (most relevant for the paper's focus)
# 2. If that fails, try attention-based selection
# 3. If that fails, use document-mentioned features
# 4. If none of those work, use available features in the data

selected_features = {}

# Try to select features using cascade of methods
for data_type in ['leaf_spectral', 'leaf_metabolite', 'root_spectral', 'root_metabolite']:
    tissue = data_type.split('_')[0]
    feature_type = 'Spectral' if 'spectral' in data_type else 'Metabolite'
    
    # First try SHAP for genotype prediction (main focus of paper)
    if f"{tissue}_genotype" in shap_data and shap_data[f"{tissue}_genotype"] is not None:
        selected_features[data_type] = get_shap_features(tissue, 'genotype', feature_type, n=3)
    # Then try attention-based selection
    elif f"{tissue}_pairs" in attention_data and attention_data[f"{tissue}_pairs"] is not None:
        selected_features[data_type] = get_attention_features(tissue, feature_type.lower(), n=3)
    # Then try document-mentioned features
    elif data_type in document_features:
        if data_type in data and data[data_type] is not None:
            existing_features = [f for f in document_features[data_type] 
                                if f in data[data_type].columns]
            if existing_features:
                selected_features[data_type] = existing_features
            else:
                # Last resort: pick features from available data
                if 'spectral' in data_type:
                    available_features = [col for col in data[data_type].columns 
                                         if col.startswith('W_')]
                else:
                    available_features = [col for col in data[data_type].columns 
                                         if 'Cluster' in col]
                
                if available_features:
                    selected_features[data_type] = sorted(available_features)[:3]
                else:
                    selected_features[data_type] = []
        else:
            selected_features[data_type] = []
    else:
        selected_features[data_type] = []

# Filter to stress treatment (T1) to focus on stress response patterns
for key in data:
    if data[key] is not None and 'Treatment' in data[key].columns:
        # Try different possible encodings of treatment
        for treatment_val in ['T1', 1, '1']:
            stress_data = data[key][data[key]['Treatment'] == treatment_val]
            if not stress_data.empty:
                data[key] = stress_data
                break

# Enhanced plotting function with better styling
def plot_feature_trends(ax, df, features, tissue, data_type):
    """Plot temporal feature trends (G1 vs G2) for spectral or molecular features"""
    if df is None or df.empty or not features:
        ax.text(0.5, 0.5, f"No data available for {tissue} {data_type}", 
                ha='center', va='center')
        ax.set_title(f"{tissue} {data_type} Features")
        return
    
    # Check which features actually exist in the data
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        ax.text(0.5, 0.5, f"None of the selected features found in {tissue} {data_type} data", 
                ha='center', va='center')
        ax.set_title(f"{tissue} {data_type} Features")
        return

    for i, feature in enumerate(available_features):
        # Try different encodings of genotype (string vs numeric)
        for genotype_name, genotype_values, linestyle, marker in [
            ('G1', ['G1', 1, '1'], '-', 'o'),  # Tolerant genotype
            ('G2', ['G2', 2, '2'], '--', '^')  # Susceptible genotype
        ]:
            # Try each possible encoding
            for genotype_val in genotype_values:
                subset = df[df['Genotype'] == genotype_val]
                if not subset.empty:
                    # Found data with this encoding
                    break
            
            if subset.empty:
                continue
            
            # Group by day and calculate statistics
            day_groups = subset.groupby('Day')
            means = day_groups[feature].mean()
            sem = day_groups[feature].sem()
            
            # Rename cluster features for display in legend
            display_feature_name = feature.replace('N_Cluster_', 'N_').replace('P_Cluster_', 'P_')
            
            # Plot
            ax.errorbar(
                means.index,
                means.values,
                yerr=sem.values,
                linestyle=linestyle,
                marker=marker,
                color=COLORS[genotype_name],
                label=f"{genotype_name} - {display_feature_name}",
                alpha=0.8,
                capsize=3
            )
    
    # Determine title based on data type
    plot_data_type_name = "Molecular Feature" if data_type == "Metabolite" else data_type

    # Construct title string, avoiding double "Features"
    title_string = f"{tissue} {plot_data_type_name}"
    if "feature" not in plot_data_type_name.lower():
        title_string += " Features"
    title_string += " Under Stress (T1)"
    ax.set_title(title_string, fontweight='bold')

    ax.set_xlabel("Time point")
    ax.set_ylabel("Feature Value")
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to be just the day values present
    all_days = set()
    for feature in available_features:
        for genotype_val in ['G1', 'G2', 1, 2, '1', '2']:
            subset = df[df['Genotype'] == genotype_val]
            if not subset.empty:
                days = sorted(subset['Day'].unique())
                all_days.update(days)
    
    if all_days:
        ax.set_xticks(sorted(all_days))
    
    # Add legend with better styling
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            fontsize=FONTS_SANS['legend_text'],
            loc='upper left',
            framealpha=0.6,
            ncol=1
        )

# Apply styling for the plot
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # For newer matplotlib versions
except:
    try:
        plt.style.use('seaborn')  # Fallback option
    except:
        pass  # Use default matplotlib style

# Apply font settings from FONTS_SANS
plt.rcParams.update({
    'font.family': FONTS_SANS['family'],
    'font.sans-serif': FONTS_SANS['sans_serif'],
    'axes.titlesize': FONTS_SANS['panel_title'],
    'axes.labelsize': FONTS_SANS['axis_label'],
    'xtick.labelsize': FONTS_SANS['tick_label'],
    'ytick.labelsize': FONTS_SANS['tick_label'],
    'legend.fontsize': FONTS_SANS['legend_text'],
    'figure.titlesize': FONTS_SANS['main_title']
})

# Create a wider, shorter figure with 1x4 layout (1 row, 4 columns)
fig, axes = plt.subplots(1, 4, figsize=(28, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.0, bottom=0.15)

# Plot each panel
plot_feature_trends(axes[0], data.get('leaf_spectral'), 
                    selected_features.get('leaf_spectral', []), 'Leaf', 'Spectral')
plot_feature_trends(axes[1], data.get('leaf_metabolite'), 
                    selected_features.get('leaf_metabolite', []), 'Leaf', 'Metabolite')
plot_feature_trends(axes[2], data.get('root_spectral'), 
                    selected_features.get('root_spectral', []), 'Root', 'Spectral')
plot_feature_trends(axes[3], data.get('root_metabolite'), 
                    selected_features.get('root_metabolite', []), 'Root', 'Metabolite')

# Add panel labels (J, K, L, M)
panel_labels = ["J)", "K)", "L)", "M)"]
for i, ax in enumerate(axes):
    ax.text(-0.1, 1.05, panel_labels[i],
            transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'],
            fontweight='bold',
            va='top', ha='right')

# Save figure with timestamp for version tracking
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(output_dir, f"fig15_temporal_evolution_{timestamp}.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')

# Also save with standard name for the paper
standard_output = os.path.join(output_dir, "fig15_temporal_evolution.png")
plt.savefig(standard_output, dpi=300, bbox_inches='tight')

plt.close()