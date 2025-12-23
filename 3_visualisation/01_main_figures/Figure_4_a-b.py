#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4, Panels A-B: Biological Systems Advantage Landscapes.

This script generates a visualization showing the coordination advantage of one
genotype (G1) over another (G2) across different biological systems and time
points. The visualization consists of two heatmaps (panels A and B) for leaf
and root tissues respectively.

The script performs the following steps:
1. Loads attention trend data for leaf and root tissues.
2. Calculates the advantage of G1 over G2 based on mean attention scores.
3. Determines statistical significance using the Mann-Whitney U test.
4. Categorizes spectral and metabolite features into biological systems.
5. Generates heatmap visualizations with significance overlays.
"""

import os
import warnings
from typing import Tuple, List, Optional, Dict, Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# === CONFIGURATION ===

# Figure dimensions
FIG_SIZE = (19, 7.5)

# Styling and Colors
COLORS = {
    'G1': '#00FA9A',              # Tolerant genotype
    'G2': '#48D1CC',              # Susceptible genotype
    'Advantage_High': '#006837',  # High G1 advantage
    'Advantage_Low': '#f7f7f7',   # Low/No advantage
    'Disadvantage': '#d73027',    # G2 advantage
    'Significance': '#252525',    # Significance markers
    'Background': '#FFFFFF',
    'Text_Dark': '#252525',
    'Grid': '#e5e5e5',
}

# Fonts and Typography
FONTS_SANS = {
    'family': 'sans-serif',
    'sans_serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'main_title': 22,
    'panel_label': 19,
    'panel_title': 15,
    'axis_label': 14,
    'tick_label': 14,
    'legend_title': 14,
    'legend_text': 14,
    'annotation': 14,
}

# Apply global plotting styles
plt.rcParams.update({
    'font.family': FONTS_SANS['family'],
    'font.sans-serif': FONTS_SANS['sans_serif'],
})

# === FILE PATHS ===
# Note: Paths are configured for the specific environment as requested.
OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot\test"
LEAF_TRENDS_PATH = os.path.join(OUTPUT_DIR, "processed_attention_trends_top_500_Leaf.csv")
ROOT_TRENDS_PATH = os.path.join(OUTPUT_DIR, "processed_attention_trends_top_500_Root.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_dummy_data() -> None:
    """
    Creates dummy data files for demonstration purposes if they do not exist.
    
    This function generates synthetic data for testing the visualization pipeline
    when the actual dataset is unavailable.
    """
    if os.path.exists(LEAF_TRENDS_PATH) and os.path.exists(ROOT_TRENDS_PATH):
        return

    print("Generating dummy data for demonstration...")
    
    spectral_features_leaf = [f"W_{w}" for w in range(500, 750, 50)]
    spectral_features_root = [f"W_{w}" for w in range(400, 2500, 100)]
    metabolite_features = [f"N_Cluster_{i}" for i in range(5)] + [f"P_Cluster_{i}" for i in range(5)]

    def _generate_file_data(spectral_features: List[str], file_path: str) -> None:
        data = []
        for spec_feat in spectral_features:
            for met_feat in metabolite_features:
                for day in [1, 2, 3]:
                    for genotype in ['1', '2']:
                        for treatment in ['1']:
                            # Simulate some advantage for G1
                            attention = np.random.rand() * 0.02 + (0.01 if genotype == '1' else 0)
                            row = {
                                "Spectral_Feature": spec_feat,
                                "Metabolite_Feature": met_feat,
                                "Genotype": genotype,
                                "Day": day,
                                "Treatment": treatment,
                                "Mean_Attention_S2M_Group_AvgHeads": attention
                            }
                            data.append(row)
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f"Created dummy file: {file_path}")

    _generate_file_data(spectral_features_leaf, LEAF_TRENDS_PATH)
    _generate_file_data(spectral_features_root, ROOT_TRENDS_PATH)


def load_and_preprocess_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads data from CSV files and applies preprocessing.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: A tuple containing
        the leaf and root DataFrames. Returns (None, None) if loading fails.
    """
    print("Loading and preprocessing data...")

    try:
        leaf_df = pd.read_csv(LEAF_TRENDS_PATH)
        root_df = pd.read_csv(ROOT_TRENDS_PATH)
        print(f"Data loaded successfully: Leaf {leaf_df.shape}, Root {root_df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Data Cleaning and Standardization
    for _, df in [("Leaf", leaf_df), ("Root", root_df)]:
        # Standardize column names
        df.columns = df.columns.str.strip()
        
        # Standardize Genotype labels
        df['Genotype'] = df['Genotype'].astype(str).replace({
            '1': 'G1', '2': 'G2', '1.0': 'G1', '2.0': 'G2'
        })

        # Standardize Time Point
        if 'Day' in df.columns:
            df['Time_Point'] = df['Day'].astype(int)
        elif 'Time_Point' in df.columns:
            df['Time_Point'] = df['Time_Point'].astype(int)

        # Standardize Treatment
        df['Treatment'] = df['Treatment'].astype(str).replace({
            '1': 'T1', '1.0': 'T1'
        })

        # Clean Metabolite Feature names
        if 'Metabolite_Feature' in df.columns:
            df['Metabolite_Feature'] = df['Metabolite_Feature'].str.replace('N_Cluster_', 'N_')
            df['Metabolite_Feature'] = df['Metabolite_Feature'].str.replace('P_Cluster_', 'P_')

    return leaf_df, root_df


def categorize_spectral_features(spectral_feature: Any) -> str:
    """
    Categorizes spectral features into biological regions based on wavelength.

    Args:
        spectral_feature: The spectral feature identifier (e.g., "W_450").

    Returns:
        str: The biological region category.
    """
    if pd.isna(spectral_feature):
        return 'Unknown'

    feature_str = str(spectral_feature)
    if 'W_' not in feature_str:
        return 'Other'

    try:
        wavelength = int(feature_str.replace('W_', ''))

        if 400 <= wavelength < 500:
            return 'Blue (400-500nm)'
        elif 500 <= wavelength < 600:
            return 'Green (500-600nm)'
        elif 600 <= wavelength < 700:
            return 'Red (600-700nm)'
        elif 700 <= wavelength < 800:
            return 'Red-Edge (700-800nm)'
        elif 800 <= wavelength < 1000:
            return 'NIR-1 (800-1000nm)'
        elif 1000 <= wavelength < 1300:
            return 'NIR-2 (1000-1300nm)'
        elif 1300 <= wavelength < 1800:
            return 'SWIR-1 (1300-1800nm)'
        elif 1800 <= wavelength <= 2500:
            return 'SWIR-2 (1800-2500nm)'
        else:
            return 'Extended Range'
    except (ValueError, TypeError):
        return 'Other'


def categorize_metabolite_features(metabolite_feature: Any) -> str:
    """
    Categorizes metabolite features into chemical classes.

    Args:
        metabolite_feature: The metabolite feature identifier.

    Returns:
        str: The metabolite class category.
    """
    if pd.isna(metabolite_feature):
        return 'Unknown'

    feature_str = str(metabolite_feature)

    if feature_str.startswith('N_'):
        return 'N-Metabolites'
    elif feature_str.startswith('P_'):
        return 'P-Metabolites'
    else:
        return 'Other Metabolites'


def calculate_advantage_matrix(leaf_df: pd.DataFrame, root_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the G1 advantage matrix across biological systems and time points.

    Advantage is defined as (G1 mean - G2 mean). Significance is determined
    using the Mann-Whitney U test.

    Args:
        leaf_df: DataFrame containing leaf data.
        root_df: DataFrame containing root data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the advantage matrix
        DataFrame and the boolean significance matrix DataFrame.
    """
    print("Calculating biological advantage matrix...")

    # Combine datasets with tissue labels
    leaf_df = leaf_df.copy()
    root_df = root_df.copy()
    leaf_df['Tissue'] = 'Leaf'
    root_df['Tissue'] = 'Root'

    combined_df = pd.concat([leaf_df, root_df], ignore_index=True)

    # Filter for specific treatment condition (T1)
    stress_df = combined_df[combined_df['Treatment'] == 'T1'].copy()

    # Add biological categories
    stress_df['Spectral_Category'] = stress_df['Spectral_Feature'].apply(categorize_spectral_features)
    stress_df['Metabolite_Category'] = stress_df['Metabolite_Feature'].apply(categorize_metabolite_features)

    # Create biological system combinations
    stress_df['Biological_System'] = (stress_df['Tissue'] + ' ' +
                                     stress_df['Spectral_Category'] + ' → ' +
                                     stress_df['Metabolite_Category'])

    # Initialize matrices
    biological_systems = sorted(stress_df['Biological_System'].unique())
    time_points = sorted(stress_df['Time_Point'].unique())

    print(f"Analyzing {len(biological_systems)} biological systems across {len(time_points)} time points...")

    advantage_data = []
    significance_data = []

    for bio_sys in biological_systems:
        sys_data = stress_df[stress_df['Biological_System'] == bio_sys]
        
        adv_row = []
        sig_row = []

        for tp in time_points:
            tp_data = sys_data[sys_data['Time_Point'] == tp]

            g1_values = tp_data[tp_data['Genotype'] == 'G1']['Mean_Attention_S2M_Group_AvgHeads']
            g2_values = tp_data[tp_data['Genotype'] == 'G2']['Mean_Attention_S2M_Group_AvgHeads']

            if len(g1_values) > 1 and len(g2_values) > 1:
                g1_mean = g1_values.mean()
                g2_mean = g2_values.mean()
                advantage = g1_mean - g2_mean

                try:
                    # Two-sided Mann-Whitney U test
                    _, p_value = mannwhitneyu(g1_values, g2_values, alternative='two-sided')
                    significant = p_value < 0.05
                except ValueError:
                    # Occurs if all values are identical
                    significant = False

                adv_row.append(advantage)
                sig_row.append(significant)
            else:
                adv_row.append(0.0)
                sig_row.append(False)

        advantage_data.append(adv_row)
        significance_data.append(sig_row)

    # Create DataFrames
    columns = [str(tp) for tp in time_points]
    advantage_df = pd.DataFrame(advantage_data, index=biological_systems, columns=columns)
    significance_df = pd.DataFrame(significance_data, index=biological_systems, columns=columns)

    print(f"Advantage matrix created with shape: {advantage_df.shape}")
    return advantage_df, significance_df


def create_custom_colormap() -> LinearSegmentedColormap:
    """
    Creates a custom colormap for visualizing G1 advantage.
    
    Returns:
        LinearSegmentedColormap: The custom colormap.
    """
    colors = [
        '#f7fcf0', "#cff9c3", "#99d78b", "#2fdd5b",
        '#7bccc4', "#7bf1fa", "#50aa75", "#107a39", "#0E6068"
    ]
    return LinearSegmentedColormap.from_list('g1_advantage_custom', colors, N=256)


def plot_single_landscape(ax: plt.Axes, 
                         advantage_df: pd.DataFrame, 
                         significance_df: pd.DataFrame, 
                         cmap: Any, 
                         norm: Any,
                         title: str, 
                         panel_label: str, 
                         tissue_name: str) -> None:
    """
    Plots an advantage landscape for a single tissue type.

    Args:
        ax: The matplotlib Axes object.
        advantage_df: DataFrame of advantage values.
        significance_df: DataFrame of boolean significance flags.
        cmap: Colormap to use.
        norm: Normalization for the colormap.
        title: Title for the subplot.
        panel_label: Label for the panel (e.g., 'a', 'b').
        tissue_name: Name of the tissue (e.g., 'LEAF', 'ROOT').
    """
    im = ax.imshow(advantage_df.values, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    # Overlay significance markers
    rows, cols = advantage_df.shape
    for i in range(rows):
        for j in range(cols):
            if significance_df.iloc[i, j]:
                ax.plot(j, i, 'o', markersize=5, markerfacecolor='white',
                       markeredgewidth=1.5, markeredgecolor='black', alpha=0.9)

    # Configure axes
    ax.set_xticks(range(cols))
    ax.set_xticklabels(advantage_df.columns, fontsize=FONTS_SANS['tick_label'])
    ax.set_xlabel('Time Points', fontsize=FONTS_SANS['axis_label'], fontweight='bold')

    # Simplify Y-axis labels
    simplified_labels = []
    for label in advantage_df.index:
        clean_label = label.replace(f'{tissue_name.title()} ', '')
        clean_label = clean_label.replace('-Metabolites', '-Met').replace('Metabolites', 'Met')
        simplified_labels.append(clean_label)

    ax.set_yticks(range(rows))
    ax.set_yticklabels(simplified_labels, fontsize=FONTS_SANS['tick_label'])
    ax.set_ylabel(f'{tissue_name.upper()} Systems (Spectral → Metabolite)',
                  fontsize=FONTS_SANS['axis_label'], fontweight='bold')

    # Set title and panel label
    ax.set_title(f'{title}\nAdvantage Landscape',
                 fontsize=FONTS_SANS['panel_title'], fontweight='bold', pad=15)
    ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold')

    # Highlight top advantages
    _highlight_top_advantages(ax, advantage_df)

    # Add legend to the first panel only
    if panel_label == 'a':
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                       markeredgecolor='black', markeredgewidth=1.5,
                       markersize=6, label='Significant (p < 0.05)', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                 fontsize=FONTS_SANS['legend_text'], framealpha=0.9)


def _highlight_top_advantages(ax: plt.Axes, advantage_df: pd.DataFrame) -> None:
    """
    Highlights the top 2 strongest advantages in the heatmap.

    Args:
        ax: The matplotlib Axes object.
        advantage_df: DataFrame of advantage values.
    """
    strongest_advantages = []
    for i, row_name in enumerate(advantage_df.index):
        if not advantage_df.iloc[i].empty:
            max_adv_idx = advantage_df.iloc[i].idxmax()
            max_adv_val = advantage_df.iloc[i].max()
            if max_adv_val > 0.001:
                col_idx = list(advantage_df.columns).index(max_adv_idx)
                strongest_advantages.append((i, col_idx, max_adv_val))

    # Sort by value
    strongest_advantages.sort(key=lambda x: x[2], reverse=True)
    highlight_colors = ['#FFD700', '#FFA500']

    for idx, (i, j, val) in enumerate(strongest_advantages[:2]):
        # Draw rectangle
        rect = Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, linewidth=3,
                        edgecolor=highlight_colors[idx], facecolor='none', zorder=10)
        ax.add_patch(rect)

        # Add rank annotation
        ax.text(j + 0.35, i + 0.35, f'{idx + 1}', ha='center', va='center',
               fontsize=FONTS_SANS['annotation'], fontweight='bold',
               color='white',
               bbox=dict(boxstyle='circle,pad=0.15', facecolor=highlight_colors[idx],
                        edgecolor='black', linewidth=1), zorder=11)


def plot_advantage_landscapes(advantage_df: pd.DataFrame, 
                              significance_df: pd.DataFrame) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes], pd.DataFrame, pd.DataFrame]:
    """
    Creates the combined advantage landscape plot for leaf and root systems.

    Args:
        advantage_df: Combined advantage DataFrame.
        significance_df: Combined significance DataFrame.

    Returns:
        Tuple: Figure object, axes tuple, leaf advantage DataFrame, root advantage DataFrame.
    """
    print("Generating leaf and root advantage landscape plots...")

    leaf_systems = [idx for idx in advantage_df.index if idx.startswith('Leaf')]
    root_systems = [idx for idx in advantage_df.index if idx.startswith('Root')]

    leaf_advantage_df = advantage_df.loc[leaf_systems]
    leaf_significance_df = significance_df.loc[leaf_systems]
    root_advantage_df = advantage_df.loc[root_systems]
    root_significance_df = significance_df.loc[root_systems]

    fig = plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.6)
    ax_leaf = fig.add_subplot(gs[0, 0])
    ax_root = fig.add_subplot(gs[0, 1])

    cmap = create_custom_colormap()

    # Determine normalization range across both datasets
    all_values = pd.concat([leaf_advantage_df, root_advantage_df]).values.flatten()
    max_val = max(all_values.max(), 0.001)
    min_val = all_values.min()
    
    if min_val < 0:
        norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val)
    else:
        norm = plt.Normalize(vmin=0, vmax=max_val)

    plot_single_landscape(ax_leaf, leaf_advantage_df, leaf_significance_df,
                             cmap, norm, 'Leaf Biological Systems', 'a', 'LEAF')

    plot_single_landscape(ax_root, root_advantage_df, root_significance_df,
                             cmap, norm, 'Root Biological Systems', 'b', 'ROOT')

    fig.tight_layout(rect=[0, 0.08, 1, 0.90])

    # Add Colorbar
    cbar_ax = fig.add_axes([0.2, 0.97, 0.6, 0.03])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label('G1 Coordination Advantage (G1 - G2 Attention)',
                   fontsize=FONTS_SANS['legend_title'], rotation=0, labelpad=10)
    cbar.ax.tick_params(labelsize=FONTS_SANS['tick_label'])

    return fig, (ax_leaf, ax_root), leaf_advantage_df, root_advantage_df


def print_top_advantages(tissue_name: str, advantage_df: pd.DataFrame, top_n: int = 3) -> None:
    """
    Prints a summary of the top advantages for a given tissue.

    Args:
        tissue_name: Name of the tissue (e.g. "LEAF").
        advantage_df: DataFrame containing advantage values.
        top_n: Number of top advantages to display.
    """
    print(f"\n{tissue_name} SYSTEMS:")
    strongest = []
    
    for i, row_name in enumerate(advantage_df.index):
        if not advantage_df.iloc[i].empty:
            max_adv_idx = advantage_df.iloc[i].idxmax()
            max_adv_val = advantage_df.iloc[i].max()
            if max_adv_val > 0.001:
                strongest.append((max_adv_val, row_name, max_adv_idx))

    strongest.sort(reverse=True)
    
    for idx, (val, system, timepoint) in enumerate(strongest[:top_n]):
        print(f"   {idx + 1}. {system} at TP {timepoint}: {val:.4f} advantage")


def generate_figure_panels_a_b() -> Optional[str]:
    """
    Main execution function to generate and save Figure 4, Panels A-B.
    
    Returns:
        Optional[str]: Path to the saved PNG file, or None if execution failed.
    """
    print("\nStarting generation of Figure 4, Panels A-B...")

    leaf_df, root_df = load_and_preprocess_data()
    if leaf_df is None or root_df is None:
        print("Failed to load data. Exiting.")
        return None

    advantage_df, significance_df = calculate_advantage_matrix(leaf_df, root_df)

    fig, _, leaf_adv_df, root_adv_df = plot_advantage_landscapes(advantage_df, significance_df)

    # Save outputs
    output_path_png = os.path.join(OUTPUT_DIR, "Figure4_Panels_a-b_Advantage_Landscapes.png")
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path_png}")

    output_path_pdf = output_path_png.replace('.png', '.pdf')
    fig.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {output_path_pdf}")

    plt.close()

    # Save separate tissue advantage matrices
    leaf_data_path = os.path.join(OUTPUT_DIR, "Figure4_Panel_a_Leaf_Advantage_Matrix.csv")
    leaf_adv_df.to_csv(leaf_data_path)
    print(f"Leaf advantage matrix saved to: {leaf_data_path}")

    root_data_path = os.path.join(OUTPUT_DIR, "Figure4_Panel_b_Root_Advantage_Matrix.csv")
    root_adv_df.to_csv(root_data_path)
    print(f"Root advantage matrix saved to: {root_data_path}")

    # Summary
    print("\n--- Summary of Top Advantages ---")
    print_top_advantages("LEAF", leaf_adv_df)
    print_top_advantages("ROOT", root_adv_df)
    
    print("\nScript execution finished successfully.")
    return output_path_png


if __name__ == "__main__":
    # Create dummy data if needed
    create_dummy_data()
    
    # Generate the main figure
    generate_figure_panels_a_b()
