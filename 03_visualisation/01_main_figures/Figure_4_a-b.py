#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4, Panels A-B: Biological Systems Advantage Landscapes
============================================================

This script generates a visualization showing the coordination advantage of one
genotype (G1) over another (G2) across different biological systems and time
points.

The visualization consists of two heatmaps (panels A and B), one for leaf
tissue and one for root tissue.

- X-axis: Time progression (e.g., TP1, TP2, TP3)
- Y-axis: Biological feature categories (combinations of spectral regions and
          metabolite types)
- Color: Magnitude of G1's coordination advantage, calculated as the
         difference in mean attention scores (G1 - G2).
- Overlays: Markers for statistical significance.

This visualization aims to synthesize complex network patterns into a clear
biological narrative, highlighting when and where the key physiological
differences between the genotypes manifest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu
import os
import warnings

warnings.filterwarnings('ignore')

# === STYLING ===
COLORS = {
    'G1': '#00FA9A',             # Tolerant (green)
    'G2': '#48D1CC',             # Susceptible (blue)
    'Advantage_High': '#006837',  # Dark green for high G1 advantage
    'Advantage_Low': '#f7f7f7',   # Light gray for no advantage
    'Disadvantage': '#d73027',    # Red for G2 advantage (rare)
    'Significance': '#252525',    # Black for significance contours
    'Background': '#FFFFFF',
    'Text_Dark': '#252525',
    'Grid': '#e5e5e5',
}

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

plt.rcParams.update({
    'font.family': FONTS_SANS['family'],
    'font.sans-serif': FONTS_SANS['sans_serif'],
})

# === FILE PATHS ===
output_dir = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot\test"
os.makedirs(output_dir, exist_ok=True)

# Define paths for dummy data.
leaf_trends_path = os.path.join(output_dir, "processed_attention_trends_top_500_Leaf.csv")
root_trends_path = os.path.join(output_dir, "processed_attention_trends_top_500_Root.csv")


def create_dummy_data():
    """Creates dummy data files for demonstration purposes if they don't exist."""
    if os.path.exists(leaf_trends_path) and os.path.exists(root_trends_path):
        return

    print("Creating dummy data for demonstration...")
    # Dummy data creation logic
    spectral_features_leaf = [f"W_{w}" for w in range(500, 750, 50)]
    spectral_features_root = [f"W_{w}" for w in range(400, 2500, 100)]
    metabolite_features = [f"N_Cluster_{i}" for i in range(5)] + [f"P_Cluster_{i}" for i in range(5)]

    def generate_data(spectral_features, file_path):
        data = []
        for spec_feat in spectral_features:
            for met_feat in metabolite_features:
                for day in [1, 2, 3]:
                    for genotype in ['1', '2']:
                        for treatment in ['1']:
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
        print(f"✓ Created dummy file: {file_path}")

    generate_data(spectral_features_leaf, leaf_trends_path)
    generate_data(spectral_features_root, root_trends_path)


def load_and_preprocess_data():
    """Load data and apply necessary preprocessing and cleaning."""
    print("Loading and preprocessing data...")

    try:
        leaf_df = pd.read_csv(leaf_trends_path)
        root_df = pd.read_csv(root_trends_path)
        print(f"✓ Loaded data: Leaf {leaf_df.shape}, Root {root_df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

    # Clean data
    for df_name, df in [("Leaf", leaf_df), ("Root", root_df)]:
        df.columns = df.columns.str.strip()
        df['Genotype'] = df['Genotype'].astype(str).replace({
            '1': 'G1', '2': 'G2', '1.0': 'G1', '2.0': 'G2'
        })

        if 'Day' in df.columns:
            df['Time_Point'] = df['Day'].astype(int)
        elif 'Time_Point' in df.columns:
            df['Time_Point'] = df['Time_Point'].astype(int)

        df['Treatment'] = df['Treatment'].astype(str).replace({
            '1': 'T1', '1.0': 'T1'
        })

        if 'Metabolite_Feature' in df.columns:
            df['Metabolite_Feature'] = df['Metabolite_Feature'].str.replace('N_Cluster_', 'N_')
            df['Metabolite_Feature'] = df['Metabolite_Feature'].str.replace('P_Cluster_', 'P_')

    return leaf_df, root_df


def categorize_spectral_features(spectral_feature):
    """Categorize spectral features into biological regions based on wavelength."""
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


def categorize_metabolite_features(metabolite_feature):
    """Categorize metabolite features into chemical classes."""
    if pd.isna(metabolite_feature):
        return 'Unknown'

    feature_str = str(metabolite_feature)

    if feature_str.startswith('N_'):
        return 'N-Metabolites'
    elif feature_str.startswith('P_'):
        return 'P-Metabolites'
    else:
        return 'Other Metabolites'


def calculate_advantage_matrix(leaf_df, root_df):
    """
    Calculate G1 advantage across biological systems and time points.

    Returns two DataFrames: one for the advantage values (G1 mean - G2 mean)
    and one for the statistical significance (p < 0.05 from Mann-Whitney U test).
    """
    print("Calculating biological advantage matrix...")

    # Combine datasets with tissue labels
    leaf_df = leaf_df.copy()
    root_df = root_df.copy()
    leaf_df['Tissue'] = 'Leaf'
    root_df['Tissue'] = 'Root'

    combined_df = pd.concat([leaf_df, root_df], ignore_index=True)

    # Filter for stress condition
    stress_df = combined_df[combined_df['Treatment'] == 'T1'].copy()

    # Add biological categories
    stress_df['Spectral_Category'] = stress_df['Spectral_Feature'].apply(categorize_spectral_features)
    stress_df['Metabolite_Category'] = stress_df['Metabolite_Feature'].apply(categorize_metabolite_features)

    # Create biological system combinations
    stress_df['Biological_System'] = (stress_df['Tissue'] + ' ' +
                                     stress_df['Spectral_Category'] + ' → ' +
                                     stress_df['Metabolite_Category'])

    # Calculate advantage matrix
    advantage_matrix = []
    significance_matrix = []

    biological_systems = sorted(stress_df['Biological_System'].unique())
    time_points = sorted(stress_df['Time_Point'].unique())

    print(f"Analyzing {len(biological_systems)} biological systems across {len(time_points)} time points...")

    for bio_sys in biological_systems:
        sys_data = stress_df[stress_df['Biological_System'] == bio_sys]

        advantage_row = []
        significance_row = []

        for tp in time_points:
            tp_data = sys_data[sys_data['Time_Point'] == tp]

            g1_values = tp_data[tp_data['Genotype'] == 'G1']['Mean_Attention_S2M_Group_AvgHeads']
            g2_values = tp_data[tp_data['Genotype'] == 'G2']['Mean_Attention_S2M_Group_AvgHeads']

            if len(g1_values) > 1 and len(g2_values) > 1:
                g1_mean = g1_values.mean()
                g2_mean = g2_values.mean()
                advantage = g1_mean - g2_mean

                try:
                    _, p_value = mannwhitneyu(g1_values, g2_values, alternative='two-sided')
                    significant = p_value < 0.05
                except ValueError:
                    significant = False  # Occurs if all values are the same

                advantage_row.append(advantage)
                significance_row.append(significant)
            else:
                advantage_row.append(0)
                significance_row.append(False)

        advantage_matrix.append(advantage_row)
        significance_matrix.append(significance_row)

    advantage_df = pd.DataFrame(advantage_matrix,
                               index=biological_systems,
                               columns=[str(tp) for tp in time_points])

    significance_df = pd.DataFrame(significance_matrix,
                                  index=biological_systems,
                                  columns=[str(tp) for tp in time_points])

    print(f"✓ Created advantage matrix: {advantage_df.shape}")
    return advantage_df, significance_df


def create_custom_colormap():
    """Create a custom colormap for visualizing G1 advantage."""
    colors = [
        '#f7fcf0', "#cff9c3", "#99d78b", "#2fdd5b",
        '#7bccc4', "#7bf1fa", "#50aa75", "#107a39", "#0E6068"
    ]
    return LinearSegmentedColormap.from_list('g1_advantage_custom', colors, N=256)


def plot_advantage_landscapes(advantage_df, significance_df):
    """Create separate advantage landscape plots for leaf and root systems."""
    print("Creating leaf and root advantage landscape plots...")

    leaf_systems = [idx for idx in advantage_df.index if idx.startswith('Leaf')]
    root_systems = [idx for idx in advantage_df.index if idx.startswith('Root')]

    leaf_advantage_df = advantage_df.loc[leaf_systems]
    leaf_significance_df = significance_df.loc[leaf_systems]
    root_advantage_df = advantage_df.loc[root_systems]
    root_significance_df = significance_df.loc[root_systems]

    fig = plt.figure(figsize=(19, 7.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.6)
    ax_leaf = fig.add_subplot(gs[0, 0])
    ax_root = fig.add_subplot(gs[0, 1])

    cmap = create_custom_colormap()

    all_values = pd.concat([leaf_advantage_df, root_advantage_df]).values.flatten()
    max_val = max(all_values.max(), 0.001)
    min_val = all_values.min()
    norm = TwoSlopeNorm(vmin=min_val, vcenter=0, vmax=max_val) if min_val < 0 else plt.Normalize(vmin=0, vmax=max_val)

    plot_single_landscape(ax_leaf, leaf_advantage_df, leaf_significance_df,
                             cmap, norm, 'Leaf Biological Systems', 'a', 'LEAF')

    plot_single_landscape(ax_root, root_advantage_df, root_significance_df,
                             cmap, norm, 'Root Biological Systems', 'b', 'ROOT')

    fig.tight_layout(rect=[0, 0.08, 1, 0.90])

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


def plot_single_landscape(ax, advantage_df, significance_df, cmap, norm,
                             title, panel_label, tissue_name):
    """Helper function to plot an advantage landscape for a single tissue."""

    ax.imshow(advantage_df.values, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    for i in range(len(advantage_df.index)):
        for j in range(len(advantage_df.columns)):
            if significance_df.iloc[i, j]:
                ax.plot(j, i, 'o', markersize=5, markerfacecolor='white',
                       markeredgewidth=1.5, markeredgecolor='black', alpha=0.9)

    ax.set_xticks(range(len(advantage_df.columns)))
    ax.set_xticklabels(advantage_df.columns, fontsize=FONTS_SANS['tick_label'])
    ax.set_xlabel('Time Points', fontsize=FONTS_SANS['axis_label'], fontweight='bold')

    simplified_labels = []
    for label in advantage_df.index:
        clean_label = label.replace(f'{tissue_name.title()} ', '')
        clean_label = clean_label.replace('-Metabolites', '-Met').replace('Metabolites', 'Met')
        simplified_labels.append(clean_label)

    ax.set_yticks(range(len(advantage_df.index)))
    ax.set_yticklabels(simplified_labels, fontsize=FONTS_SANS['tick_label'])
    ax.set_ylabel(f'{tissue_name.upper()} Systems (Spectral → Metabolite)',
                  fontsize=FONTS_SANS['axis_label'], fontweight='bold')

    ax.set_title(f'{title}\nAdvantage Landscape',
                 fontsize=FONTS_SANS['panel_title'], fontweight='bold', pad=15)
    ax.text(-0.1, 1.05, panel_label, transform=ax.transAxes,
            fontsize=FONTS_SANS['panel_label'], fontweight='bold')

    strongest_advantages = []
    for i, row_name in enumerate(advantage_df.index):
        if not advantage_df.iloc[i].empty:
            max_adv_idx = advantage_df.iloc[i].idxmax()
            max_adv_val = advantage_df.iloc[i].max()
            if max_adv_val > 0.001:
                strongest_advantages.append((i, list(advantage_df.columns).index(max_adv_idx),
                                           max_adv_val, row_name, max_adv_idx))

    strongest_advantages.sort(key=lambda x: (x[2], x[3]), reverse=True)
    highlight_colors = ['#FFD700', '#FFA500']

    for idx, (i, j, val, system, timepoint) in enumerate(strongest_advantages[:2]):
        rect = Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, linewidth=3,
                        edgecolor=highlight_colors[idx], facecolor='none', zorder=10)
        ax.add_patch(rect)

        ax.text(j + 0.35, i + 0.35, f'{idx + 1}', ha='center', va='center',
               fontsize=FONTS_SANS['annotation'], fontweight='bold',
               color='white',
               bbox=dict(boxstyle='circle,pad=0.15', facecolor=highlight_colors[idx],
                        edgecolor='black', linewidth=1), zorder=11)

    if panel_label == 'a':
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                       markeredgecolor='black', markeredgewidth=1.5,
                       markersize=6, label='Significant (p < 0.05)', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                 fontsize=FONTS_SANS['legend_text'], framealpha=0.9)


def generate_figure_panels_a_b():
    """
    Main function to generate and save Figure 4, Panels A-B.
    This includes data loading, processing, plotting, and saving outputs.
    """
    print("\n--- Generating Figure 4, Panels A-B ---")

    leaf_df, root_df = load_and_preprocess_data()
    if leaf_df is None or root_df is None:
        print("❌ Failed to load data. Exiting.")
        return

    advantage_df, significance_df = calculate_advantage_matrix(leaf_df, root_df)

    fig, (ax_leaf, ax_root), leaf_adv_df, root_adv_df = plot_advantage_landscapes(advantage_df, significance_df)

    # Save the figure
    output_path_png = os.path.join(output_dir, "Figure4_Panels_a-b_Advantage_Landscapes.png")
    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Figure saved to: {output_path_png}")

    # Save as PDF
    output_path_pdf = output_path_png.replace('.png', '.pdf')
    fig.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ PDF saved to: {output_path_pdf}")

    plt.close()

    # Save separate tissue advantage matrices
    leaf_data_path = os.path.join(output_dir, "Figure4_Panel_a_Leaf_Advantage_Matrix.csv")
    leaf_adv_df.to_csv(leaf_data_path)
    print(f"✓ Leaf advantage matrix saved to: {leaf_data_path}")

    root_data_path = os.path.join(output_dir, "Figure4_Panel_b_Root_Advantage_Matrix.csv")
    root_adv_df.to_csv(root_data_path)
    print(f"✓ Root advantage matrix saved to: {root_data_path}")

    print("\n--- Summary of Top Advantages ---")

    print("\n🍃 LEAF SYSTEMS:")
    leaf_strongest = []
    for i, row_name in enumerate(leaf_adv_df.index):
        if not leaf_adv_df.iloc[i].empty:
            max_adv_idx = leaf_adv_df.iloc[i].idxmax()
            max_adv_val = leaf_adv_df.iloc[i].max()
            if max_adv_val > 0.001:
                leaf_strongest.append((max_adv_val, row_name, max_adv_idx))

    leaf_strongest.sort(reverse=True)
    for idx, (val, system, timepoint) in enumerate(leaf_strongest[:3]):
        print(f"   {idx + 1}. {system} at TP {timepoint}: {val:.4f} advantage")

    print("\n🌱 ROOT SYSTEMS:")
    root_strongest = []
    for i, row_name in enumerate(root_adv_df.index):
        if not root_adv_df.iloc[i].empty:
            max_adv_idx = root_adv_df.iloc[i].idxmax()
            max_adv_val = root_adv_df.iloc[i].max()
            if max_adv_val > 0.001:
                root_strongest.append((max_adv_val, row_name, max_adv_idx))

    root_strongest.sort(reverse=True)
    for idx, (val, system, timepoint) in enumerate(root_strongest[:3]):
        print(f"   {idx + 1}. {system} at TP {timepoint}: {val:.4f} advantage")

    print("\n--- Script finished ---")
    
    return output_path_png


if __name__ == "__main__":
    # Create dummy data for demonstration if real data is not present.
    create_dummy_data()
    # Generate the main figure.
    generate_figure_panels_a_b()