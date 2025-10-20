#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4 (c-f): Hub-Based Temporal Dynamics of Network Properties
=================================================================

This script generates the final, publication-quality plots for Figure 4, panels c-f.
It directly supports the manuscript's narrative by quantifying two key aspects of
network architecture over time: Coordination Strength and Network Focus.

Key Features of this Final Script:
- REPRODUCIBILITY: Metrics are calculated directly from the raw HDF5 attention
  tensor and metadata every time, ensuring no reliance on intermediate files.
- AESTHETICS: Y-axis labels are professionally formatted, and a shared legend
  is used for clarity.
- ACCURACY: Statistical significance (p < 0.05, Mann-Whitney U) between
  genotypes is calculated and displayed with asterisks.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import h5py
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')

# Exact styling from original (DO NOT CHANGE)
COLORS = {
    'G1': '#00FA9A',  # Tolerant Genotype (Medium Spring Green)
    'G2': '#48D1CC',  # Susceptible Genotype (Medium Turquoise)
    'G1_Light': '#98FB98',
    'G2_Light': '#B0E0E6',
    'Background': '#FFFFFF',
    'Panel_Background': '#f9f9f9',
    'Text_Dark': '#252525',
    'Grid': '#e5e5e5',
    'Significance': '#FF6347', # Tomato Red for asterisks
}

FONTS_SANS = {
    'family': 'sans-serif',
    'sans_serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'main_title': 19,
    'panel_label': 20,
    'panel_title': 18,
    'axis_label': 17,
    'tick_label': 16,
    'legend_title': 17,
    'legend_text': 16,
}

plt.rcParams.update({
    'font.family': FONTS_SANS['family'],
    'font.sans-serif': FONTS_SANS['sans_serif'],
    'axes.labelsize': FONTS_SANS['axis_label'],
    'xtick.labelsize': FONTS_SANS['tick_label'],
    'ytick.labelsize': FONTS_SANS['tick_label'],
    'figure.facecolor': COLORS['Background'],
    'axes.facecolor': COLORS['Panel_Background'],
})


# === FILE PATHS ===
# --- !!! IMPORTANT: SET YOUR BASE PATH HERE !!! ---
BASE_PATH = r"C:\Users\ms\Desktop\hyper"
# ---

BASE_OUTPUT_DIR = os.path.join(BASE_PATH, "output", "transformer", "v3_feature_attention")
OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot\test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HDF5_PATHS = {
    "Leaf": os.path.join(BASE_OUTPUT_DIR, "leaf", "results", "raw_attention_data_Leaf.h5"),
    "Root": os.path.join(BASE_OUTPUT_DIR, "root", "results", "raw_attention_data_Root.h5")
}

METADATA_CSV_PATHS = {
    "Leaf": os.path.join(BASE_OUTPUT_DIR, "leaf", "results", "raw_attention_metadata_Leaf.csv"),
    "Root": os.path.join(BASE_OUTPUT_DIR, "root", "results", "raw_attention_metadata_Root.csv")
}


def calculate_hub_based_coordination(attn_tensor: np.ndarray, metadata_df: pd.DataFrame,
                                   tissue_name: str, top_k: int = 100) -> pd.DataFrame:
    """
    Calculates coordination metrics based on the top S->M hub pairs, matching the manuscript's methods.
    (e.g., "average attention across the top 100 S2M pairs").
    """
    print(f"\n🧬 Calculating HUB-BASED coordination for {tissue_name}...")
    n_samples, n_heads, n_spec, n_metab = attn_tensor.shape

    # Step 1: Identify top k hub pairs from the attention tensor averaged over samples and heads.
    overall_attention = np.mean(attn_tensor, axis=(0, 1))
    flat_indices = np.argsort(overall_attention.flatten())[-top_k:]
    top_pairs = [(i // n_metab, i % n_metab) for i in flat_indices]
    print(f"Identified {len(top_pairs)} hub pairs for {tissue_name}.")

    # Step 2: For each sample, calculate coordination metrics based on these specific hub pairs.
    coordination_metrics = []
    for sample_idx in range(n_samples):
        sample_attn_to_hubs = [np.mean(attn_tensor[sample_idx, :, spec_idx, metab_idx]) for spec_idx, metab_idx in top_pairs]
        sample_attn_to_hubs = np.array(sample_attn_to_hubs)

        coordination_metrics.append({
            'Coordination_Strength': np.mean(sample_attn_to_hubs),
            'Network_Focus': np.std(sample_attn_to_hubs),
        })

    coord_df = pd.DataFrame(coordination_metrics, index=metadata_df.index)

    # Step 3: Merge with metadata to create the final analysis dataframe.
    result_df = pd.concat([metadata_df, coord_df], axis=1)
    print(f"Final {tissue_name} analysis dataframe created. Shape: {result_df.shape}")
    return result_df


def standardize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names and values in the raw metadata dataframe with
    enhanced debugging and robust filtering.
    """
    
    # Standardize Genotype and Time Point (this part is fine)
    df['Genotype'] = df['Genotype'].astype(str).str.strip().replace({'Gladius': 'G1', 'DAS5_003811': 'G2'})
    df['Time_Point'] = pd.to_numeric(df['Day'], errors='coerce').astype('Int64')

    # --- THE ROBUST FIX IS HERE ---
    
    # 1. Ensure the 'Treatment' column exists before proceeding
    if 'Treatment' not in df.columns:
        print("❌ CRITICAL ERROR in standardize_metadata: 'Treatment' column not found!")
        return pd.DataFrame() # Return an empty dataframe to trigger the error correctly

    # 2. Convert the 'Treatment' column to a consistent string type.
    df['Treatment'] = df['Treatment'].astype(str).str.strip()

    # 3. **Add a diagnostic print statement.** This is crucial for debugging.
    #    It will show us exactly what values are in the column before we filter.
    print(f"DEBUG: Unique values found in 'Treatment' column: {df['Treatment'].unique()}")

    # 4. Use a more flexible filter that accepts multiple possible values for "stressed".
    #    This handles '1', '1.0', or even 'T1' if it existed.
    stress_conditions = ['1', '1.0', 'T1']
    df_filtered = df[df['Treatment'].isin(stress_conditions)].copy()
    
    # 5. Report the result of the filtering step.
    print(f"DEBUG: Found {len(df_filtered)} rows after filtering for stress conditions {stress_conditions}.")

    # Return the filtered and cleaned dataframe
    return df_filtered.dropna(subset=['Genotype', 'Time_Point'])


def load_and_process_tissue_data(tissue_name: str) -> pd.DataFrame:
    """
    Loads raw HDF5 and metadata, VALIDATES, FILTERS for stress samples,
    and returns a single dataframe ready for plotting.
    This is the final, robust data pipeline.
    """
    print(f"\n{'='*60}\nPROCESSING {tissue_name.upper()} FROM RAW DATA\n{'='*60}")

    # Step 1: Load FULL raw data
    try:
        with h5py.File(HDF5_PATHS[tissue_name], 'r') as f:
            attn_tensor = f['attention_spec_to_metab'][:]
        metadata_df = pd.read_csv(METADATA_CSV_PATHS[tissue_name], index_col='Row_names')
    except Exception as e:
        print(f"❌ ERROR: Failed to load raw HDF5 or Metadata for {tissue_name}: {e}")
        return None

    # Step 2: VALIDATE that the FULL raw datasets match
    if len(metadata_df) != attn_tensor.shape[0]:
        print(f"❌ ERROR: Initial mismatch between full metadata ({len(metadata_df)}) and tensor ({attn_tensor.shape[0]}) for {tissue_name}.")
        return None
    print(f"✅ Initial validation passed: Full metadata ({len(metadata_df)}) matches full tensor ({attn_tensor.shape[0]}).")


    # Step 3: FILTER metadata to get only the stressed samples
    metadata_stressed = standardize_metadata(metadata_df.copy())
    if metadata_stressed.empty:
        print(f"❌ ERROR: No stressed samples found in metadata for {tissue_name} after filtering.")
        return None


    # Step 4: FILTER the tensor to match the stressed metadata
    # Get the original integer positions of the samples we want to keep
    original_indices = metadata_df.index.get_indexer(metadata_stressed.index)
    
    # Slice the numpy tensor using these integer positions
    tensor_stressed = attn_tensor[original_indices]
    
    print(f"✅ Filtering complete. Kept {len(metadata_stressed)} stressed samples.")
    print(f"   - Filtered metadata shape: {metadata_stressed.shape}")
    print(f"   - Filtered tensor shape:   {tensor_stressed.shape}")

    # Step 5: Final validation and calculation
    if len(metadata_stressed) != tensor_stressed.shape[0]:
         print(f"❌ CRITICAL ERROR: Final mismatch after filtering! Metadata={len(metadata_stressed)}, Tensor={tensor_stressed.shape[0]}")
         return None

    # Now, perform calculations on the perfectly aligned, stressed-only data
    result_df = calculate_hub_based_coordination(tensor_stressed, metadata_stressed, tissue_name)
    return result_df


def plot_panel(ax, data_df: pd.DataFrame, metric_col: str, title: str, ylabel: str, panel_label: str):
    """Plots a single panel for the figure, including error bars and significance."""
    
    genotype_styles = {
        'G1': {'color': COLORS['G1'], 'marker': 'o', 'linestyle': '-', 'label': 'G1 (Tolerant)'},
        'G2': {'color': COLORS['G2'], 'marker': 's', 'linestyle': '--', 'label': 'G2 (Susceptible)'}
    }
    
    stats_data = data_df.groupby(['Time_Point', 'Genotype'])[metric_col].agg(['mean', 'sem']).reset_index()

    for genotype, style in genotype_styles.items():
        geno_data = stats_data[stats_data['Genotype'] == genotype].sort_values('Time_Point')
        if not geno_data.empty:
            ax.errorbar(
                geno_data['Time_Point'], geno_data['mean'], yerr=geno_data['sem'],
                color=style['color'], marker=style['marker'], linestyle=style['linestyle'],
                label=style['label'], linewidth=2.5, markersize=7, capsize=4,
                capthick=1.5, alpha=0.95, markeredgecolor='white', markeredgewidth=0.5
            )

    ax.set_title(title, fontsize=FONTS_SANS['panel_title'], fontweight='bold', pad=15)
    ax.set_xlabel('Time Point', fontsize=FONTS_SANS['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONTS_SANS['axis_label'])
    ax.text(-0.15, 1.1, panel_label, transform=ax.transAxes, fontsize=FONTS_SANS['panel_label'], fontweight='bold', va='top', ha='left')

    # Add legend to the first panel only
    if panel_label.lower() == 'c':
        ax.legend(loc='best', fontsize=FONTS_SANS['legend_text'],
                  framealpha=0.95, fancybox=True)

    time_points = sorted(data_df['Time_Point'].unique())
    ax.set_xticks(time_points)
    ax.set_xlim(min(time_points) - 0.3, max(time_points) + 0.3)
    ax.grid(True, axis='y', which='major', linestyle='--', color=COLORS['Grid'], alpha=0.7)

    # --- FINAL: Fix Y-axis formatting to be clean and readable ---
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Add significance asterisks
    for tp in time_points:
        g1_data = data_df[(data_df['Time_Point'] == tp) & (data_df['Genotype'] == 'G1')][metric_col].dropna()
        g2_data = data_df[(data_df['Time_Point'] == tp) & (data_df['Genotype'] == 'G2')][metric_col].dropna()

        if len(g1_data) > 1 and len(g2_data) > 1:
            _, p_value = mannwhitneyu(g1_data, g2_data, alternative='two-sided')
            if p_value < 0.05:
                y_max = stats_data[stats_data['Time_Point'] == tp]['mean'].max()
                y_pos = y_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
                ax.text(tp, y_pos, '*', ha='center', va='center', fontweight='bold', 
                        fontsize=FONTS_SANS['panel_label'], color=COLORS['Significance'])

def create_figure4_panels_c_f():
    """Main function to orchestrate data loading, processing, and plotting."""
    
    leaf_data = load_and_process_tissue_data("Leaf")
    root_data = load_and_process_tissue_data("Root")
    
    if leaf_data is None or root_data is None:
        print("❌ CRITICAL ERROR: Could not process data for both tissues. Aborting plot generation.")
        return

    # --- Create the multi-panel figure ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)

    panel_definitions = {
        'c': (axes[0], leaf_data, 'Coordination_Strength', 'Leaf', 'Coordination Strength'),
        'd': (axes[1], leaf_data, 'Network_Focus', 'Leaf', 'Network Focus'),
        'e': (axes[2], root_data, 'Coordination_Strength', 'Root', 'Coordination Strength'),
        'f': (axes[3], root_data, 'Network_Focus', 'Root', 'Network Focus'),
    }

    for panel_label, (ax, data, metric, tissue, ylabel) in panel_definitions.items():
        title = f"{tissue}\n{ylabel}"
        plot_panel(ax, data, metric, title, ylabel, panel_label)
    
    plt.tight_layout()
    
    # Save the final figure
    output_png = os.path.join(OUTPUT_DIR, "fig4.png")
    output_pdf = os.path.join(OUTPUT_DIR, "fig4.pdf")
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf, format='pdf')

    print(f"\n✅ Final Figure 4 (c-f) saved successfully to:\n  - {output_png}\n  - {output_pdf}")
    plt.show()

if __name__ == "__main__":
    create_figure4_panels_c_f()