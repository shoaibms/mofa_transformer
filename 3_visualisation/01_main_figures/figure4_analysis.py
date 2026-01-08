#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEFINITIVE FIX: Figure 4 (c-f) Hub-Based Coordination Analysis
=================================================================

This script implements the CORRECT approach:
1. COMPLETE augmentation filtering (all 7 types)
2. PAIR-LEVEL analysis (n=100 pairs, not n~10 samples)
3. CONSISTENT statistics (paired Wilcoxon, not sample-level MWU)

Run this AFTER running filter_test_samples_for_interpretability.py
OR this script can handle raw data directly.

Author: Analysis fix for MOFA+ Transformer manuscript
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import h5py
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CRITICAL: COMPLETE AUGMENTATION REGEX (7 TYPES)
# =============================================================================
# This is the CORRECT regex that matches ALL augmentation suffixes
AUG_REGEX = r'(?:_GP|_WARP|_SCALE|_NOISE|_ADD|_MULT|_MIX)$'

# Also need to handle _L tissue marker (not an augmentation, just identifier)
# Original samples end with _L (for Leaf tissue marker from Row_names)

# =============================================================================
# CONFIGURATION
# =============================================================================
TOP_K = 100  # Number of hub pairs for analysis

COLORS = {
    'G1': '#00FA9A',
    'G2': '#48D1CC', 
    'G1_Light': '#98FB98',
    'G2_Light': '#B0E0E6',
    'Background': '#FFFFFF',
    'Panel_Background': '#f9f9f9',
    'Text_Dark': '#252525',
    'Grid': '#e5e5e5',
    'Significance': '#FF6347',
}

FONTS = {
    'panel_label': 20,
    'panel_title': 18,
    'axis_label': 17,
    'tick_label': 16,
    'legend_text': 14,
}


def diagnose_augmentation(metadata_df):
    """
    DIAGNOSTIC: Exhaustively identify all augmentation patterns.
    This should be run FIRST to understand your data.
    """
    print("\n" + "="*70)
    print("AUGMENTATION DIAGNOSTIC")
    print("="*70)
    
    row_ids = pd.Series(metadata_df.index.astype(str))
    
    # Check for all known augmentation suffixes
    known_augs = ['_GP', '_WARP', '_SCALE', '_NOISE', '_ADD', '_MULT', '_MIX']
    
    print("\nChecking for augmentation suffixes:")
    total_aug = 0
    for aug in known_augs:
        count = row_ids.str.endswith(aug).sum()
        total_aug += count
        status = f"{count} samples" if count > 0 else "not found"
        print(f"  {aug}: {status}")
    
    # Check what regex catches
    old_regex = r'(?:_WARP|_NOISE|_MULT|_MIX)$'
    new_regex = AUG_REGEX
    
    caught_old = row_ids.str.contains(old_regex, regex=True, na=False).sum()
    caught_new = row_ids.str.contains(new_regex, regex=True, na=False).sum()
    
    print(f"\nRegex comparison:")
    print(f"  Old regex (4 types): catches {caught_old} samples")
    print(f"  New regex (7 types): catches {caught_new} samples")
    print(f"  DIFFERENCE (leaked pseudo-replicates): {caught_new - caught_old}")
    
    # True originals
    true_originals = len(metadata_df) - caught_new
    print(f"\n  TOTAL samples: {len(metadata_df)}")
    print(f"  TRUE ORIGINALS: {true_originals}")
    print(f"  AUGMENTED: {caught_new}")
    
    return caught_new, true_originals


def load_and_clean_data(tissue_name, base_path):
    """
    Load data with CORRECT augmentation filtering.
    """
    print(f"\n{'='*60}")
    print(f"Loading {tissue_name.upper()}")
    print("="*60)
    
    # Paths - adjust these to your setup
    h5_path = os.path.join(base_path, tissue_name.lower(), "results", 
                           f"raw_attention_data_{tissue_name}.h5")
    meta_path = os.path.join(base_path, tissue_name.lower(), "results",
                             f"raw_attention_metadata_{tissue_name}.csv")
    
    # Load data
    with h5py.File(h5_path, 'r') as f:
        attn_tensor = f['attention_spec_to_metab'][:]
    metadata_df = pd.read_csv(meta_path, index_col='Row_names')
    
    print(f"Raw data: {len(metadata_df)} samples, tensor shape {attn_tensor.shape}")
    
    # Run diagnostic
    n_aug, n_orig = diagnose_augmentation(metadata_df)
    
    # CORRECT filtering with complete regex
    row_ids = pd.Series(metadata_df.index.astype(str))
    mask_original = ~row_ids.str.contains(AUG_REGEX, regex=True, na=False).to_numpy()
    
    metadata_df = metadata_df.loc[mask_original].copy()
    attn_tensor = attn_tensor[mask_original]
    
    print(f"\nAfter filtering: {len(metadata_df)} samples")
    
    # Standardize metadata
    metadata_df['Genotype'] = metadata_df['Genotype'].astype(str).str.strip().replace(
        {'Gladius': 'G1', 'DAS5_003811': 'G2'})
    metadata_df['Day'] = pd.to_numeric(metadata_df['Day'], errors='coerce').astype('Int64')
    metadata_df['Treatment'] = pd.to_numeric(metadata_df['Treatment'], errors='coerce')
    
    # Filter to stressed samples only
    stressed_mask = (metadata_df['Treatment'] == 1).values
    metadata_stressed = metadata_df[stressed_mask].copy()
    tensor_stressed = attn_tensor[stressed_mask]
    
    print(f"Stressed samples: {len(metadata_stressed)}")
    
    # Sample counts per condition
    for day in sorted(metadata_stressed['Day'].dropna().unique()):
        n_g1 = ((metadata_stressed['Genotype'] == 'G1') & (metadata_stressed['Day'] == day)).sum()
        n_g2 = ((metadata_stressed['Genotype'] == 'G2') & (metadata_stressed['Day'] == day)).sum()
        print(f"  Day {int(day)}: G1={n_g1}, G2={n_g2}")
    
    # Average over heads
    tensor_mean = np.mean(tensor_stressed, axis=1)
    
    return tensor_mean, metadata_stressed


def define_hub_pairs(tensor, top_k=100):
    """
    Define hub pairs from POOLED stressed data (genotype-blind, day-blind).
    This ensures fair comparison: same pairs for G1 and G2.
    """
    pooled_mean = np.mean(tensor, axis=0)
    n_spec, n_metab = pooled_mean.shape
    flat_indices = np.argsort(pooled_mean.flatten())[-top_k:]
    hub_pairs = [(int(i // n_metab), int(i % n_metab)) for i in flat_indices]
    print(f"  Defined {top_k} hub pairs from pooled stressed data")
    return hub_pairs


def compute_pair_level_data(tensor, metadata, hub_pairs, tissue_name):
    """
    CORRECT APPROACH: For each hub pair, compute mean attention in G1 vs G2.
    This gives n=100 paired observations (not n~10 samples).
    """
    print(f"\nComputing pair-level data for {tissue_name}...")
    
    days = sorted(metadata['Day'].dropna().unique())
    results = []
    
    for day in days:
        day_mask = (metadata['Day'] == day).values
        g1_mask = ((metadata['Genotype'] == 'G1') & (metadata['Day'] == day)).values
        g2_mask = ((metadata['Genotype'] == 'G2') & (metadata['Day'] == day)).values
        
        for pair_idx, (spec_idx, metab_idx) in enumerate(hub_pairs):
            # Mean attention for this pair across all G1 samples on this day
            g1_attn = np.mean(tensor[g1_mask, spec_idx, metab_idx]) if g1_mask.sum() > 0 else np.nan
            g2_attn = np.mean(tensor[g2_mask, spec_idx, metab_idx]) if g2_mask.sum() > 0 else np.nan
            
            results.append({
                'Tissue': tissue_name,
                'Day': int(day),
                'Pair_ID': pair_idx,
                'Spec_Idx': spec_idx,
                'Metab_Idx': metab_idx,
                'G1_Attn': g1_attn,
                'G2_Attn': g2_attn,
            })
    
    df = pd.DataFrame(results)
    print(f"  Created {len(df)} pair-day observations")
    return df


def compute_statistics(pair_df, tissue_name):
    """
    CORRECT STATISTICS: Paired Wilcoxon on 100 pairs (not MWU on ~10 samples).
    """
    print(f"\nComputing statistics for {tissue_name}...")
    
    days = sorted(pair_df['Day'].unique())
    stats_results = []
    pvals = []
    
    for day in days:
        day_df = pair_df[pair_df['Day'] == day]
        g1_vals = day_df['G1_Attn'].values
        g2_vals = day_df['G2_Attn'].values
        
        # Paired Wilcoxon (100 paired observations)
        _, p_wilcox = wilcoxon(g1_vals, g2_vals, alternative='two-sided')
        pvals.append(p_wilcox)
        
        # Effect sizes
        median_g1 = np.median(g1_vals)
        median_g2 = np.median(g2_vals)
        fold_change = median_g1 / median_g2 if median_g2 > 0 else np.nan
        
        stats_results.append({
            'Tissue': tissue_name,
            'Day': day,
            'n_pairs': len(day_df),
            'median_G1': median_g1,
            'median_G2': median_g2,
            'fold_change': fold_change,
            'pct_diff': (median_g1 - median_g2) / median_g2 * 100 if median_g2 > 0 else np.nan,
            'p_wilcoxon': p_wilcox,
        })
    
    # FDR correction across days within this tissue
    _, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    
    stats_df = pd.DataFrame(stats_results)
    stats_df['p_fdr'] = pvals_fdr
    stats_df['significant'] = stats_df['p_fdr'] < 0.05
    
    print(f"\n  RESULTS - {tissue_name}:")
    print("  " + "-"*60)
    for _, row in stats_df.iterrows():
        sig = '***' if row['significant'] else ''
        print(f"  Day {int(row['Day'])}: G1={row['median_G1']:.4f}, G2={row['median_G2']:.4f}, "
              f"Fold={row['fold_change']:.2f}x, FDR={row['p_fdr']:.2e} {sig}")
    
    return stats_df


def plot_figure4(leaf_df, leaf_stats, root_df, root_stats, output_dir):
    """
    Create Figure 4 c-f with CORRECT pair-level visualization.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    panels = [
        (axes[0], leaf_df, leaf_stats, 'Leaf', 'c'),
        (axes[1], leaf_df, leaf_stats, 'Leaf', 'd'),  # Could be different metric
        (axes[2], root_df, root_stats, 'Root', 'e'),
        (axes[3], root_df, root_stats, 'Root', 'f'),
    ]
    
    for ax, df, stats_df, tissue, panel_label in panels:
        days = sorted(df['Day'].unique())
        
        # For panels c, e: Trajectory plot (G1 vs G2 lines)
        # For panels d, f: Boxplot comparison
        
        if panel_label in ['c', 'e']:
            # Trajectory: median with IQR
            g1_meds, g2_meds = [], []
            g1_errs, g2_errs = [], []
            
            for day in days:
                day_df = df[df['Day'] == day]
                g1_meds.append(np.median(day_df['G1_Attn']))
                g2_meds.append(np.median(day_df['G2_Attn']))
                g1_errs.append([np.median(day_df['G1_Attn']) - np.percentile(day_df['G1_Attn'], 25),
                               np.percentile(day_df['G1_Attn'], 75) - np.median(day_df['G1_Attn'])])
                g2_errs.append([np.median(day_df['G2_Attn']) - np.percentile(day_df['G2_Attn'], 25),
                               np.percentile(day_df['G2_Attn'], 75) - np.median(day_df['G2_Attn'])])
            
            g1_errs = np.array(g1_errs).T
            g2_errs = np.array(g2_errs).T
            
            ax.errorbar(days, g1_meds, yerr=g1_errs, 
                       color=COLORS['G1'], marker='o', linestyle='-',
                       linewidth=2.5, markersize=8, capsize=4, label='G1 (Tolerant)')
            ax.errorbar(days, g2_meds, yerr=g2_errs,
                       color=COLORS['G2'], marker='s', linestyle='--', 
                       linewidth=2.5, markersize=8, capsize=4, label='G2 (Susceptible)')
            
            ax.set_title(f'{tissue}\nCoordination Strength', fontsize=FONTS['panel_title'], fontweight='bold')
            ax.set_ylabel('Hub Pair Attention', fontsize=FONTS['axis_label'])
            
        else:
            # Boxplot: G1 and G2 side by side per day
            positions = []
            g1_data, g2_data = [], []
            
            for i, day in enumerate(days):
                day_df = df[df['Day'] == day]
                g1_data.append(day_df['G1_Attn'].values)
                g2_data.append(day_df['G2_Attn'].values)
                positions.append(i)
            
            width = 0.35
            for i, day in enumerate(days):
                bp1 = ax.boxplot([g1_data[i]], positions=[i - width/2], widths=width,
                                patch_artist=True, boxprops=dict(facecolor=COLORS['G1'], alpha=0.7))
                bp2 = ax.boxplot([g2_data[i]], positions=[i + width/2], widths=width,
                                patch_artist=True, boxprops=dict(facecolor=COLORS['G2'], alpha=0.7))
            
            ax.set_xticks(positions)
            ax.set_xticklabels([f'Day {d}' for d in days])
            ax.set_title(f'{tissue}\nG1 vs G2 Distribution', fontsize=FONTS['panel_title'], fontweight='bold')
            ax.set_ylabel('Hub Pair Attention', fontsize=FONTS['axis_label'])
        
        # Significance markers
        for day in days:
            if day in stats_df['Day'].values:
                row = stats_df[stats_df['Day'] == day].iloc[0]
                if row['significant']:
                    if panel_label in ['c', 'e']:
                        y_pos = max(g1_meds[days.index(day)], g2_meds[days.index(day)]) * 1.1
                    else:
                        y_pos = ax.get_ylim()[1] * 0.95
                    ax.text(days.index(day) if panel_label in ['d', 'f'] else day, 
                           y_pos, '*', ha='center', fontsize=20, 
                           fontweight='bold', color=COLORS['Significance'])
        
        ax.text(-0.15, 1.1, panel_label, transform=ax.transAxes,
               fontsize=FONTS['panel_label'], fontweight='bold', va='top')
        ax.set_xlabel('Time Point', fontsize=FONTS['axis_label'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        if panel_label == 'c':
            ax.legend(loc='upper left', fontsize=FONTS['legend_text'])
    
    fig.text(0.5, 0.02,
            '* BH-FDR < 0.05 (paired Wilcoxon on n=100 hub pairs)',
            ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'fig4_cf_FIXED.png')
    pdf_path = os.path.join(output_dir, 'fig4_cf_FIXED.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    print(f"\nSaved: {png_path}")
    return fig


def generate_manuscript_table(all_stats, output_dir):
    """
    Generate Table S9 with correct values for manuscript.
    """
    combined = pd.concat([all_stats['Leaf'], all_stats['Root']], ignore_index=True)
    
    # Format for manuscript
    combined['FDR_formatted'] = combined['p_fdr'].apply(lambda x: f"{x:.2e}")
    combined['Fold_formatted'] = combined['fold_change'].apply(lambda x: f"{x:.2f}×")
    combined['Pct_Diff_formatted'] = combined['pct_diff'].apply(lambda x: f"{x:.1f}%")
    
    output_path = os.path.join(output_dir, 'Table_S9_FIXED.csv')
    combined.to_csv(output_path, index=False)
    
    print(f"\nTable S9 saved: {output_path}")
    return combined


def main():
    """Main execution."""
    print("="*70)
    print("FIGURE 4 (c-f): DEFINITIVE FIX")
    print("="*70)
    print("\nThis script uses:")
    print("  1. COMPLETE augmentation regex (7 types)")
    print("  2. PAIR-LEVEL analysis (n=100, not sample-level)")
    print("  3. Paired Wilcoxon statistics (not MWU)")
    
    # Configuration - UPDATE THESE PATHS
    BASE_PATH = r"C:\Users\ms\Desktop\hyper\output\transformer\v3_feature_attention"
    OUTPUT_DIR = r"C:\Users\ms\Desktop\hyper\output\transformer\novility_plot\fixed"
    
    all_results = {}
    all_stats = {}
    
    for tissue in ["Leaf", "Root"]:
        tensor, metadata = load_and_clean_data(tissue, BASE_PATH)
        hub_pairs = define_hub_pairs(tensor, top_k=TOP_K)
        pair_df = compute_pair_level_data(tensor, metadata, hub_pairs, tissue)
        stats_df = compute_statistics(pair_df, tissue)
        
        all_results[tissue] = pair_df
        all_stats[tissue] = stats_df
    
    # Create combined dataframe
    combined_df = pd.concat([all_results['Leaf'], all_results['Root']], ignore_index=True)
    
    # Export source of truth
    source_path = os.path.join(OUTPUT_DIR, 'fig4_source_data_FIXED.csv')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined_df.to_csv(source_path, index=False)
    print(f"\nSource data: {source_path}")
    
    # Generate figure
    fig = plot_figure4(
        all_results['Leaf'], all_stats['Leaf'],
        all_results['Root'], all_stats['Root'],
        OUTPUT_DIR
    )
    
    # Generate table
    table = generate_manuscript_table(all_stats, OUTPUT_DIR)
    
    # Print final summary for manuscript
    print("\n" + "="*70)
    print("MANUSCRIPT VALUES (for Abstract/Results)")
    print("="*70)
    
    for tissue in ["Leaf", "Root"]:
        print(f"\n{tissue}:")
        for _, row in all_stats[tissue].iterrows():
            sig = '✓ SIGNIFICANT' if row['significant'] else ''
            print(f"  Day {int(row['Day'])}: {row['fold_change']:.2f}× stronger coordination, "
                  f"FDR = {row['p_fdr']:.2e} {sig}")
    
    # Key result for abstract
    leaf_d3 = all_stats['Leaf'][all_stats['Leaf']['Day'] == 3].iloc[0]
    print(f"\n>>> ABSTRACT (Leaf Peak Stress Day 3):")
    print(f"    {leaf_d3['fold_change']:.2f}× stronger coordination")
    print(f"    FDR = {leaf_d3['p_fdr']:.2e}")
    
    return combined_df, all_stats


if __name__ == "__main__":
    combined_df, all_stats = main()