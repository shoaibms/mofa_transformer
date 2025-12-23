#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 4 (c-f): Hub-Based Temporal Dynamics of Network Coordination

REFINED VERSION - Plant-Level Statistics with Robustness Checks

Key features:
1. Uses ALL_ORIG attention export (all non-augmented originals)
2. Statistics use PLANTS as the unit of analysis (not pairs)
3. GENOTYPE-BALANCED hub definition (equal weight to G1 and G2)
4. Genotype × Day INTERACTION TEST for timing claim
5. TOP_K SWEEP for robustness assessment
6. Bootstrap CIs for fold-change estimates
7. Proper effect sizes (Cohen's d)

Output files:
- fig4_cf.png / fig4_cf.pdf (main figure)
- fig4_plant_data.csv (source data)
- fig4_statistics.csv (plant-level stats)
- fig4_interaction_test.csv (genotype × day interaction)
- fig4_topk_robustness.csv (TOP_K sweep results)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import h5py
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests
from itertools import product

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

# CRITICAL: Correct 7-type augmentation regex
AUG_REGEX = r"(?:_GP|_WARP|_SCALE|_NOISE|_ADD|_MULT|_MIX)$"

# File tag: "" for test-only, "_ALL_ORIG" for all-originals
ATTENTION_FILE_TAG = "_ALL_ORIG"

# Hub pair configuration
TOP_K = 100  # Primary hub count
TOP_K_SWEEP = [50, 100, 200]  # For robustness check

# Bootstrap configuration
N_BOOTSTRAP = 1000  # Number of bootstrap iterations for CI
N_PERMUTATIONS = 1000  # For interaction test

# Paths - UPDATE THESE TO YOUR SETUP
BASE_PATH = r"C:\Users\ms\Desktop\hyper"
BASE_OUTPUT_DIR = os.path.join(BASE_PATH, "output", "transformer", "v3_feature_attention")
OUTPUT_DIR = os.path.join(BASE_PATH, "output", "transformer", "novility_plot", "final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

COLORS = {
    'G1': '#00FA9A',
    'G2': '#48D1CC',
    'G1_Light': '#98FB98',
    'G2_Light': '#B0E0E6',
    'Background': '#FFFFFF',
    'Panel_Background': '#FFFFFF',
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
    'annotation': 12,
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': FONTS['axis_label'],
    'xtick.labelsize': FONTS['tick_label'],
    'ytick.labelsize': FONTS['tick_label'],
    'figure.facecolor': COLORS['Background'],
    'axes.facecolor': COLORS['Panel_Background'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})


# =============================================================================
# DATA LOADING
# =============================================================================

def get_file_paths(tissue_name):
    """Generate file paths for a tissue."""
    return {
        'h5': os.path.join(
            BASE_OUTPUT_DIR, "results",
            f"raw_attention_data_{tissue_name}{ATTENTION_FILE_TAG}.h5"
        ),
        'metadata_feather': os.path.join(
            BASE_OUTPUT_DIR, "results",
            f"raw_attention_metadata_{tissue_name}{ATTENTION_FILE_TAG}.feather"
        ),
        'metadata_csv': os.path.join(
            BASE_OUTPUT_DIR, "results",
            f"raw_attention_metadata_{tissue_name}{ATTENTION_FILE_TAG}.csv"
        ),
    }


def load_tissue_data(tissue_name):
    """
    Load attention data for a tissue.
    Returns tensor (averaged over heads) and metadata.
    """
    print(f"\n{'='*60}")
    print(f"Loading {tissue_name.upper()}")
    print("="*60)
    
    paths = get_file_paths(tissue_name)
    
    # Load HDF5
    if not os.path.exists(paths['h5']):
        raise FileNotFoundError(f"HDF5 not found: {paths['h5']}")
    
    with h5py.File(paths['h5'], 'r') as f:
        attn_tensor = f['attention_spec_to_metab'][:]
    
    print(f"Tensor shape: {attn_tensor.shape}")
    
    # Load metadata (try Feather first, then CSV)
    if os.path.exists(paths['metadata_feather']):
        metadata_df = pd.read_feather(paths['metadata_feather'])
        if 'Row_names' in metadata_df.columns:
            metadata_df = metadata_df.set_index('Row_names')
    elif os.path.exists(paths['metadata_csv']):
        metadata_df = pd.read_csv(paths['metadata_csv'], index_col='Row_names')
    else:
        raise FileNotFoundError(f"Metadata not found for {tissue_name}")
    
    print(f"Metadata samples: {len(metadata_df)}")
    
    # Verify no augmented samples (should be clean from ALL_ORIG export)
    row_ids = pd.Series(metadata_df.index.astype(str))
    n_aug = row_ids.str.contains(AUG_REGEX, regex=True, na=False).sum()
    if n_aug > 0:
        print(f"WARNING: Found {n_aug} augmented samples - filtering...")
        mask_orig = ~row_ids.str.contains(AUG_REGEX, regex=True, na=False).values
        metadata_df = metadata_df.loc[mask_orig]
        attn_tensor = attn_tensor[mask_orig]
    
    print(f"Final samples: {len(metadata_df)}")
    
    # Standardize metadata columns
    metadata_df['Genotype'] = metadata_df['Genotype'].astype(str).str.strip().replace(
        {'Gladius': 'G1', 'DAS5_003811': 'G2'})
    
    # Handle Day vs Time_Point column (robust to either)
    if 'Day' in metadata_df.columns:
        metadata_df['Time_Point'] = pd.to_numeric(metadata_df['Day'], errors='coerce').astype('Int64')
    elif 'Time_Point' in metadata_df.columns:
        metadata_df['Time_Point'] = pd.to_numeric(metadata_df['Time_Point'], errors='coerce').astype('Int64')
    else:
        raise ValueError(f"Neither 'Day' nor 'Time_Point' column found in {tissue_name} metadata")
    
    # Handle Treatment column
    if 'Treatment' in metadata_df.columns:
        metadata_df['Treatment'] = pd.to_numeric(metadata_df['Treatment'], errors='coerce')
    
    # Filter to stressed samples
    stressed_mask = (metadata_df['Treatment'] == 1).values
    metadata_stressed = metadata_df[stressed_mask].copy()
    tensor_stressed = attn_tensor[stressed_mask]
    
    print(f"Stressed samples: {len(metadata_stressed)}")
    
    # Sample counts per condition
    for day in sorted(metadata_stressed['Time_Point'].dropna().unique()):
        n_g1 = ((metadata_stressed['Genotype'] == 'G1') & (metadata_stressed['Time_Point'] == day)).sum()
        n_g2 = ((metadata_stressed['Genotype'] == 'G2') & (metadata_stressed['Time_Point'] == day)).sum()
        print(f"  Day {int(day)}: G1={n_g1}, G2={n_g2}")
    
    # Average over attention heads
    tensor_mean = np.mean(tensor_stressed, axis=1)
    print(f"Tensor after head averaging: {tensor_mean.shape}")
    
    return tensor_mean, metadata_stressed


# =============================================================================
# GENOTYPE-BALANCED HUB DEFINITION
# =============================================================================

def define_hub_pairs_balanced(tensor, metadata, top_k=100):
    """
    Define hub pairs using GENOTYPE-BALANCED aggregation.
    
    This ensures each genotype contributes equally to hub definition,
    preventing bias toward the genotype with stronger attention.
    
    Method:
    1. Compute mean attention map for G1 stressed samples
    2. Compute mean attention map for G2 stressed samples
    3. Pooled map = (mean_G1 + mean_G2) / 2  (equal weight)
    4. Select top-K pairs from pooled map
    
    This is more defensible than simple pooling when sample sizes differ.
    """
    print(f"\nDefining genotype-balanced hub pairs (top_k={top_k})...")
    
    # Get genotype masks
    g1_mask = (metadata['Genotype'] == 'G1').values
    g2_mask = (metadata['Genotype'] == 'G2').values
    
    n_g1 = g1_mask.sum()
    n_g2 = g2_mask.sum()
    print(f"  G1 samples: {n_g1}, G2 samples: {n_g2}")
    
    # Compute per-genotype mean attention maps
    mean_g1 = np.mean(tensor[g1_mask], axis=0) if n_g1 > 0 else np.zeros_like(tensor[0])
    mean_g2 = np.mean(tensor[g2_mask], axis=0) if n_g2 > 0 else np.zeros_like(tensor[0])
    
    # Balanced pooled map (equal weight to each genotype)
    pooled_balanced = (mean_g1 + mean_g2) / 2
    
    # Select top-K pairs
    n_spec, n_metab = pooled_balanced.shape
    flat_indices = np.argsort(pooled_balanced.flatten())[-top_k:]
    hub_pairs = [(int(i // n_metab), int(i % n_metab)) for i in flat_indices]
    
    print(f"  Defined {top_k} genotype-balanced hub pairs")
    
    return hub_pairs


def define_hub_pairs_simple(tensor, top_k=100):
    """
    Simple pooled hub definition (for comparison in robustness checks).
    """
    pooled_mean = np.mean(tensor, axis=0)
    n_spec, n_metab = pooled_mean.shape
    flat_indices = np.argsort(pooled_mean.flatten())[-top_k:]
    hub_pairs = [(int(i // n_metab), int(i % n_metab)) for i in flat_indices]
    return hub_pairs


# =============================================================================
# PLANT-LEVEL COORDINATION SCORE
# =============================================================================

def compute_plant_coordination_scores(tensor, metadata, hub_pairs, tissue_name):
    """
    Compute coordination score for each PLANT (not each pair).
    
    Coordination Score = mean attention across top-k hub pairs for that plant.
    
    This is the PRIMARY metric for statistical inference.
    Unit of analysis = plants (n = number of plants per condition)
    """
    print(f"\nComputing plant-level coordination scores for {tissue_name}...")
    
    n_samples = tensor.shape[0]
    
    # Compute per-plant coordination score
    coordination_scores = []
    for sample_idx in range(n_samples):
        # Mean attention across all hub pairs for this plant
        sample_hub_attn = [
            tensor[sample_idx, spec_idx, metab_idx]
            for spec_idx, metab_idx in hub_pairs
        ]
        score = np.mean(sample_hub_attn)
        coordination_scores.append(score)
    
    # Create result dataframe
    result_df = metadata.copy()
    result_df['Coordination_Score'] = coordination_scores
    result_df['Tissue'] = tissue_name
    
    print(f"  Computed scores for {n_samples} plants")
    
    return result_df


# =============================================================================
# STATISTICS (PLANT-LEVEL)
# =============================================================================

def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def bootstrap_fold_change(g1_scores, g2_scores, n_boot=1000, ci=95):
    """
    Bootstrap confidence interval for median fold-change (G1/G2).
    """
    ratios = []
    for _ in range(n_boot):
        g1_boot = np.random.choice(g1_scores, size=len(g1_scores), replace=True)
        g2_boot = np.random.choice(g2_scores, size=len(g2_scores), replace=True)
        if np.median(g2_boot) > 0:
            ratios.append(np.median(g1_boot) / np.median(g2_boot))
    
    if not ratios:
        return np.nan, np.nan
    
    lower = np.percentile(ratios, (100 - ci) / 2)
    upper = np.percentile(ratios, 100 - (100 - ci) / 2)
    return lower, upper


def compute_plant_level_statistics(data_df, tissue_name):
    """
    Compute PLANT-LEVEL statistics for G1 vs G2 comparison.
    
    Primary test: Mann-Whitney U (plants as n)
    Effect size: Cohen's d
    Uncertainty: Bootstrap 95% CI on fold-change
    """
    print(f"\nComputing plant-level statistics for {tissue_name}...")
    
    days = sorted(data_df['Time_Point'].dropna().unique())
    stats_results = []
    pvals = []
    
    for day in days:
        day_df = data_df[data_df['Time_Point'] == day]
        
        g1_scores = day_df[day_df['Genotype'] == 'G1']['Coordination_Score'].values
        g2_scores = day_df[day_df['Genotype'] == 'G2']['Coordination_Score'].values
        
        n_g1, n_g2 = len(g1_scores), len(g2_scores)
        
        if n_g1 < 2 or n_g2 < 2:
            print(f"  Day {int(day)}: Insufficient samples (G1={n_g1}, G2={n_g2})")
            continue
        
        # Mann-Whitney U test (plant-level)
        stat, p_mwu = mannwhitneyu(g1_scores, g2_scores, alternative='two-sided')
        pvals.append(p_mwu)
        
        # Effect sizes
        median_g1 = np.median(g1_scores)
        median_g2 = np.median(g2_scores)
        mean_g1 = np.mean(g1_scores)
        mean_g2 = np.mean(g2_scores)
        fold_change = median_g1 / median_g2 if median_g2 > 0 else np.nan
        
        # Cohen's d
        d = cohens_d(g1_scores, g2_scores)
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_fold_change(g1_scores, g2_scores, n_boot=N_BOOTSTRAP)
        
        stats_results.append({
            'Tissue': tissue_name,
            'Day': int(day),
            'n_G1': n_g1,
            'n_G2': n_g2,
            'median_G1': median_g1,
            'median_G2': median_g2,
            'mean_G1': mean_g1,
            'mean_G2': mean_g2,
            'fold_change': fold_change,
            'fold_change_CI_lower': ci_lower,
            'fold_change_CI_upper': ci_upper,
            'cohens_d': d,
            'p_MWU': p_mwu,
        })
    
    if not stats_results:
        return pd.DataFrame()
    
    # FDR correction across days within this tissue
    _, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    
    stats_df = pd.DataFrame(stats_results)
    stats_df['p_FDR'] = pvals_fdr
    stats_df['significant'] = stats_df['p_FDR'] < 0.05
    
    # Print results
    print(f"\n  PLANT-LEVEL RESULTS - {tissue_name}:")
    print("  " + "-"*70)
    for _, row in stats_df.iterrows():
        sig_str = '***' if row['significant'] else ''
        print(f"  Day {int(row['Day'])}: n=({row['n_G1']},{row['n_G2']}), "
              f"Fold={row['fold_change']:.2f}× [95% CI: {row['fold_change_CI_lower']:.2f}-{row['fold_change_CI_upper']:.2f}], "
              f"d={row['cohens_d']:.2f}, FDR={row['p_FDR']:.3f} {sig_str}")
    
    return stats_df


# =============================================================================
# GENOTYPE × DAY INTERACTION TEST
# =============================================================================

def compute_interaction_test(data_df, tissue_name, n_permutations=1000):
    """
    Test for Genotype × Day interaction using permutation-based approach.
    
    This tests whether the TEMPORAL PATTERN differs between genotypes,
    which is the core of the "timing architecture" claim.
    
    Method: 
    - Compute observed interaction statistic (difference in slopes or variance explained)
    - Permute genotype labels within each day
    - Compare observed to null distribution
    
    Returns: p-value for interaction, effect size
    """
    print(f"\nComputing Genotype × Day interaction test for {tissue_name}...")
    
    df = data_df.copy()
    days = sorted(df['Time_Point'].dropna().unique())
    
    if len(days) < 2:
        print("  Insufficient time points for interaction test")
        return None
    
    # Compute observed interaction statistic
    # We use the variance in fold-change across days as the interaction metric
    # If genotypes have different temporal patterns, fold-change will vary by day
    
    def compute_interaction_stat(df):
        """Compute interaction statistic: variance in day-wise fold-changes."""
        fold_changes = []
        for day in days:
            day_df = df[df['Time_Point'] == day]
            g1_mean = day_df[day_df['Genotype'] == 'G1']['Coordination_Score'].mean()
            g2_mean = day_df[day_df['Genotype'] == 'G2']['Coordination_Score'].mean()
            if g2_mean > 0:
                fold_changes.append(g1_mean / g2_mean)
        
        if len(fold_changes) < 2:
            return 0
        return np.var(fold_changes)  # Higher variance = stronger interaction
    
    observed_stat = compute_interaction_stat(df)
    
    # Permutation test: shuffle genotype labels within each day
    null_stats = []
    for _ in range(n_permutations):
        df_perm = df.copy()
        for day in days:
            day_mask = df_perm['Time_Point'] == day
            day_genotypes = df_perm.loc[day_mask, 'Genotype'].values.copy()
            np.random.shuffle(day_genotypes)
            df_perm.loc[day_mask, 'Genotype'] = day_genotypes
        null_stats.append(compute_interaction_stat(df_perm))
    
    # P-value: proportion of null stats >= observed
    p_value = (np.sum(np.array(null_stats) >= observed_stat) + 1) / (n_permutations + 1)
    
    # Effect size: how many SDs above null mean
    null_mean = np.mean(null_stats)
    null_std = np.std(null_stats)
    effect_size = (observed_stat - null_mean) / null_std if null_std > 0 else 0
    
    result = {
        'Tissue': tissue_name,
        'observed_interaction_stat': observed_stat,
        'null_mean': null_mean,
        'null_std': null_std,
        'effect_size_z': effect_size,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'significant': p_value < 0.05,
    }
    
    print(f"  Interaction test: stat={observed_stat:.4f}, null={null_mean:.4f}±{null_std:.4f}, "
          f"z={effect_size:.2f}, p={p_value:.4f} {'***' if p_value < 0.05 else ''}")
    
    return result


# =============================================================================
# TOP_K ROBUSTNESS SWEEP
# =============================================================================

def run_topk_robustness_sweep(tensor, metadata, tissue_name, top_k_values=[50, 100, 200]):
    """
    Test robustness of key claims across different TOP_K values.
    
    The timing claim should hold regardless of whether we use 50, 100, or 200 hub pairs.
    """
    print(f"\nRunning TOP_K robustness sweep for {tissue_name}...")
    
    results = []
    
    for k in top_k_values:
        print(f"\n  TOP_K = {k}:")
        
        # Define hubs with this k
        hub_pairs = define_hub_pairs_balanced(tensor, metadata, top_k=k)
        
        # Compute coordination scores
        plant_df = compute_plant_coordination_scores(tensor, metadata, hub_pairs, tissue_name)
        
        # Get fold-changes per day
        days = sorted(plant_df['Time_Point'].dropna().unique())
        for day in days:
            day_df = plant_df[plant_df['Time_Point'] == day]
            g1_scores = day_df[day_df['Genotype'] == 'G1']['Coordination_Score'].values
            g2_scores = day_df[day_df['Genotype'] == 'G2']['Coordination_Score'].values
            
            if len(g1_scores) >= 2 and len(g2_scores) >= 2:
                median_g1 = np.median(g1_scores)
                median_g2 = np.median(g2_scores)
                fold_change = median_g1 / median_g2 if median_g2 > 0 else np.nan
                _, p_mwu = mannwhitneyu(g1_scores, g2_scores, alternative='two-sided')
                
                results.append({
                    'Tissue': tissue_name,
                    'TOP_K': k,
                    'Day': int(day),
                    'fold_change': fold_change,
                    'p_MWU': p_mwu,
                    'G1_higher': median_g1 > median_g2,
                })
                
                print(f"    Day {int(day)}: Fold={fold_change:.2f}×, p={p_mwu:.3f}")
    
    return pd.DataFrame(results)


# =============================================================================
# PAIR-LEVEL DESCRIPTIVE (NO P-VALUES)
# =============================================================================

def compute_pair_level_descriptive(tensor, metadata, hub_pairs, tissue_name):
    """
    Compute pair-level attention for VISUALIZATION only.
    NO p-values - this is descriptive/supplementary.
    """
    print(f"\nComputing pair-level descriptive for {tissue_name}...")
    
    days = sorted(metadata['Time_Point'].dropna().unique())
    results = []
    
    for day in days:
        g1_mask = ((metadata['Genotype'] == 'G1') & (metadata['Time_Point'] == day)).values
        g2_mask = ((metadata['Genotype'] == 'G2') & (metadata['Time_Point'] == day)).values
        
        for pair_idx, (spec_idx, metab_idx) in enumerate(hub_pairs):
            g1_attn = np.mean(tensor[g1_mask, spec_idx, metab_idx]) if g1_mask.sum() > 0 else np.nan
            g2_attn = np.mean(tensor[g2_mask, spec_idx, metab_idx]) if g2_mask.sum() > 0 else np.nan
            
            results.append({
                'Tissue': tissue_name,
                'Day': int(day),
                'Pair_ID': pair_idx,
                'G1_Attn': g1_attn,
                'G2_Attn': g2_attn,
                'G1_higher': g1_attn > g2_attn if not np.isnan(g1_attn) and not np.isnan(g2_attn) else None,
            })
    
    df = pd.DataFrame(results)
    
    # Summary: % of pairs where G1 > G2
    for day in days:
        day_df = df[df['Day'] == int(day)]
        pct_g1_higher = day_df['G1_higher'].sum() / len(day_df) * 100 if len(day_df) > 0 else 0
        print(f"  Day {int(day)}: {pct_g1_higher:.1f}% of hub pairs have higher attention in G1")
    
    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_figure4(plant_data, plant_stats, pair_data, output_dir):
    """
    Create Figure 4 c-f.
    
    Layout:
    - c: Leaf coordination trajectory (G1 vs G2, plant-level)
    - d: Leaf boxplot (G1 vs G2 distribution)
    - e: Root coordination trajectory
    - f: Root boxplot
    """
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 6.5))
    
    tissues = ['Leaf', 'Root']
    
    for t_idx, tissue in enumerate(tissues):
        tissue_data = plant_data[plant_data['Tissue'] == tissue]
        tissue_stats = plant_stats[plant_stats['Tissue'] == tissue]
        
        ax_traj = axes[t_idx * 2]
        ax_box = axes[t_idx * 2 + 1]
        panel_traj = 'c' if tissue == 'Leaf' else 'e'
        panel_box = 'd' if tissue == 'Leaf' else 'f'
        
        days = sorted(tissue_data['Time_Point'].dropna().unique())
        
        # --- Trajectory Plot ---
        g1_medians, g2_medians = [], []
        g1_errs, g2_errs = [], []
        
        for day in days:
            day_df = tissue_data[tissue_data['Time_Point'] == day]
            g1_vals = day_df[day_df['Genotype'] == 'G1']['Coordination_Score'].values
            g2_vals = day_df[day_df['Genotype'] == 'G2']['Coordination_Score'].values
            
            g1_medians.append(np.median(g1_vals) if len(g1_vals) > 0 else np.nan)
            g2_medians.append(np.median(g2_vals) if len(g2_vals) > 0 else np.nan)
            
            # IQR for error bars
            if len(g1_vals) > 0:
                g1_errs.append([np.median(g1_vals) - np.percentile(g1_vals, 25),
                               np.percentile(g1_vals, 75) - np.median(g1_vals)])
            else:
                g1_errs.append([0, 0])
            
            if len(g2_vals) > 0:
                g2_errs.append([np.median(g2_vals) - np.percentile(g2_vals, 25),
                               np.percentile(g2_vals, 75) - np.median(g2_vals)])
            else:
                g2_errs.append([0, 0])
        
        g1_errs = np.array(g1_errs).T
        g2_errs = np.array(g2_errs).T
        
        ax_traj.errorbar(days, g1_medians, yerr=g1_errs,
                        color=COLORS['G1'], marker='o', linestyle='-',
                        linewidth=2.5, markersize=8, capsize=4, capthick=1.5,
                        markeredgecolor='white', markeredgewidth=0.5,
                        label='G1 (Tolerant)', alpha=0.95)
        ax_traj.errorbar(days, g2_medians, yerr=g2_errs,
                        color=COLORS['G2'], marker='s', linestyle='--',
                        linewidth=2.5, markersize=8, capsize=4, capthick=1.5,
                        markeredgecolor='white', markeredgewidth=0.5,
                        label='G2 (Susceptible)', alpha=0.95)
        
        # Significance markers
        sig_y_positions = []
        for i, day in enumerate(days):
            row = tissue_stats[tissue_stats['Day'] == int(day)]
            if len(row) > 0 and row.iloc[0]['significant']:
                y_max = max(g1_medians[i] + g1_errs[1, i], g2_medians[i] + g2_errs[1, i])
                sig_y = y_max * 1.1
                sig_y_positions.append(sig_y)
                ax_traj.text(day, sig_y, '*', ha='center', va='bottom',
                            fontsize=FONTS['panel_label'], fontweight='bold',
                            color=COLORS['Significance'])
        
        # Extend y-axis to accommodate significance markers
        if sig_y_positions:
            current_ylim = ax_traj.get_ylim()
            max_sig_y = max(sig_y_positions) * 1.15  # Add 15% padding above asterisk
            ax_traj.set_ylim(current_ylim[0], max(current_ylim[1], max_sig_y))
        
        ax_traj.set_title(f'{tissue}\nCoordination Strength', fontsize=FONTS['panel_title'], fontweight='bold')
        ax_traj.set_xlabel('Time Point', fontsize=FONTS['axis_label'])
        ax_traj.set_ylabel('Coordination Score\n(mean hub-pair attention)', fontsize=FONTS['axis_label'])
        ax_traj.set_xticks(days)
        ax_traj.set_xlim(min(days) - 0.3, max(days) + 0.3)
        ax_traj.grid(True, axis='y', linestyle='--', alpha=0.5, color=COLORS['Grid'])
        ax_traj.text(-0.15, 1.1, panel_traj, transform=ax_traj.transAxes,
                    fontsize=FONTS['panel_label'], fontweight='bold', va='top')
        
        if panel_traj == 'c':
            ax_traj.legend(loc='upper right', fontsize=FONTS['legend_text'], frameon=False)
        
        # --- Boxplot ---
        positions_g1 = [d - 0.15 for d in days]
        positions_g2 = [d + 0.15 for d in days]
        
        g1_data_by_day = [tissue_data[(tissue_data['Time_Point'] == d) & (tissue_data['Genotype'] == 'G1')]['Coordination_Score'].values for d in days]
        g2_data_by_day = [tissue_data[(tissue_data['Time_Point'] == d) & (tissue_data['Genotype'] == 'G2')]['Coordination_Score'].values for d in days]
        
        bp1 = ax_box.boxplot(g1_data_by_day, positions=positions_g1, widths=0.25,
                            patch_artist=True, boxprops=dict(facecolor=COLORS['G1'], alpha=0.7),
                            medianprops=dict(color='black', linewidth=1.5))
        bp2 = ax_box.boxplot(g2_data_by_day, positions=positions_g2, widths=0.25,
                            patch_artist=True, boxprops=dict(facecolor=COLORS['G2'], alpha=0.7),
                            medianprops=dict(color='black', linewidth=1.5))
        
        # Significance markers on boxplot
        box_sig_y_positions = []
        for i, day in enumerate(days):
            row = tissue_stats[tissue_stats['Day'] == int(day)]
            if len(row) > 0 and row.iloc[0]['significant']:
                y_max = max(np.max(g1_data_by_day[i]) if len(g1_data_by_day[i]) > 0 else 0,
                           np.max(g2_data_by_day[i]) if len(g2_data_by_day[i]) > 0 else 0)
                sig_y = y_max * 1.15
                box_sig_y_positions.append(sig_y)
                ax_box.text(day, sig_y, '*', ha='center', va='bottom',
                           fontsize=FONTS['panel_label'], fontweight='bold',
                           color=COLORS['Significance'])
        
        # Extend y-axis to accommodate significance markers
        if box_sig_y_positions:
            current_ylim = ax_box.get_ylim()
            max_sig_y = max(box_sig_y_positions) * 1.15  # Add 15% padding above asterisk
            ax_box.set_ylim(current_ylim[0], max(current_ylim[1], max_sig_y))
        
        ax_box.set_title(f'{tissue}\nG1 vs G2 Distribution', fontsize=FONTS['panel_title'], fontweight='bold')
        ax_box.set_xlabel('Time Point', fontsize=FONTS['axis_label'])
        ax_box.set_ylabel('Coordination Score', fontsize=FONTS['axis_label'])
        ax_box.set_xticks(days)
        ax_box.set_xlim(min(days) - 0.5, max(days) + 0.5)
        ax_box.grid(True, axis='y', linestyle='--', alpha=0.5, color=COLORS['Grid'])
        ax_box.text(-0.15, 1.1, panel_box, transform=ax_box.transAxes,
                   fontsize=FONTS['panel_label'], fontweight='bold', va='top')
        
        # Add sample sizes at top of plot (inside, just below border)
        for i, day in enumerate(days):
            n_g1 = len(g1_data_by_day[i])
            n_g2 = len(g2_data_by_day[i])
            ax_box.text(
                day, 0.84, f'n=({n_g1},{n_g2})',
                transform=ax_box.get_xaxis_transform(),
                ha='center', va='top', fontsize=11, color='gray'
            )
        
        # Force clean ticks
        ax_box.set_xticks(days)
        ax_box.set_xticklabels([str(int(d)) for d in days])
    
    # Figure caption
    fig.text(0.5, 0.02,
            '* FDR < 0.05 (Mann-Whitney U, plants as unit of analysis). '
            'Error bars: IQR. Hub pairs defined using genotype-balanced aggregation.',
            ha='center', fontsize=13, style='italic', color=COLORS['Text_Dark'])
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # Save
    png_path = os.path.join(output_dir, 'fig4_cf.png')
    pdf_path = os.path.join(output_dir, 'fig4_cf.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    
    print(f"\nFigure saved: {png_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("FIGURE 4 (c-f): PLANT-LEVEL COORDINATION ANALYSIS (REFINED)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Attention file tag: '{ATTENTION_FILE_TAG}'")
    print(f"  Hub pairs: {TOP_K} (genotype-balanced)")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  Permutations for interaction test: {N_PERMUTATIONS}")
    print(f"  TOP_K sweep values: {TOP_K_SWEEP}")
    
    all_plant_data = []
    all_plant_stats = []
    all_pair_data = []
    all_interaction_tests = []
    all_topk_robustness = []
    
    for tissue in ["Leaf", "Root"]:
        # Load data
        tensor, metadata = load_tissue_data(tissue)
        
        # Define hub pairs (GENOTYPE-BALANCED)
        hub_pairs = define_hub_pairs_balanced(tensor, metadata, top_k=TOP_K)
        
        # Plant-level analysis (PRIMARY)
        plant_df = compute_plant_coordination_scores(tensor, metadata, hub_pairs, tissue)
        plant_stats = compute_plant_level_statistics(plant_df, tissue)
        
        # Interaction test (TIMING CLAIM)
        interaction_result = compute_interaction_test(plant_df, tissue, n_permutations=N_PERMUTATIONS)
        if interaction_result:
            all_interaction_tests.append(interaction_result)
        
        # TOP_K robustness sweep
        topk_df = run_topk_robustness_sweep(tensor, metadata, tissue, top_k_values=TOP_K_SWEEP)
        all_topk_robustness.append(topk_df)
        
        # Pair-level analysis (DESCRIPTIVE ONLY)
        pair_df = compute_pair_level_descriptive(tensor, metadata, hub_pairs, tissue)
        
        all_plant_data.append(plant_df)
        all_plant_stats.append(plant_stats)
        all_pair_data.append(pair_df)
    
    # Combine
    combined_plant_data = pd.concat(all_plant_data, ignore_index=True)
    combined_plant_stats = pd.concat(all_plant_stats, ignore_index=True)
    combined_pair_data = pd.concat(all_pair_data, ignore_index=True)
    combined_topk = pd.concat(all_topk_robustness, ignore_index=True)
    interaction_df = pd.DataFrame(all_interaction_tests)
    
    # Save source data
    combined_plant_data.to_csv(os.path.join(OUTPUT_DIR, 'fig4_plant_data.csv'), index=False)
    combined_plant_stats.to_csv(os.path.join(OUTPUT_DIR, 'fig4_statistics.csv'), index=False)
    combined_pair_data.to_csv(os.path.join(OUTPUT_DIR, 'fig4_pair_descriptive.csv'), index=False)
    interaction_df.to_csv(os.path.join(OUTPUT_DIR, 'fig4_interaction_test.csv'), index=False)
    combined_topk.to_csv(os.path.join(OUTPUT_DIR, 'fig4_topk_robustness.csv'), index=False)
    
    print(f"\nSource data saved to: {OUTPUT_DIR}")
    
    # Plot
    fig = plot_figure4(combined_plant_data, combined_plant_stats, combined_pair_data, OUTPUT_DIR)
    
    # Print manuscript values
    print("\n" + "="*70)
    print("MANUSCRIPT VALUES (Plant-Level Statistics)")
    print("="*70)
    
    for tissue in ["Leaf", "Root"]:
        print(f"\n{tissue}:")
        tissue_stats = combined_plant_stats[combined_plant_stats['Tissue'] == tissue]
        for _, row in tissue_stats.iterrows():
            sig_str = 'SIGNIFICANT' if row['significant'] else ''
            print(f"  Day {int(row['Day'])}: {row['fold_change']:.2f}× "
                  f"[95% CI: {row['fold_change_CI_lower']:.2f}-{row['fold_change_CI_upper']:.2f}], "
                  f"Cohen's d = {row['cohens_d']:.2f}, "
                  f"p = {row['p_MWU']:.3f}, FDR = {row['p_FDR']:.3f} {sig_str}")
    
    # Interaction test summary
    print("\n" + "="*70)
    print("INTERACTION TEST (Genotype × Day)")
    print("="*70)
    for _, row in interaction_df.iterrows():
        sig_str = 'SIGNIFICANT' if row['significant'] else ''
        print(f"  {row['Tissue']}: z = {row['effect_size_z']:.2f}, p = {row['p_value']:.4f} {sig_str}")
    
    # TOP_K robustness summary
    print("\n" + "="*70)
    print("TOP_K ROBUSTNESS CHECK")
    print("="*70)
    print("\nDoes the G1 > G2 pattern hold across different TOP_K values?")
    for tissue in ["Leaf", "Root"]:
        print(f"\n  {tissue}:")
        tissue_topk = combined_topk[combined_topk['Tissue'] == tissue]
        for k in TOP_K_SWEEP:
            k_data = tissue_topk[tissue_topk['TOP_K'] == k]
            all_g1_higher = k_data['G1_higher'].all()
            status = "All days G1 > G2" if all_g1_higher else "Pattern inconsistent"
            print(f"    TOP_K={k}: {status}")
    
    # Key result for abstract
    print("\n" + "="*70)
    print("FOR ABSTRACT")
    print("="*70)
    for tissue in ["Leaf", "Root"]:
        tissue_stats = combined_plant_stats[combined_plant_stats['Tissue'] == tissue]
        if len(tissue_stats) > 0:
            peak_row = tissue_stats.loc[tissue_stats['fold_change'].idxmax()]
            print(f"  {tissue} (Day {int(peak_row['Day'])}): {peak_row['fold_change']:.1f}-fold "
                  f"[95% CI: {peak_row['fold_change_CI_lower']:.1f}-{peak_row['fold_change_CI_upper']:.1f}], "
                  f"FDR = {peak_row['p_FDR']:.3f}")
    
    # Interaction for abstract
    if len(interaction_df) > 0:
        print("\n  Timing claim (Genotype × Day interaction):")
        for _, row in interaction_df.iterrows():
            if row['significant']:
                print(f"    {row['Tissue']}: p = {row['p_value']:.3f} (significant temporal pattern difference)")
    
    return combined_plant_data, combined_plant_stats, interaction_df, combined_topk


if __name__ == "__main__":
    plant_data, plant_stats, interaction_df, topk_df = main()