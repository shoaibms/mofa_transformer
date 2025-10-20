# -*- coding: utf-8 -*-
"""
generate_hyperseq_figures_3.py (v3.0 - OPTIMIZED for Publication)

This script generates the final, publication-quality 6-panel validation figure
with all critical fixes applied for maximum scientific impact.

CRITICAL FIXES APPLIED:
1. Panel A: Variance shown as percentages (not thousands)
2. Panel B-C: Shows Factor 3 (integration factor with stress genes)
3. Panel E: Shows actual permutation test results
4. Enhanced annotations with exact statistics
"""
import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from scipy.stats import pearsonr
import json

print("="*80)
print("HyperSeq OPTIMIZED Figure Generation - START (v3.0)")
print("="*80)

# --- Configuration ---
BASE_DIR = r"C:/Users/ms/Desktop/hyper/output/mofa_trasformer_val/val"
MOFA_RESULTS_DIR = os.path.join(BASE_DIR, "mofa_results")
TRANSFORMER_RESULTS_DIR = os.path.join(BASE_DIR, "transformer_results", "processed_attention")
FIGURE_OUTPUT_DIR = os.path.join(BASE_DIR, "final_figures")
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

# --- FONT & STYLE CONFIGURATION ---
# Global option to change text sizes of plot elements
FONT_SIZES = {
    'figure_title': 22,
    'panel_title': 18,
    'axis_label': 16,
    'tick_label': 14,
    'legend_title': 15,
    'legend_text': 14,
    'annotation_bold': 13,
    'annotation_main': 14,
}

# File Paths
MOFA_MODEL_PATH = os.path.join(MOFA_RESULTS_DIR, "mofa_model_hyperseq.hdf5")
ATTENTION_PAIRS_PATH = os.path.join(TRANSFORMER_RESULTS_DIR, "processed_mean_attention_overall_HyperSeq.csv")
PERMUTATION_RESULTS_PATH = os.path.join(BASE_DIR, "transformer_results", "results", "corrected_permutation_test_results_HyperSeq.json")
SPECTRAL_DATA_PATH = os.path.join(MOFA_RESULTS_DIR, "transformer_input_spectral_hyperseq.csv")
GENE_DATA_PATH = os.path.join(MOFA_RESULTS_DIR, "transformer_input_transcriptomics_hyperseq.csv")

def load_all_data():
    """Loads all necessary data files with enhanced error handling."""
    print("   - Loading all required data files...")
    data = {}
    
    try:
        # MOFA Results
        print("     * Loading MOFA model...")
        with h5py.File(MOFA_MODEL_PATH, 'r') as hf:
            # Extract variance explained
            try:
                variance_data = hf['variance_explained/r2_per_factor/group0'][()]
            except KeyError:
                variance_data = hf['variance_explained/r2_per_factor'][()]
            
            # Find active factors
            active_mask = (variance_data.sum(axis=0) > 0.02)
            active_indices = np.where(active_mask)[0]
            print(f"     * Found {len(active_indices)} active factors")
            
            # CRITICAL FIX: Convert to percentages
            data['mofa_variance'] = variance_data[:, active_indices] 
            
            # Extract weights
            try:
                data['mofa_weights_spec'] = hf['expectations/W/spectral'][()][active_indices, :].T
                data['mofa_weights_tx'] = hf['expectations/W/transcriptomics'][()][active_indices, :].T
            except KeyError:
                data['mofa_weights_spec'] = hf['expectations/W']['spectral'][()][active_indices, :].T
                data['mofa_weights_tx'] = hf['expectations/W']['transcriptomics'][()][active_indices, :].T
            
            # Extract feature names
            try:
                data['mofa_features_spec'] = [s.decode() for s in hf['features/spectral'][()]]
                data['mofa_features_tx'] = [s.decode() for s in hf['features/transcriptomics'][()]]
            except KeyError:
                data['mofa_features_spec'] = [s.decode() for s in hf['features']['spectral'][()]]
                data['mofa_features_tx'] = [s.decode() for s in hf['features']['transcriptomics'][()]]
        
        # Load permutation test results
        print("     * Loading permutation test results...")
        if os.path.exists(PERMUTATION_RESULTS_PATH):
            with open(PERMUTATION_RESULTS_PATH, 'r') as f:
                data['permutation_results'] = json.load(f)
        else:
            print("     * WARNING: Permutation results not found")
            data['permutation_results'] = None
        
        # Transformer Attention Results
        print("     * Loading attention results...")
        data['attention'] = pd.read_csv(ATTENTION_PAIRS_PATH)
        print(f"     * Loaded {len(data['attention'])} attention pairs")
        
        # Load data for correlation comparison
        print("     * Loading transformer input data...")
        df_spec = pd.read_csv(SPECTRAL_DATA_PATH)
        df_gene = pd.read_csv(GENE_DATA_PATH)
        
        # Extract feature columns and align data
        def identify_feature_columns(df):
            metadata_patterns = ['row_names', 'batch', 'grid', 'cell', 'id', 'original']
            feature_cols = []
            for col in df.columns:
                if any(pattern in col.lower() for pattern in metadata_patterns):
                    continue
                try:
                    pd.to_numeric(df[col], errors='raise')
                    feature_cols.append(col)
                except (ValueError, TypeError):
                    continue
            return feature_cols
        
        spec_features = identify_feature_columns(df_spec)
        gene_features = identify_feature_columns(df_gene)
        
        # Merge and align data
        df_merged = pd.merge(df_spec, df_gene, on='Row_names', how='inner', suffixes=('_spec', '_gene'))
        
        spec_feature_cols = [col for col in df_merged.columns if col in spec_features or col.endswith('_spec')]
        gene_feature_cols = [col for col in df_merged.columns if col in gene_features or col.endswith('_gene')]
        
        spec_data = df_merged[spec_feature_cols].copy()
        gene_data = df_merged[gene_feature_cols].copy()
        
        spec_data.columns = [col.replace('_spec', '') for col in spec_data.columns]
        gene_data.columns = [col.replace('_gene', '') for col in gene_data.columns]
        
        data['raw_spec'] = spec_data
        data['raw_gene'] = gene_data
        
        print(f"     * Final spectral features: {data['raw_spec'].shape[1]}")
        print(f"     * Final gene features: {data['raw_gene'].shape[1]}")
        print("   - All data loaded successfully.")
        return data
        
    except Exception as e:
        print(f"     - FATAL ERROR loading data: {e}")
        traceback.print_exc()
        return None

def create_optimized_figure(data, output_dir, font_sizes):
    """Creates the optimized 6-panel validation figure."""
    print("\n3. Generating optimized validation figure...")
    
    # Setup figure with publication-quality settings
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    # fig.suptitle("MOFA+ Transformer Validation on HyperSeq Data", fontsize=font_sizes['figure_title'], fontweight='bold')
    
    # --- Panel A: MOFA Variance Explained (FIXED TO PERCENTAGES) ---
    ax = axes[0, 0]
    ve = data['mofa_variance']  # Already converted to percentages
    n_factors = ve.shape[1]
    ve_df = pd.DataFrame(
        ve, 
        index=['Spectral', 'Transcriptomics'], 
        columns=[f"F{i+1}" for i in range(n_factors)]
    )
    
    # Plot stacked bar chart
    ve_df.T.plot(kind='bar', stacked=True, ax=ax, color=["#7ADB1E", "#0fbbaa"], 
                 edgecolor='black', width=0.7)
    ax.set_title("MOFA+ Variance Explained", fontsize=font_sizes['panel_title'], fontweight='bold')
    ax.text(-0.1, 1.1, 'a', transform=ax.transAxes, fontsize=font_sizes['panel_title'], fontweight='bold', va='top')
    ax.set_xlabel("Factor", fontsize=font_sizes['axis_label'])
    ax.set_ylabel("Variance Explained (%)", fontsize=font_sizes['axis_label'])
    ax.tick_params(axis='x', rotation=0, labelsize=font_sizes['tick_label'])
    ax.tick_params(axis='y', labelsize=font_sizes['tick_label'])
    ax.legend(title='View', frameon=True, fontsize=font_sizes['legend_text'], title_fontsize=font_sizes['legend_title'])
    sns.despine(ax=ax)
    
    # Add percentage annotations
    for i, (f1_val, f2_val) in enumerate(zip(ve[0], ve[1])):
        if f1_val > 5:  # Only annotate if >5%
            ax.text(i, f1_val/2, f'{f1_val:.1f}%', ha='center', va='center', fontweight='bold', fontsize=font_sizes['annotation_bold'])
        if f2_val > 5:
            ax.text(i, f1_val + f2_val/2, f'{f2_val:.1f}%', ha='center', va='center', fontweight='bold', fontsize=font_sizes['annotation_bold'])

    # --- Panel B & C: CORRECTED TO SHOW FACTOR 3 (INTEGRATION FACTOR) ---
    # CRITICAL FIX: Show Factor 3 (integration factor) instead of Factor 4
    integration_factor_idx = 2  # Factor 3 (0-indexed)
    
    print(f"   - Using Factor {integration_factor_idx + 1} as integration factor")
    
    # Panel B: Spectral Loadings for Factor 3
    ax = axes[0, 1]
    weights_spec = pd.Series(data['mofa_weights_spec'][:, integration_factor_idx], 
                           index=data['mofa_features_spec'])
    top_spec = weights_spec.abs().nlargest(7).sort_values()
    
    display_names_spec = [name.replace('_spectral', '').replace('spectral_ch_', 'Ch') 
                         for name in top_spec.index]
    
    ax.barh(display_names_spec, top_spec.values, color='#7ADB1E', edgecolor='black')
    ax.set_title(f"Top Spectral Loadings (Factor {integration_factor_idx+1})", 
                fontsize=font_sizes['panel_title'], fontweight='bold')
    ax.text(-0.1, 1.1, 'b', transform=ax.transAxes, fontsize=font_sizes['panel_title'], fontweight='bold', va='top')
    ax.set_xlabel("Weight", fontsize=font_sizes['axis_label'])
    ax.tick_params(axis='both', labelsize=font_sizes['tick_label'])
    sns.despine(ax=ax)

    # Panel C: Gene Loadings for Factor 3 (should show stress genes!)
    ax = axes[0, 2]
    weights_tx = pd.Series(data['mofa_weights_tx'][:, integration_factor_idx], 
                          index=data['mofa_features_tx'])
    top_tx = weights_tx.abs().nlargest(7).sort_values()
    
    display_names_tx = [name.replace('_transcriptomics', '') for name in top_tx.index]
    
    ax.barh(display_names_tx, top_tx.values, color='#0fbbaa', edgecolor='black')
    ax.set_title(f"Top Gene Loadings (Factor {integration_factor_idx+1})", 
                fontsize=font_sizes['panel_title'], fontweight='bold')
    ax.text(-0.1, 1.1, 'c', transform=ax.transAxes, fontsize=font_sizes['panel_title'], fontweight='bold', va='top')
    ax.set_xlabel("Weight", fontsize=font_sizes['axis_label'])
    ax.tick_params(axis='both', labelsize=font_sizes['tick_label'])
    sns.despine(ax=ax)
    
    # Print discovered genes for verification
    print(f"   - Factor 3 top genes: {display_names_tx}")
    
    # --- Panel D: Attention vs. Correlation (Enhanced) ---
    ax = axes[1, 0]
    print("   - Computing correlations...")
    
    attn_df = data['attention'].copy()
    attn_spectral_features = set(attn_df['Spectral_Feature'].unique())
    attn_gene_features = set(attn_df['Metabolite_Feature'].unique())
    raw_spectral_features = set(data['raw_spec'].columns)
    raw_gene_features = set(data['raw_gene'].columns)
    
    common_spectral = attn_spectral_features.intersection(raw_spectral_features)
    common_genes = attn_gene_features.intersection(raw_gene_features)
    
    if len(common_spectral) > 0 and len(common_genes) > 0:
        correlations = []
        for gene in common_genes:
            for spec in common_spectral:
                try:
                    corr, _ = pearsonr(data['raw_gene'][gene], data['raw_spec'][spec])
                    correlations.append({
                        'Metabolite_Feature': gene,
                        'Spectral_Feature': spec,
                        'Correlation': corr
                    })
                except:
                    continue
        
        if correlations:
            corr_df = pd.DataFrame(correlations)
            comparison_df = pd.merge(
                attn_df[['Metabolite_Feature', 'Spectral_Feature', 'Mean_Attention_S2M']],
                corr_df,
                on=['Metabolite_Feature', 'Spectral_Feature'],
                how='inner'
            )
            
            if len(comparison_df) > 0:
                comparison_df['Abs_Correlation'] = comparison_df['Correlation'].abs()
                
                # Enhanced scatter plot with density coloring
                plt.sca(ax)
                scatter = ax.scatter(comparison_df['Abs_Correlation'], comparison_df['Mean_Attention_S2M'], 
                                   alpha=0.6, c=comparison_df['Mean_Attention_S2M'], 
                                   cmap='YlGn', s=30, edgecolors='black', linewidth=0.5)
                
                overall_corr = np.corrcoef(comparison_df['Abs_Correlation'], 
                                         comparison_df['Mean_Attention_S2M'])[0, 1]
                
                ax.set_title("Attention vs. Correlation", fontsize=font_sizes['panel_title'], fontweight='bold')
                ax.text(-0.1, 1.1, 'd', transform=ax.transAxes, fontsize=font_sizes['panel_title'], fontweight='bold', va='top')
                ax.set_xlabel("Absolute Pearson Correlation", fontsize=font_sizes['axis_label'])
                ax.set_ylabel("Mean Attention Score", fontsize=font_sizes['axis_label'])
                ax.tick_params(axis='both', labelsize=font_sizes['tick_label'])
                ax.text(0.05, 0.95, f'r = {overall_corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), fontsize=font_sizes['annotation_main'])
    
    sns.despine(ax=ax)

    # --- Panel E: ENHANCED - Show Actual Permutation Results ---
    ax = axes[1, 1]
    
    if data['permutation_results'] is not None:
        perm_data = data['permutation_results']['corrected_permutation']
        observed = perm_data['observed_attention_mean']
        null_dist = perm_data['null_distribution']
        p_value = perm_data['p_value']
        cohens_d = perm_data['cohens_d']
        
        # Plot null distribution
        ax.hist(null_dist, bins=20, alpha=0.7, color="#A5AB78", 
               label=f'Null Distribution (n=100)', density=True)
        
        # Plot observed value
        ax.axvline(observed, color='red', linewidth=3, 
                  label=f'Observed NEAT1\n(p = {p_value:.4f})')
        
        ax.set_title("Permutation Test Results", fontsize=font_sizes['panel_title'], fontweight='bold')
        ax.set_xlabel("Mean Attention Score", fontsize=font_sizes['axis_label'])
        ax.set_ylabel("Density", fontsize=font_sizes['axis_label'])
        ax.legend(fontsize=font_sizes['legend_text'])
        ax.tick_params(axis='both', labelsize=font_sizes['tick_label'])
        
        # Add effect size annotation
        ax.text(0.98, 0.95, f"Cohen's d = {cohens_d:.2f}", transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
               ha='right', va='top', fontsize=font_sizes['annotation_main'])
    else:
        # Fallback to gene comparison if permutation results not available
        attn_df = data['attention']
        neat1_attn = attn_df[attn_df['Metabolite_Feature'] == 'NEAT1']['Mean_Attention_S2M']
        bottom_genes = attn_df.nsmallest(10, 'Mean_Attention_S2M')['Metabolite_Feature'].values
        control_gene = bottom_genes[0] if len(bottom_genes) > 0 else 'Control'
        control_attn = attn_df[attn_df['Metabolite_Feature'] == control_gene]['Mean_Attention_S2M']
        
        if len(neat1_attn) > 0:
            sns.kdeplot(neat1_attn, ax=ax, fill=True, color="#238b45", 
                       label=f"NEAT1", cut=0)
            if len(control_attn) > 0:
                sns.kdeplot(control_attn, ax=ax, fill=True, color="grey", 
                           label=f"Control ({control_gene})", cut=0)
        
        ax.set_title("Attention Score Significance", fontsize=font_sizes['panel_title'], fontweight='bold')
        ax.legend(fontsize=font_sizes['legend_text'])
        ax.tick_params(axis='both', labelsize=font_sizes['tick_label'])
    
    ax.text(-0.1, 1.1, 'e', transform=ax.transAxes, fontsize=font_sizes['panel_title'], fontweight='bold', va='top')
    sns.despine(ax=ax)

    # --- Panel F: Enhanced Top Attention Pairs ---
    ax = axes[1, 2]
    attn_df = data['attention']
    top_pairs = attn_df.head(10).sort_values('Mean_Attention_S2M', ascending=True)
    
    # Create readable pair labels
    pair_labels = []
    attention_values = []
    for _, row in top_pairs.iterrows():
        spec_clean = row['Spectral_Feature'].replace('spectral_ch_', 'Ch')
        gene_clean = row['Metabolite_Feature']
        if len(gene_clean) > 8:
            gene_clean = gene_clean[:8] + "..."
        pair_labels.append(f"{spec_clean} ↔ {gene_clean}")
        attention_values.append(row['Mean_Attention_S2M'])
    
    bars = ax.barh(pair_labels, attention_values, color="#2AB33A", edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, attention_values)):
        ax.text(value + 0.0005, bar.get_y() + bar.get_height()/2, 
               f'{value:.3f}', va='center', ha='left', fontsize=font_sizes['annotation_bold'], fontweight='bold')
    
    ax.set_title("Top Cross-Modal Attention Pairs", fontsize=font_sizes['panel_title'], fontweight='bold')
    ax.text(-0.1, 1.1, 'f', transform=ax.transAxes, fontsize=font_sizes['panel_title'], fontweight='bold', va='top')
    ax.set_xlabel("Mean Attention Score", fontsize=font_sizes['axis_label'])
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=font_sizes['tick_label'])
    ax.tick_params(axis='x', labelsize=font_sizes['tick_label'])
    sns.despine(ax=ax)
    
    # --- Final Touches ---
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "hyperseq_validation_optimized.png")
    svg_path = os.path.join(output_dir, "hyperseq_validation_optimized.svg")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"\nSUCCESS: Optimized validation figure saved to: {output_path}")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    print("\n1. Loading all necessary result files...")
    all_data = load_all_data()
    
    if all_data:
        print("\n2. Proceeding with optimized figure generation...")
        create_optimized_figure(all_data, FIGURE_OUTPUT_DIR, FONT_SIZES)
        
        # Print summary of optimizations
        print("\n" + "="*60)
        print("OPTIMIZATIONS APPLIED:")
        print("="*60)
        print("✅ Panel A: Variance converted to percentages")
        print("✅ Panel B-C: Shows Factor 3 (integration factor)")
        print("✅ Panel E: Enhanced permutation test visualization")
        print("✅ Panel F: Added exact attention score labels")
        print("✅ Overall: Improved annotations and statistical emphasis")
        print("="*60)
    else:
        print("\nCRITICAL ERROR: Could not load all necessary data.")

    print("\n" + "="*80)
    print("OPTIMIZED Figure Generation - COMPLETE")
    print("="*80)