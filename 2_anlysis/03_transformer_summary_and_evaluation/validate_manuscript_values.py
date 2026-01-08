# -*- coding: utf-8 -*-
"""
MANUSCRIPT VALIDATION REGISTRY
==============================
Single source of truth mapping manuscript claims to their authoritative data files.

Run this script to validate ALL manuscript statistics against source files.
Any discrepancy = immediate alert.

Usage:
    python validate_manuscript_values.py
"""

import pandas as pd
import json
import os
import sys

# =============================================================================
# CONFIGURATION - UPDATE PATHS IF NEEDED
# =============================================================================
BASE_DIR = r"C:\Users\ms\Desktop\hyper"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# =============================================================================
# AUTHORITATIVE SOURCE REGISTRY
# =============================================================================
REGISTRY = {
    # -------------------------------------------------------------------------
    # TABLE S2: Model Performance Metrics
    # Source: v3_feature_attention/{tissue}/results/transformer_class_performance_{Tissue}.csv
    # -------------------------------------------------------------------------
    "Table_S2_Leaf_Genotype_F1": {
        "manuscript_value": 0.9505,
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "v3_feature_attention", "leaf", "results", "transformer_class_performance_Leaf.csv"),
        "extraction": lambda df: df[(df['Task']=='Genotype') & (df['Metric']=='F1_Macro')]['Score'].values[0],
        "tolerance": 0.001
    },
    "Table_S2_Leaf_Treatment_F1": {
        "manuscript_value": 0.9802,
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "v3_feature_attention", "leaf", "results", "transformer_class_performance_Leaf.csv"),
        "extraction": lambda df: df[(df['Task']=='Treatment') & (df['Metric']=='F1_Macro')]['Score'].values[0],
        "tolerance": 0.001
    },
    "Table_S2_Leaf_TimePoint_F1": {
        "manuscript_value": 0.7559,
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "v3_feature_attention", "leaf", "results", "transformer_class_performance_Leaf.csv"),
        "extraction": lambda df: df[(df['Task']=='Day') & (df['Metric']=='F1_Macro')]['Score'].values[0],
        "tolerance": 0.001
    },
    "Table_S2_Root_Genotype_F1": {
        "manuscript_value": 0.8096,
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "v3_feature_attention", "root", "results", "transformer_class_performance_Root.csv"),
        "extraction": lambda df: df[(df['Task']=='Genotype') & (df['Metric']=='F1_Macro')]['Score'].values[0],
        "tolerance": 0.001
    },
    "Table_S2_Root_Treatment_F1": {
        "manuscript_value": 1.0,
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "v3_feature_attention", "root", "results", "transformer_class_performance_Root.csv"),
        "extraction": lambda df: df[(df['Task']=='Treatment') & (df['Metric']=='F1_Macro')]['Score'].values[0],
        "tolerance": 0.001
    },
    "Table_S2_Root_TimePoint_F1": {
        "manuscript_value": 0.8671,  # CORRECTED from 0.8373
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "v3_feature_attention", "root", "results", "transformer_class_performance_Root.csv"),
        "extraction": lambda df: df[(df['Task']=='Day') & (df['Metric']=='F1_Macro')]['Score'].values[0],
        "tolerance": 0.001
    },
    
    # -------------------------------------------------------------------------
    # MOFA-SHAP OVERLAP (Figure 6 / Table S4)
    # Source: transformer/shap_analysis_ggl/mofa_shap_overlap_summary.json
    # -------------------------------------------------------------------------
    "Figure6_Leaf_Genotype_Jaccard": {
        "manuscript_value": 0.1765,
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "shap_analysis_ggl", "mofa_shap_overlap_summary.json"),
        "extraction": lambda data: data.get('Leaf', {}).get('Genotype', {}).get('jaccard', data.get('results', {}).get('Leaf', {}).get('Genotype', {}).get('jaccard_top50')),
        "tolerance": 0.001,
        "is_json": True
    },
    
    # -------------------------------------------------------------------------
    # HYPERSEQ PERMUTATION TEST (Figure 8)
    # Source: mofa_trasformer_val/val/transformer_results/results/corrected_permutation_test_results_HyperSeq.json
    # -------------------------------------------------------------------------
    "Figure8_HyperSeq_pvalue": {
        "manuscript_value": 0.0002,
        "source_file": os.path.join(OUTPUT_DIR, "mofa_trasformer_val", "val", "transformer_results", "results", "corrected_permutation_test_results_HyperSeq.json"),
        "extraction": lambda data: data.get('permutation_test', {}).get('p_value', data.get('p_value')),
        "tolerance": 0.0001,
        "is_json": True
    },
    "Figure8_HyperSeq_n_permutations": {
        "manuscript_value": 5000,
        "source_file": os.path.join(OUTPUT_DIR, "mofa_trasformer_val", "val", "transformer_results", "results", "corrected_permutation_test_results_HyperSeq.json"),
        "extraction": lambda data: data.get('permutation_test', {}).get('total_permutations', data.get('total_permutations')),
        "tolerance": 0,
        "is_json": True
    },
    
    # -------------------------------------------------------------------------
    # SHAP ANALYSIS METADATA
    # Source: transformer/shap_analysis_ggl/shap_run_manifest_Leaf.json
    # -------------------------------------------------------------------------
    "SHAP_n_original_samples": {
        "manuscript_value": 168,
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "shap_analysis_ggl", "shap_run_manifest_Leaf.json"),
        "extraction": lambda data: data.get('n_original', data.get('sample_counts', {}).get('n_original')),
        "tolerance": 0,
        "is_json": True
    },
    "SHAP_n_background": {
        "manuscript_value": 100,
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "shap_analysis_ggl", "shap_run_manifest_Leaf.json"),
        "extraction": lambda data: data.get('n_background', data.get('sample_counts', {}).get('n_background')),
        "tolerance": 0,
        "is_json": True
    },
    
    # -------------------------------------------------------------------------
    # COORDINATION ANALYSIS (Figure 4)
    # Source: transformer/novility_plot/final/fig4_summary_stats.json (if exists)
    # -------------------------------------------------------------------------
    "Figure4_Leaf_TP3_FoldChange": {
        "manuscript_value": 4.74,
        "source_file": os.path.join(OUTPUT_DIR, "transformer", "novility_plot", "final", "fig4_plant_data.csv"),
        "extraction": "MANUAL_CHECK",  # Complex calculation
        "tolerance": 0.1
    },
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def validate_all():
    """Run validation on all registered manuscript values."""
    print("=" * 70)
    print("MANUSCRIPT VALIDATION REPORT")
    print("=" * 70)
    
    results = []
    errors = []
    warnings = []
    
    for key, config in REGISTRY.items():
        source_file = config['source_file']
        manuscript_val = config['manuscript_value']
        tolerance = config.get('tolerance', 0.001)
        is_json = config.get('is_json', False)
        extraction = config['extraction']
        
        # Check if file exists
        if not os.path.exists(source_file):
            errors.append(f"[MISSING FILE] {key}: {source_file}")
            continue
        
        # Skip manual checks
        if extraction == "MANUAL_CHECK":
            warnings.append(f"[MANUAL CHECK REQUIRED] {key}")
            continue
        
        # Load data
        try:
            if is_json:
                with open(source_file, 'r') as f:
                    data = json.load(f)
                actual_val = extraction(data)
            else:
                df = pd.read_csv(source_file)
                actual_val = extraction(df)
        except Exception as e:
            errors.append(f"[EXTRACTION ERROR] {key}: {e}")
            continue
        
        # Compare
        if actual_val is None:
            errors.append(f"[NULL VALUE] {key}: Could not extract value")
            continue
            
        diff = abs(actual_val - manuscript_val)
        status = "PASS" if diff <= tolerance else "FAIL"
        
        results.append({
            'key': key,
            'manuscript': manuscript_val,
            'actual': round(actual_val, 6) if isinstance(actual_val, float) else actual_val,
            'diff': round(diff, 6),
            'status': status
        })
        
        if status == "FAIL":
            errors.append(f"[MISMATCH] {key}: manuscript={manuscript_val}, actual={actual_val}")
    
    # Print results
    print("\n--- VALIDATION RESULTS ---\n")
    for r in results:
        icon = "OK" if r['status'] == "PASS" else "XX"
        print(f"[{icon}] {r['key']}")
        print(f"     Manuscript: {r['manuscript']}, Actual: {r['actual']}, Diff: {r['diff']}")
    
    # Print warnings
    if warnings:
        print("\n--- WARNINGS ---")
        for w in warnings:
            print(f"  {w}")
    
    # Print errors
    if errors:
        print("\n--- ERRORS ---")
        for e in errors:
            print(f"  {e}")
        print(f"\n!!! {len(errors)} ERROR(S) FOUND !!!")
        return False
    else:
        print("\n" + "=" * 70)
        print("ALL VALIDATIONS PASSED")
        print("=" * 70)
        return True


def generate_source_map():
    """Generate a human-readable source map for documentation."""
    print("\n" + "=" * 70)
    print("AUTHORITATIVE SOURCE MAP")
    print("=" * 70)
    
    for key, config in REGISTRY.items():
        print(f"\n{key}:")
        print(f"  Manuscript value: {config['manuscript_value']}")
        print(f"  Source: {config['source_file']}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    success = validate_all()
    print("\n")
    generate_source_map()
    
    if not success:
        sys.exit(1)