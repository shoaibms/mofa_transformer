# REPRODUCE_02_mofa.md

> **Purpose**: Decompose multi-omic variance into interpretable latent factors and select features for downstream analysis.

---

## 📋 Overview

MOFA+ identifies shared variation patterns across 4 data views (leaf/root × spectral/molecular). Core workflow: **Data alignment** → **Factor decomposition** → **Feature selection** → **Validation**.

---

## 🔬 Core Analysis

### Step 1: MOFA+ Factor Decomposition

```bash
python run_mofa_analysis.py
```

**Purpose**: Identify latent factors capturing biological variation (genotype, treatment, timepoint).

**Inputs**:
- Augmented spectral data (leaf/root)
- Augmented molecular data (leaf/root)
- Sample metadata (genotype, treatment, timepoint, batch)
- Mapping file for sample alignment

**Key parameters**:
- `num_factors=20` (initial)
- `drop_factor_threshold=0.01` (prune inactive factors)
- `maxiter=500`
- `convergence_mode="medium"`

**Process**:
1. Align samples across 4 views using mapping file
2. Standardize features per view
3. Train MOFA+ to decompose variance
4. Extract active factors (typically ~11-15 retained)

**Outputs**:
- `mofa_model.hdf5`: Trained model with factor weights
- `variance_explained_per_factor.csv`: R² per view/factor
- `aligned_combined_metadata.csv`: Sample metadata aligned to factors
- Factor values for all samples

**What next**: Factor-metadata associations, feature selection

---

### Step 2: Feature Selection

```bash
python select_mofa_features.py
```

**Purpose**: Select top features (50-200 per view) based on factor loadings for Transformer input.

**Inputs**:
- `mofa_model.hdf5`
- Metadata with biological annotations

**Selection strategy**:
1. Calculate factor-metadata correlations (Spearman + FDR)
2. Identify biologically relevant factors
3. Compute feature importance across relevant factors
4. Apply hybrid stratified selection:
   - Prioritize high-loading features
   - Balance across spectral bands/molecular clusters
   - Target N features per view (configurable)

**Outputs**:
- `mofa_overall_feature_importance_{view}.csv`: Ranked features
- `transformer_input_{view}.csv`: Selected features + metadata for Transformer
- `mofa_feature_selection_summary.json`: Selection statistics

**What next**: Transformer training (see `REPRODUCE_03_transformer.md`)

---

## 📊 Validation & Interpretation

### Factor-Metadata Associations

**Automated in `run_mofa_analysis.py`:**

Calculates Spearman correlations between factors and experimental variables:
- Genotype (G1 vs G2)
- Treatment (T0 vs T1)
- Timepoint (TP1, TP2, TP3)
- Batch effects

**Outputs**:
- `mofa_factor_metadata_correlations.csv`: Correlation matrix + FDR-adjusted p-values
- Identifies factors driving variance in each experimental axis

---

### Statistical Tests

**Genotype differences**:
```bash
# Automated in run_mofa_analysis.py
# Mann-Whitney U tests for factor values: G1 vs G2
```
**Output**: `mofa_genotype_factor_differences.csv`

**Cross-tissue coordination**:
```bash
# Automated in run_mofa_analysis.py
# Identifies factors capturing shared leaf/root signals
```
**Output**: `mofa_tissue_coordination_factors.csv`

---

### Stability Analysis (Optional)

**Bootstrap resampling**:
```bash
python mofa_bootstrap.py
```
**Purpose**: Assess feature selection stability across 100 bootstrap iterations.  
**Warning**: Computationally intensive (re-trains MOFA+ 100×).

**Outputs**:
- `bootstrap_feature_stability.csv`: Selection frequency per feature
- Confidence intervals for factor loadings

---

**Permutation testing**:
```bash
python mofa_permutation_test.py
```
**Purpose**: Test significance of factor-metadata associations via permutation (n=1000).

**Outputs**:
- `permutation_test_results.csv`: Empirical p-values for factor associations

---

## 🎨 Visualization

```bash
python viz_mofa_results.py
```

**Generates**:
- Variance explained barplots (Fig 1A)
- Factor-metadata correlation heatmap (Fig 1B)
- Factor scatter plots colored by condition (Fig 1C-E)
- Factor distribution boxplots (Fig 1F)

**Outputs**: Multi-panel figure saved as PNG/SVG

---

## 📦 Outputs Summary

**Model files**:
- `mofa_model.hdf5`: Full model (factors, weights, variance)
- `variance_explained_per_factor.csv`
- `mofa_factor_values.csv`

**Feature selection**:
- `transformer_input_{view}.csv` (4 files: leaf/root × spectral/molecular)
- `mofa_overall_feature_importance_{view}.csv`

**Validation**:
- `mofa_factor_metadata_correlations.csv`
- `mofa_genotype_factor_differences.csv`
- `mofa_tissue_coordination_factors.csv`

**Optional**:
- `bootstrap_feature_stability.csv`
- `permutation_test_results.csv`

---

## ⚠️ Notes

- **File paths**: Update in script configuration sections
- **Factor count**: Auto-pruned based on `drop_factor_threshold` (typically 11-15 retained from 20 initial)
- **Feature selection**: Target count is configurable (`transformer_feature_cap` parameter)
- **Mapping file**: Critical for aligning samples across views—verify Row_names match

---

**Next**: `REPRODUCE_03_transformer.md` for cross-attention modeling on selected features