# REPRODUCE_03_transformer.md

> **Purpose**: Train cross-attention Transformer on MOFA-selected features, extract attention weights, and perform interpretability analysis.

---

## 📋 Overview

Two-phase workflow: **(1) Model training & attention extraction** → **(2) Analysis & interpretation**. Trains multi-task Transformer to predict genotype, treatment, and timepoint while capturing cross-modal feature interactions.

---

## 🔬 Phase 1: Model Training & Attention Extraction

### Step 1: Train Transformer with Feature-Level Attention

```bash
python train_transformer_attn.py
```

**Purpose**: Train multi-task cross-attention model and extract feature interaction patterns.

**Inputs**:
- `transformer_input_{view}.csv` (4 files from MOFA+ feature selection)
- Sample metadata

**Model architecture**:
- **Input**: Spectral features (view 1) + Molecular features (view 2)
- **Encoder**: Separate projection layers per view → hidden_dim
- **Cross-attention**: Bidirectional attention between views (spectral↔molecular)
- **Decoder**: Separate heads for 3 tasks (Genotype, Treatment, Timepoint)

**Key parameters**:
- `hidden_dim=64`
- `num_heads=4`
- `num_layers=2`
- `dropout=0.1`
- `epochs=150`
- `batch_size=16`

**Process**:
1. Split data (train/val/test: 60/20/20)
2. Standardize features per view
3. Train with early stopping
4. Extract attention weights from final cross-attention layer
5. Save raw 4D attention tensors (samples × heads × features)

**Outputs**:
- `best_model_{task}_{pairing}.pth`: Trained model checkpoints
- `raw_attention_data_{Tissue}.h5`: 4D attention tensors (HDF5)
- `raw_attention_metadata_{Tissue}.feather`: Sample metadata
- `training_log_{timestamp}.txt`: Loss curves, F1 scores

**What next**: Raw attention data to processing pipeline

---

### Step 2: Filter Test Samples

```bash
python filter_test_samples_for_interpretability.py
```

**Purpose**: Ensure test set contains only original (non-augmented) samples for interpretability analysis.

**Inputs**:
- `raw_attention_metadata_{Tissue}.feather`

**Process**:
1. Identify augmented samples by suffix patterns (_GP, _WARP, _SCALE, _NOISE, _ADD, _MULT, _MIX)
2. Filter to retain only original samples (ending with _L suffix)
3. Validate sample counts match expected biological replicates

**Outputs**:
- Filtered metadata for downstream analysis
- Sample count verification logs

**What next**: Process attention data with clean test samples

---

### Step 3: Process Raw Attention Data

```bash
python process_attention_data.py
```

**Purpose**: Aggregate 4D attention tensors into interpretable feature-pair statistics.

**Inputs**:
- `raw_attention_data_{Tissue}.h5`
- `raw_attention_metadata_{Tissue}.feather`

**Process**:
1. Load 4D tensors (samples × heads × spectral_features × molecular_features)
2. Average across attention heads
3. Calculate overall mean attention per feature pair
4. Compute conditional attention (by genotype, treatment, timepoint)
5. Calculate view-level statistics (mean, std, P95)
6. Rank top pairs by attention strength

**Outputs**:
- `processed_mean_attention_overall_{Tissue}.csv`: Overall attention per pair
- `processed_mean_attention_conditional_{Tissue}.csv`: Condition-specific attention
- `processed_top_500_pairs_overall_{Tissue}.csv`: Ranked top pairs
- `processed_view_level_attention_{Tissue}.csv`: Network-level statistics

**What next**: Processed attention for visualization (Figure 3-4) and temporal analysis

---

## 📊 Phase 2: Analysis & Interpretation

### Step 4: SHAP Feature Importance Analysis

```bash
python analyse_transformer_shap.py
```

**Purpose**: Identify features driving prediction accuracy using SHAP (SHapley Additive exPlanations).

**Inputs**:
- Trained model checkpoints
- `transformer_input_{view}.csv`
- Sample metadata

**Process**:
1. Load trained model for each task
2. Apply GradientExplainer with 100 background samples
3. Calculate SHAP values on test set
4. Aggregate absolute SHAP values per feature
5. Rank features by mean absolute SHAP

**Outputs** (per tissue × task):
- `shap_importance_{Tissue}_{Task}.csv`: Ranked feature importance
- SHAP summary plots (bar charts, clustermaps)
- Omics contribution stacked bars (spectral vs molecular)

**What next**: SHAP results for Figure 2 and MOFA/SHAP complementarity analysis (Figure 6)

---

### Step 5: MOFA+/SHAP Complementarity Analysis

```bash
python analyse_mofa_shap_overlap.py
```

**Purpose**: Compare variance-driven (MOFA+) vs prediction-driven (SHAP) feature discovery.

**Inputs**:
- MOFA+ feature loadings
- SHAP importance rankings

**Metrics**:
- Jaccard index (top feature overlap)
- Pearson correlation (loading weights vs SHAP values)

**Outputs**:
- `mofa_shap_overlap_analysis.csv`: Overlap metrics per task/tissue

**Key finding**: Limited overlap (Jaccard ≈ 0.035–0.055) with convergence localised to 546–560 nm spectral range, indicating complementary insights from variance-driving vs prediction-driving features.

**What next**: Results inform Figure 6 analysis

---

### Step 6: Generate Robustness Contract

```bash
python generate_robustness_contract.py
```

**Purpose**: Create robustness verification for MOFA+/SHAP overlap across sensitivity thresholds.

**Inputs**:
- MOFA+ feature loadings
- SHAP importance rankings

**Process**:
1. Sweep across top percentile thresholds (1%, 2.5%, 5%, 10%)
2. Calculate Jaccard overlap at each threshold
3. Identify spectral range of overlapping features
4. Verify in-band percentage (546–560 nm range)

**Outputs**:
- `robustness_contract.json`: Threshold-specific overlap metrics
  - Jaccard indices per threshold
  - Spectral ranges
  - In-band percentages

**What next**: Figure 6 reads this contract for consistent reporting

---

### Step 7: Attention-Based Network Analysis

**View-level statistics**:
```bash
python analyse_view_attn_stats.py
```
**Purpose**: Aggregate attention into network-level metrics (mean, std, P95).  
**Outputs**: `view_level_attention_summary_{Tissue}.csv`

**Feature-level analysis**:
```bash
python analyse_feature_attn_v2.py
```
**Purpose**: Analyze conditional attention patterns (genotype, treatment, timepoint) with plant-level statistics.  
**Outputs**: 
- `feature_attention_by_condition_{Tissue}.csv`
- Temporal trajectory data for Figure 4
- Plant-level coordination scores with bootstrap CIs

**Key results** (Leaf tissue):
- Day 1: 2.11× [95% CI: 1.09–4.91], Cohen's d = 1.01, FDR = 0.009
- Day 2: 4.18× [95% CI: 1.96–7.72], Cohen's d = 1.67, FDR = 0.002
- Day 3: 4.74× [95% CI: 3.38–6.70], Cohen's d = 2.52, FDR < 0.001
- Genotype × Time interaction: p = 0.007

**Key results** (Root tissue):
- No significant coordination differences (p = 0.705)

---

### Step 8: Summary Statistics

**MOFA results**:
```bash
python summarise_mofa.py
python count_mofa_features.py
```
**Purpose**: Extract factor counts, variance explained, feature selection statistics.

**Model performance**:
```bash
python aggregate_model_perf.py
```
**Purpose**: Aggregate F1 scores, accuracy across all tasks/tissues.  
**Outputs**: `model_performance_summary.csv` (used in Figure 5)

---

## 📦 Outputs Summary

**Model artifacts**:
- Trained model checkpoints (`.pth`)
- Raw attention tensors (`.h5`)
- Training logs

**Processed attention**:
- Overall attention scores per feature pair
- Conditional attention (by genotype/treatment/timepoint)
- Top-ranked pairs for visualization
- View-level network statistics

**Interpretability**:
- SHAP importance rankings (per task/tissue)
- MOFA+/SHAP overlap analysis
- Robustness contract (JSON)
- Feature contribution breakdowns

**Summary stats**:
- Model performance metrics
- MOFA+ factor summaries
- Feature selection statistics

---

## 🔄 Optional: KNN Baseline Comparison

```bash
python train_transformer_knn.py
```

**Purpose**: Train KNN baseline for performance comparison.  
**Note**: Demonstrates Transformer superiority via cross-attention mechanism.

---

## ⚠️ Notes

- **GPU recommended**: Training takes ~2-4 hours on CPU, ~30-60 min on GPU
- **Memory**: Raw attention tensors can be large (2-5 GB per tissue)
- **File paths**: Update in configuration sections of each script
- **SHAP computation**: Computationally intensive; uses 100 background samples by default
- **Attention processing**: Runs separately for leaf/root tissues
- **Test samples**: Always use original (non-augmented) samples for interpretability to avoid pseudo-replication

---

**Next**: `REPRODUCE_04_hyperseq.md` for external validation, then `REPRODUCE_05_visualization.md` for figure generation
