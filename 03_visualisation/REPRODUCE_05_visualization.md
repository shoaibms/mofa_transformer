# REPRODUCE_05_visualization.md

> **Purpose**: Generate all manuscript figures from processed analysis outputs.

---

## 📋 Overview

Transforms MOFA+, Transformer, and SHAP analysis results into publication-ready figures. Each script is standalone and generates specific manuscript panels.

**Prerequisites**: Outputs from `02_analysis/` (MOFA models, attention weights, SHAP values)

---

## 📊 Main Figures

### **Figure 1: MOFA+ Variance Decomposition**
**Purpose**: Visualize latent factor structure and metadata associations.

```bash
python Figure_1.py mofa_model.hdf5 metadata.csv correlations.csv output_dir/
```

**Inputs**:
- `mofa_model.hdf5`: Trained MOFA+ model with factor weights
- `metadata.csv`: Sample annotations (genotype, treatment, timepoint)
- `correlations.csv`: Factor-metadata Spearman correlations with FDR

**Outputs**: 6-panel figure (variance explained, correlation heatmap, factor distributions)

**What next**: Factor loadings used in Figure 3 networks and Figure 6 complementarity analysis.

---

### **Figure 2: SHAP Predictive Feature Importance**
**Purpose**: Identify features driving prediction accuracy for each task.

```bash
python Figure_2.py
```

**Inputs**: `shap_importance_[Tissue]_[Task].csv` from `analyse_transformer_shap.py`

**Outputs**: Ranked feature importance plots (top 15-30 features per task)

**What next**: SHAP features compared with MOFA+ loadings in Figure 6.

---

### **Figure 3: Cross-Modal Attention Networks**
**Purpose**: Reveal directed spectral→metabolite coordination patterns.

```bash
python Figure_3.py
```

**Inputs**:
- `processed_mean_attention_conditional_[Tissue].csv`: Condition-specific attention
- `processed_top_500_pairs_overall_[Tissue].csv`: Overall top attention pairs
- `processed_view_level_attention_[Tissue].csv`: Network-level statistics

**Outputs**: 4 network panels (leaf/root × genotype) + statistical comparison bar chart

**What next**: Attention patterns analyzed temporally in Figure 4.

---

### **Figure 4: Temporal Attention Dynamics**
**Purpose**: Track coordination advantage (G1 vs G2) across stress phases.

**Panels A-B** (Coordination Landscapes):
```bash
python Figure_4_a-b.py
```
- **Inputs**: `processed_attention_trends_top_500_[Tissue].csv`
- **Outputs**: Heatmaps showing G1-G2 attention differences over time

**Panel C** (Network Metrics):
```bash
python Figure_4_c.py
```
- **Inputs**: Conditional attention by genotype/timepoint
- **Outputs**: Network coordination strength trajectories

**What next**: Temporal patterns inform biomarker selection in Figure 5.

---

### **Figure 5: Biomarker Identification & Validation**
**Purpose**: Demonstrate early stress detection using identified biomarkers.

```bash
python Figure_5.py
```

**Inputs**:
- Model performance metrics (F1 scores, confusion matrices)
- Ranked biomarker features with validation scores

**Outputs**: Performance comparison + biomarker ranking plots

**What next**: Final integration analysis in Figure 6.

---

### **Figure 6: MOFA+/SHAP Complementarity & Temporal Evolution**
**Purpose**: Synthesize variance-driven and prediction-driven feature discovery.

```bash
python Figure_6.py
```

**Inputs**:
- Spectral data with temporal statistics
- MOFA+ feature weights
- SHAP importance values
- Feature overlap analysis results

**Outputs**: 10-panel comprehensive figure (temporal trends, correlation, overlap quantification)

**What next**: Task-specific dissection in Figure 7.

---

### **Figure 7: Tissue-Specific Predictive Architecture**
**Purpose**: Reveal task-dependent feature utilization patterns.

**Panels A-B** (Feature Clustering):
```bash
python Figure_7_a-b.py
```
- **Outputs**: Hierarchical clustering of top 50 SHAP features across tasks

**Panels C-G** (Modality Contributions):
```bash
python Figure_7_c-g.py
```
- **Outputs**: Spectral vs metabolite contribution breakdown per tissue/task

**What next**: Framework validation on external dataset (Figure 8).

---

### **Figure 8: HyperSeq Single-Cell Validation**
**Purpose**: Demonstrate generalizability on independent dataset (GEO: GSE254034).

```bash
python Figure_8.py
```

**Inputs**:
- `mofa_model_hyperseq.hdf5`: HyperSeq MOFA+ decomposition
- `processed_mean_attention_overall_HyperSeq.csv`: Cross-attention weights
- `corrected_permutation_test_results_HyperSeq.json`: NEAT1 statistical validation

**Outputs**: 6-panel validation (variance, Factor 3 loadings, attention vs correlation, NEAT1 discovery)

**Key Result**: Non-linear spectral-NEAT1 link (p=0.0099) invisible to correlation analysis.

---

## 📑 Supplementary Figures

### **Fig S1, S9**: Flowcharts (Mermaid)
- **Files**: `Fig_S1.mmd` (LCMS), `Fig_S9.mmd` (Augmentation)
- **Purpose**: Document preprocessing and augmentation workflows

### **Fig S2, S3, S5**: Extended Attention Analysis
```bash
python Fig_S2_3_5.py
```
- **Purpose**: Genotype-specific dynamics, stereotyped patterns, biomarker scatter plots

### **Fig S4**: Transformer Performance Metrics
```bash
python Fig_S4.py
```
- **Purpose**: Validate predictive accuracy across all tasks

### **Fig S6-8**: Data Quality Control
```bash
python Fig_S6-7.py  # LC-MS QC
python Fig_S8.py    # Hyperspectral QC
```
- **Purpose**: Establish input data integrity before analysis

### **Fig S10-13**: Augmentation Validation
```bash
python Fig_S10-13.py
```
- **Purpose**: Demonstrate augmented data preserves biological signal and structure

### **Fig S14**: MOFA+ Integration Network
```bash
python Fig_S14.py
```
- **Purpose**: Visualize cross-view feature relationships from factor loadings

---

## 🔧 Usage Notes

**File Paths**: Edit input/output paths in script headers (typically lines 15-40)

**Execution**: All scripts are standalone Python files—run directly:
```bash
python [script_name].py
```

**Outputs**: Each generates PNG (300 DPI) + SVG in specified output directory

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| Missing input files | Verify upstream analysis completed (see `REPRODUCE_02-04.md`) |
| Path errors | Update hardcoded paths in script configuration section |
| Memory errors (Fig 6, S10-13) | Expected for data-intensive plots—ensure 8GB+ RAM available |

---

**Dependency Chain**: `REPRODUCE_02_mofa.md` → `REPRODUCE_03_transformer.md` → `REPRODUCE_04_hyperseq.md` → **This document**