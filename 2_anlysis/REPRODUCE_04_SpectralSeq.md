# REPRODUCE_04_hyperseq.md

> **Purpose**: Validate framework generalizability on independent single-cell dataset (GEO: GSE254034).

---

## 📋 Overview

Applies identical pipeline to HyperSeq dataset (paired single-cell hyperspectral imaging + transcriptomics) to demonstrate framework generalizability across biological systems.

---

## 🔬 Validation Pipeline

### Step 1: MOFA+ Decomposition

```bash
python 1_mofa_decomposition.py
```

**Inputs**:
- `transformer_input_hyperseq_spectral.csv`
- `transformer_input_hyperseq_gene.csv`

**Parameters**: 15 factors, 2 views (spectral + transcriptomics)

**Outputs**:
- `mofa_model_hyperseq.hdf5`
- `transformer_input_spectral_hyperseq.csv` (15 features)
- `transformer_input_transcriptomics_hyperseq.csv` (100 genes)

**Key finding**: Factor 3 emerges as "Stress & Metabolism" integration factor (*HSPA6*, *COX6C* genes).

**What next**: Selected features to Transformer

---

### Step 2: Train Transformer + Permutation Test

```bash
python 2_train_transformer.py
```

**Inputs**: Transformer input files from Step 1

**Architecture**: 64-dim, 4 heads, 2 layers

**Integrated permutation test**:
- n=100 shuffles
- Validates spectral-NEAT1 attention convergence
- Statistical result: p_perm = 0.0002, Cohen's d = 6.63

**Outputs**:
- Model checkpoints
- `raw_attention_data_HyperSeq.h5`: 4D tensors
- `corrected_permutation_test_results_HyperSeq.json`

**What next**: Raw attention to processing

---

### Step 3: Process Attention

```bash
python 3_process_attention.py
```

**Inputs**: `raw_attention_data_HyperSeq.h5`, metadata

**Process**: Average heads, calculate overall/conditional attention, rank top pairs

**Outputs**:
- `processed_mean_attention_overall_HyperSeq.csv`
- `processed_mean_attention_conditional_HyperSeq.csv`

**What next**: Processed data for visualization

---

### Step 4: Prepare Visualization Data

```bash
python 4_prepare_visualization_data.py
```

**Inputs**: MOFA model, processed attention, permutation results

**Analysis**:
- Extract Factor 3 loadings
- Compute Pearson correlations vs attention (r = −0.023)
- Organize NEAT1 convergence data

**Outputs**: Data structures for Figure 8 (6 panels)

**What next**: Figure generation (see `REPRODUCE_05_visualization.md`)

---

## 🔧 Optional Diagnostic

```bash
python utils_inspect_outputs.py
```

**Purpose**: Inspect HDF5/Feather outputs for troubleshooting.

---

## 📦 Key Validation Results

- ✔ Framework successfully applied to single-cell data
- ✔ Factor 3 captures stress-related signatures
- ✔ Discovers spectral-NEAT1 link (p_perm = 0.0002, Cohen's d = 6.63)
- ✔ Attention diverges from correlation (r = −0.023), confirming non-linear discovery
- ✔ NEAT1 = stress-responsive lncRNA, biologically coherent finding

---

## ⚠️ Notes

- **Dataset**: GEO: GSE254034 (public)
- **Permutation test**: Integrated in training script
- **File paths**: Update in script configurations

---

**Next**: `REPRODUCE_05_visualization.md` for Figure 8 generation
