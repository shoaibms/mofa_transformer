# REPRODUCE_01_preprocessing.md

> **Purpose**: Prepare raw hyperspectral and LC-MS data for multi-omic analysis through QC, imputation, and augmentation.

---

## 📋 Overview

Three-stage pipeline: **(1) LC-MS preprocessing** → **(2) Spectral QC** → **(3) Data augmentation**. Expands 336 raw samples to 2,688 samples while preserving biological signal.

---

## 🔬 Stage 1: LC-MS Preprocessing

### 1.1 Missing Data Assessment
```bash
python mcar_test.py          # Little's MCAR test
python mar_test.py           # Logistic regression for MAR
python missing_vis.py        # Visualize patterns (Fig S7 A-C)
```
**Inputs**: Raw LC-MS data  
**Outputs**: Statistical tests, missingness mechanism determination  
**What next**: If MAR confirmed → imputation

---

### 1.2 Imputation
```bash
python ml_impute.py          # KNN, Bayesian PCA, SVD, GPR, EM
python median_impute.py      # Median imputation
Rscript rf_impute.r          # Random Forest (best performer)

# Validation
python impute_dist_check.py  # EMD, Hellinger distance
python impute_validate.py    # Q-Q, ECDF, KDE plots (Fig S7 D-F)
python diversity_metrics.py  # Shannon, Simpson's diversity
```
**Inputs**: Raw LC-MS with missing values  
**Outputs**: Imputed datasets, quality metrics  
**What next**: Imputed data to outlier detection

---

### 1.3 Outlier Detection & Removal
```bash
python feature_filter.py         # Filter columns (<3 replicates)
python isolation_forest.py       # Isolation Forest
python dim_reduce_outliers.py    # PCA/t-SNE detection (Fig S7 G-H)
python outlier_vis.py            # Validation plots
```
**Inputs**: Imputed LC-MS  
**Outputs**: `core_outlier_indices.csv`, cleaned data  
**What next**: Cleaned data to transformation

---

### 1.4 Transformation
```bash
python transform_data.py         # Log, sqrt, Box-Cox, Yeo-Johnson, asinh, glog, Anscombe
python transform_eva.py          # Evaluate normality
python transform_metrics.py      # Quantitative metrics
python normality_test.py         # Shapiro-Wilk, Anderson-Darling
python normality_vis.py          # Diagnostic plots
```
**Inputs**: Cleaned LC-MS  
**Outputs**: Transformed datasets, normality results  
**What next**: Transformed data to variable selection

---

### 1.5 Variable Selection
```bash
python variance_calc.py          # rMAD-based filtering
```
**Inputs**: Transformed LC-MS  
**Outputs**: High-variance feature subset  
**What next**: Curated features ready for augmentation

---

## 📡 Stage 2: Hyperspectral QC

```bash
python spectral_qc.py
```
**Inputs**: Raw hyperspectral CSV (350-2500 nm)  
**Quality checks**: Outliers (IQR, Modified Z-score, LOF), SNR, derivatives  
**Outputs**: `core_outlier_indices.csv`, QC metrics (Fig S8)  
**What next**: Curated spectral data to augmentation

---

## 🔁 Stage 3: Data Augmentation

### 3.1 Spectral Augmentation
```bash
python aug_spectral_data.py
```
**Inputs**: Curated spectral (336 samples)  
**Methods**: GP, Mixup, Warp, Scale, Noise, Add, Mult (7 methods)  
**Outputs**: `augmented_spectral_data.csv` (2,688 samples, IDs tagged by method)

---

### 3.2 Molecular Augmentation
```bash
python aug_mol_features.py
```
**Inputs**: Curated molecular (336 samples)  
**Methods**: Scale (5×), Mix (2×)  
**Outputs**: `augmented_molecular_feature_data.csv` (2,688 samples)

---

### 3.3 Validation
```bash
# Quality control
python qc_aug_spectral.py        # Spectral QC
python sr3_4.py                  # Molecular QC with ML validation

# Cross-modality checks
python verify_augmentation.py    # Sample alignment
python sr5.py                    # Cross-view validation
python sr6_7.py                  # Statistical divergence

# Batch effects
python validate_mol_aug_batch.py # Batch preservation

# Reports
python sr1.py                    # HTML: SR1.html
python sr2.py                    # HTML: SR2.html
python plot_spectral_aug_qc.py   # Plots
```
**Outputs**: 7 HTML reports, quality scores (Fig S10-13)  
**Validation**: Distribution similarity, signal preservation, ML cross-validation, batch consistency

---

### 3.4 Integrated Pipeline (Optional)
```bash
python run_augmentation.py       # Orchestrates all augmentation steps
```

---

## 📊 Final Outputs

- `augmented_spectral_data.csv`: 2,688 × 2,151 features
- `augmented_molecular_feature_data.csv`: 2,688 × molecular features
- Validation reports: Fig S6-13

---

## 🔧 Utilities

```bash
python colour_utils.py           # Plot color standards
python metadata_tools.py         # Sample annotation tools
```

---

## ⚠️ Notes

- **Execution order**: LC-MS → Spectral QC → Augmentation
- **File paths**: Update paths in script headers
- **Dependencies**: R `missForest` package required
- **Memory**: 8GB+ RAM for augmentation

---

**Next**: `REPRODUCE_02_mofa.md` for multi-omic factor analysis