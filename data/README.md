# Dataset Description

This directory contains preprocessed input data files for the MOFA+ Transformer multi-omics integration analysis of wheat osmotic stress responses.

---

## 📊 Files

### 1. **hyper_full_w.csv** — Hyperspectral Reflectance Data
- **Samples**: 336 (168 leaf + 168 root)
- **Features**: 2,151 spectral bands (351–2,500 nm)
- **Instrument**: ASD FieldSpec 4 spectroradiometer
- **Format**: CSV (sample IDs in rows, wavelengths W_351 to W_2500 in columns)
- **Quality Control**: High signal quality (median STD=0.080, SNR≈2.39); 28 potential outliers identified but retained to preserve biological variation (Fig. S8)

### 2. **n_p_l2.csv** — Leaf Molecular Features
- **Samples**: 336
- **Features**: 1,418 molecular features (807 negative mode + 611 positive mode)
- **Format**: CSV (sample IDs in rows, features N_Cluster_* and P_Cluster_* in columns)
- **Preprocessing**: Random Forest imputation (MAR pattern), Isolation Forest outlier removal, asinh transformation, rMAD filtering (<30%) (Fig. S6-S7)

### 3. **n_p_r2.csv** — Root Molecular Features
- **Samples**: 336
- **Features**: 1,721 molecular features (982 negative mode + 739 positive mode)
- **Format**: CSV (sample IDs in rows, features N_Cluster_* and P_Cluster_* in columns)
- **Preprocessing**: Same pipeline as leaf data

**Note**: Tissue-specific counts include features shared between tissues. The combined dataset contains 2,471 unique molecular features (1,398 negative mode with 391 shared, 1,073 positive mode with 277 shared).

---

## 🧬 Experimental Design

| Factor | Levels | Description |
|--------|--------|-------------|
| **Genotype** | G1, G2 | Drought-tolerant (Gladius) vs. susceptible (DAS5_003811) |
| **Treatment** | T0, T1 | Control vs. osmotic stress (0.15–0.3 M sorbitol) |
| **Batch** | B1, B2 | Acute (0.3 M) vs. mild prolonged (0.15 M) stress protocols |
| **Time Point** | Day 1, 2, 3 | Early onset, intermediate acclimation, late adaptation |
| **Tissue** | Leaf, Root | Above-ground shoot vs. below-ground root system |
| **Replicates** | 7 per condition | Biological replicates |

**Total design**: 2 genotypes × 2 treatments × 3 time points × 2 tissues × 7 replicates × 2 batches = 336 samples

---

## 📋 Metadata Columns

All files share the following metadata structure (first 8 columns):

| Column | Description |
|--------|-------------|
| `Vac_id` | Internal sample identifier |
| `Genotype` | G1 (tolerant) or G2 (susceptible) |
| `Entry` | Genotype name (Gladius or DAS5_003811) |
| `Tissue.type` | Leaf or Root |
| `Batch` | B1 (acute stress) or B2 (mild prolonged stress) |
| `Treatment` | T0 (control) or T1 (osmotic stress) |
| `Replication` | Biological replicate number (1–7) |
| `Day` | Sampling time point (1, 2, or 3) |

---

## 🔬 Data Acquisition Methods

### Hyperspectral Reflectance
- **Instrument**: ASD FieldSpec 4 spectroradiometer
- **Wavelength range**: 351–2,500 nm
- **Environment**: Custom matte-black imaging cube with standardized illumination (2× 500W halogen lamps)
- **Calibration**: White Spectralon® reference panel
- **Acquisition**: 25-scan average per sample
- **Quality metrics**: Median STD=0.080; stable spectral shapes confirmed by Savitzky-Golay derivative analysis (window=5, polyorder=2)

### LC-MS Metabolomics
- **LC System**: Vanquish UHPLC (Thermo Scientific)
- **Column**: C18 reversed-phase (2.1 × 100 mm, 1.7 µm) at 30°C
- **Mobile phases**: (A) Water + 0.1% formic acid; (B) Acetonitrile + 0.1% formic acid
- **Gradient**: 2–100% B over 11 min, held 4 min, 5 min re-equilibration
- **MS**: Q Exactive Plus Orbitrap (70,000 FWHM resolution at m/z 200)
- **Ionization**: ESI positive (3,600V) and negative (3,300V) modes (separate runs)
- **Mass range**: m/z 100–1,500
- **Data processing**: Genedata Expressionist® Refiner MS 18.0.1 (see Table S7 for parameters)
- **QC**: Internal standards and pooled QC samples throughout analytical sequence

---

## 🗄️ Raw Data Repository

**Raw LC-MS data (.raw format) have been deposited to MetaboLights.**

📍 **Accession**: Pending (permanent accession will be provided upon manuscript acceptance)  
🔗 **Repository**: https://www.ebi.ac.uk/metabolights/

The MetaboLights repository contains:
- Raw LC-MS files (672 samples: 336 leaf + 336 root, positive and negative modes)
- Complete sample metadata and experimental protocols
- LC-MS instrument method files
- Peak picking and data processing parameters

---

## 📈 Data Processing Pipeline

These files represent **quality-controlled, preprocessed data** ready for analysis:
```
Raw Data → Preprocessing (QC, imputation, outlier removal, transformation)
   ↓
These Files (n=336) → Data Augmentation (8× expansion, n=2,688)
   ↓
MOFA+ Decomposition (11 latent factors, 519 selected features)
   ↓
Transformer Analysis (interpretable cross-modal attention)
```

For complete preprocessing workflows, see:
- **Scripts**: `01_data_preprocessing/` directory
- **Documentation**: `REPRODUCE_01_preprocessing.md`
- **QC Reports**: Supplementary Figures S6-S8 and SR1-SR7

---

## 💾 File Format Specifications

- **Encoding**: UTF-8
- **Delimiter**: Comma (`,`)
- **Decimal separator**: Period (`.`)
- **Missing values**: `NA` or empty cells
- **Header row**: Column names in Row 1
- **Data structure**: Metadata columns 1–8, feature columns 9+

---

## 📖 Citation

If you use these datasets, please cite:

> Mirza, S.M., Reddy, P., Fitzgerald, G.J., Hayden, M.J., & Kant, S. (2025). MOFA+ Transformer: An Interpretable Deep Learning Framework for Dynamic, Feature-Specific Multi-Omics Integration. *[Journal]*, *[Volume]*, *[Pages]*.

---

## 📧 Contact

**Lead Author**: Shoaib M. Mirza  
**Email**: shoaibmirza2200@gmail.com  
**GitHub**: https://github.com/shoaibms/mofa_transformer

For data access issues or questions, please open an issue on GitHub.

---

## 📄 License

This dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.

🔗 https://creativecommons.org/licenses/by/4.0/
