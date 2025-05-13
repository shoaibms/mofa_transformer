"""
This script provides a comprehensive suite of tools for metadata analysis,
dataset compatibility assessment, and MOFA+ (Multi-Omics Factor Analysis v2)
implementation planning, specifically tailored for non-parametric multi-omic
plant stress response data.

The script performs the following key functions:
1.  Analyzes individual CSV datasets (spectral and metabolite) to extract
    detailed metadata, including data shape, memory usage, missing values,
    experimental design counts, feature statistics, and data type-specific
    information (wavelengths for spectral, clusters for metabolite).
2.  Evaluates the compatibility of multiple datasets for integration with MOFA+,
    checking for sample alignment, consistent data nature (non-parametric),
    and feature counts.
3.  Generates recommendations for MOFA+ implementation, including view structures,
    parameter settings (factor count, sparsity), and critical workflow elements,
    with specific considerations for non-parametric data.
4.  Creates structured text file reports summarizing the analyses,
    including individual dataset metadata, compatibility analysis,
    MOFA+ view definitions, and a detailed implementation workflow.
5.  Merges all generated reports into a single comprehensive text file and
    also saves a consolidated JSON file containing all metadata.

The primary goal is to prepare and guide the setup of a MOFA+ analysis pipeline,
emphasizing robust practices for non-parametric data, such as appropriate scaling,
non-parametric statistical validation, and careful feature selection strategies.
"""
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def calculate_mode(numbers):
    """Calculate the most common value in a list."""
    counter = Counter(numbers)
    max_count = max(counter.values())
    return [num for num, count in counter.items() if count == max_count][0]


def analyze_metadata(file_path, output_path, file_description, file_type):
    """
    Analyze metadata from a CSV file and save insights to a text file.

    Parameters:
    - file_path: Path to the CSV file.
    - output_path: Path to save the metadata text file.
    - file_description: Description of the file.
    - file_type: Type of data (spectral or metabolite).
    """
    start_time = time.time()

    print(f"Loading {file_path}...")
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Create a placeholder metadata for files that can't be loaded
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_description": file_description,
            "file_type": file_type,
            "error": str(e),
            "load_status": "failed"
        }
        return metadata

    metadata = {
        "file_name": os.path.basename(file_path),
        "file_description": file_description,
        "file_type": file_type,
        "data_nature": "non-parametric",
        "shape": data.shape,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "metadata_columns": list(data.columns[:9]),
        "feature_columns": int(len(data.columns) - 9),
        "memory_usage_mb": float(data.memory_usage(deep=True).sum() / (1024 * 1024)),
        "data_types": {col: str(dtype) for col, dtype in data.dtypes.items() if col in data.columns[:20]},
        "missing_values": {col: int(data[col].isna().sum()) for col in data.columns if data[col].isna().sum() > 0},
        "total_missing_values": int(data.isna().sum().sum()),
        "load_status": "success",
        "processing_time_seconds": None,
    }

    metadata["experimental_design"] = {
        "tissue_types": {k: int(v) for k, v in data["Tissue.type"].value_counts().to_dict().items()},
        "genotypes": {k: int(v) for k, v in data["Genotype"].value_counts().to_dict().items()},
        "treatments": {k: int(v) for k, v in data["Treatment"].value_counts().to_dict().items()},
        "days": {k: int(v) for k, v in data["Day"].value_counts().to_dict().items()},
        "batches": {k: int(v) for k, v in data["Batch"].value_counts().to_dict().items()},
        "replications": {k: int(v) for k, v in data["Replication"].value_counts().to_dict().items()},
    }

    metadata["experimental_combinations"] = {
        "genotype_tissue_treatment_day": int(len(data.groupby(['Genotype', 'Tissue.type', 'Treatment', 'Day']))),
        "genotype_tissue_treatment_day_batch": int(len(data.groupby(['Genotype', 'Tissue.type', 'Treatment', 'Day', 'Batch']))),
        "genotype_tissue_treatment_day_batch_replication": int(len(data.groupby(['Genotype', 'Tissue.type', 'Treatment', 'Day', 'Batch', 'Replication']))),
    }

    sample_rows = data['Row_names'].iloc[:20].tolist()
    augmentation_pattern = any(['augm' in str(row).lower() for row in sample_rows])
    metadata["augmentation"] = {
        "detected_pattern": augmentation_pattern,
        "expected_augmentation": 8,
        "expected_non_augmented_rows": 168,
        "expected_augmented_rows": 1344,
        "actual_rows": int(data.shape[0]),
    }

    feature_cols = data.columns[9:]
    sample_size = min(100, len(feature_cols))
    sample_cols = np.random.choice(feature_cols, sample_size, replace=False)

    metadata["feature_stats"] = {
        "sample_size": int(sample_size),
        "mean_range": (float(data[sample_cols].mean().min()), float(data[sample_cols].mean().max())),
        "median_range": (float(data[sample_cols].median().min()), float(data[sample_cols].median().max())),
        "iqr_range": (
            float((data[sample_cols].quantile(0.75) - data[sample_cols].quantile(0.25)).min()),
            float((data[sample_cols].quantile(0.75) - data[sample_cols].quantile(0.25)).max())
        ),
        "min_range": (float(data[sample_cols].min().min()), float(data[sample_cols].min().max())),
        "max_range": (float(data[sample_cols].max().min()), float(data[sample_cols].max().max())),
        "skewness_range": (float(data[sample_cols].skew().min()), float(data[sample_cols].skew().max())),
        "kurtosis_range": (float(data[sample_cols].kurtosis().min()), float(data[sample_cols].kurtosis().max())),
    }

    metadata["preprocessing_recommendations"] = {
        "transformation": "None required - already appropriately processed",
        "scaling": "Unit variance scaling within each modality",
        "normalization": "Avoid normalization methods that assume normal distribution",
        "statistical_tests": "Use non-parametric tests (e.g., permutation testing) for validation"
    }

    if file_type == "spectral":
        wavelengths = [int(col.split('_')[1]) for col in feature_cols]
        steps = [wavelengths[i+1] - wavelengths[i] for i in range(len(wavelengths)-1)]
        step_size = calculate_mode(steps)

        metadata["wavelength_analysis"] = {
            "min_wavelength": int(min(wavelengths)),
            "max_wavelength": int(max(wavelengths)),
            "wavelength_step": int(step_size),
            "total_wavelengths": int(len(wavelengths)),
        }

        regions = {
            "UV_region": int(sum(1 for w in wavelengths if w < 400)),
            "visible_region": int(sum(1 for w in wavelengths if 400 <= w < 700)),
            "NIR_region": int(sum(1 for w in wavelengths if 700 <= w < 1000)),
            "SWIR_region": int(sum(1 for w in wavelengths if w >= 1000)),
        }
        metadata["wavelength_analysis"]["regions"] = regions

    elif file_type == "metabolite":
        positive_clusters = [col for col in feature_cols if col.startswith('P_Cluster')]
        negative_clusters = [col for col in feature_cols if col.startswith('N_Cluster')]
        metadata["cluster_analysis"] = {
            "positive_mode_clusters": int(len(positive_clusters)),
            "negative_mode_clusters": int(len(negative_clusters)),
            "total_clusters": int(len(positive_clusters) + len(negative_clusters)),
            "positive_cluster_ids": [int(col.split('_')[2]) for col in positive_clusters[:10]] + ["..."],
            "negative_cluster_ids": [int(col.split('_')[2]) for col in negative_clusters[:10]] + ["..."],
            "anish_transformation": "Already applied"
        }

    feature_variance = data[sample_cols].var().sort_values(ascending=False)
    high_var_threshold = feature_variance.quantile(0.75)
    low_var_threshold = feature_variance.quantile(0.25)

    metadata["variance_analysis"] = {
        "max_variance": float(feature_variance.max()),
        "min_variance": float(feature_variance.min()),
        "median_variance": float(feature_variance.median()),
        "high_variance_threshold": float(high_var_threshold),
        "low_variance_threshold": float(low_var_threshold),
        "high_variance_features_count": int((feature_variance > high_var_threshold).sum()),
        "low_variance_features_count": int((feature_variance < low_var_threshold).sum()),
    }

    categorical_cols = ['Genotype', 'Tissue.type', 'Treatment', 'Day', 'Batch']
    feature_factor_correlations = {}
    small_feature_sample = np.random.choice(sample_cols, min(20, len(sample_cols)), replace=False)

    for cat_col in categorical_cols:
        cat_values = data[cat_col].unique()
        means = {}
        for val in cat_values:
            means[val] = data[data[cat_col] == val][small_feature_sample].mean()

        max_diffs = {}
        for feature in small_feature_sample:
            feature_means = [means[val][feature] for val in cat_values]
            max_diffs[feature] = max(feature_means) - min(feature_means)

        feature_factor_correlations[cat_col] = {
            "max_difference": float(max(max_diffs.values())),
            "min_difference": float(min(max_diffs.values())),
            "avg_difference": float(sum(max_diffs.values()) / len(max_diffs)),
            "most_affected_features": list(sorted(max_diffs, key=max_diffs.get, reverse=True)[:5]),
        }
    metadata["feature_factor_correlations"] = feature_factor_correlations

    end_time = time.time()
    metadata["processing_time_seconds"] = float(end_time - start_time)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Metadata Analysis for {file_description}\n")
        f.write("======================================\n")
        f.write(f"File: {os.path.basename(file_path)}\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("1. Basic Information\n")
        f.write(f"   - Data Nature: {metadata['data_nature']} (No further transformation required)\n")
        f.write(f"   - Rows: {metadata['rows']}\n")
        f.write(f"   - Columns: {metadata['columns']}\n")
        f.write(f"   - Feature Columns: {metadata['feature_columns']}\n")
        f.write(f"   - Memory Usage: {metadata['memory_usage_mb']:.2f} MB\n")
        f.write(f"   - Total Missing Values: {metadata['total_missing_values']}\n\n")

        f.write("2. Experimental Design\n")
        for factor, counts in metadata["experimental_design"].items():
            f.write(f"   - {factor.capitalize()}: {counts}\n")
        f.write("\n")

        f.write("3. Experimental Combinations\n")
        for combo, count in metadata["experimental_combinations"].items():
            f.write(f"   - {combo}: {count} unique combinations\n")
        f.write("\n")

        f.write("4. Data Augmentation\n")
        f.write(f"   - Detected Augmentation Pattern: {metadata['augmentation']['detected_pattern']}\n")
        f.write(f"   - Expected Augmentation Factor: {metadata['augmentation']['expected_augmentation']}x\n")
        f.write(f"   - Expected Original Rows: {metadata['augmentation']['expected_non_augmented_rows']}\n")
        f.write(f"   - Expected Augmented Rows: {metadata['augmentation']['expected_augmented_rows']}\n")
        f.write(f"   - Actual Rows: {metadata['augmentation']['actual_rows']}\n\n")

        f.write("5. Feature Statistics (Sample of Features)\n")
        f.write(f"   - Sample Size: {metadata['feature_stats']['sample_size']} features\n")
        f.write(f"   - Mean Range: {metadata['feature_stats']['mean_range']}\n")
        f.write("   - Median Range: {0} (appropriate for non-parametric data)\n".format(metadata['feature_stats']['median_range']))
        f.write("   - IQR Range: {0} (appropriate for non-parametric data)\n".format(metadata['feature_stats']['iqr_range']))
        f.write(f"   - Min Range: {metadata['feature_stats']['min_range']}\n")
        f.write(f"   - Max Range: {metadata['feature_stats']['max_range']}\n")
        f.write(f"   - Skewness Range: {metadata['feature_stats']['skewness_range']}\n")
        f.write(f"   - Kurtosis Range: {metadata['feature_stats']['kurtosis_range']}\n\n")

        f.write("6. Preprocessing Recommendations for Non-Parametric Data\n")
        for key, value in metadata["preprocessing_recommendations"].items():
            f.write(f"   - {key.capitalize()}: {value}\n")
        f.write("\n")

        f.write("7. Variance Analysis (Important for Feature Selection)\n")
        f.write(f"   - Max Variance: {metadata['variance_analysis']['max_variance']:.6f}\n")
        f.write(f"   - Min Variance: {metadata['variance_analysis']['min_variance']:.6f}\n")
        f.write(f"   - Median Variance: {metadata['variance_analysis']['median_variance']:.6f}\n")
        f.write(f"   - High Variance Threshold (75th percentile): {metadata['variance_analysis']['high_variance_threshold']:.6f}\n")
        f.write(f"   - Low Variance Threshold (25th percentile): {metadata['variance_analysis']['low_variance_threshold']:.6f}\n")
        f.write(f"   - High Variance Features Count: {metadata['variance_analysis']['high_variance_features_count']}\n")
        f.write(f"   - Low Variance Features Count: {metadata['variance_analysis']['low_variance_features_count']}\n\n")

        f.write("8. Feature-Factor Correlations\n")
        for factor, stats in metadata["feature_factor_correlations"].items():
            f.write(f"   - {factor} factor impacts:\n")
            f.write(f"     * Max difference between groups: {stats['max_difference']:.4f}\n")
            f.write(f"     * Min difference between groups: {stats['min_difference']:.4f}\n")
            f.write(f"     * Average difference: {stats['avg_difference']:.4f}\n")
            f.write(f"     * Most affected features: {', '.join(stats['most_affected_features'])}\n")
        f.write("\n")

        if file_type == "spectral":
            f.write("9. Wavelength Analysis\n")
            f.write(f"   - Min Wavelength: {metadata['wavelength_analysis']['min_wavelength']} nm\n")
            f.write(f"   - Max Wavelength: {metadata['wavelength_analysis']['max_wavelength']} nm\n")
            f.write(f"   - Wavelength Step: {metadata['wavelength_analysis']['wavelength_step']} nm\n")
            f.write(f"   - Total Wavelengths: {metadata['wavelength_analysis']['total_wavelengths']}\n")
            f.write("   - Wavelength Regions:\n")
            for region, count in metadata["wavelength_analysis"]["regions"].items():
                f.write(f"     * {region}: {count} wavelengths\n")
            f.write("\n")
        elif file_type == "metabolite":
            f.write("9. Cluster Analysis\n")
            f.write(f"   - Positive Mode Clusters: {metadata['cluster_analysis']['positive_mode_clusters']}\n")
            f.write(f"   - Negative Mode Clusters: {metadata['cluster_analysis']['negative_mode_clusters']}\n")
            f.write(f"   - Total Clusters: {metadata['cluster_analysis']['total_clusters']}\n")
            f.write(f"   - Sample Positive Cluster IDs: {metadata['cluster_analysis']['positive_cluster_ids']}\n")
            f.write(f"   - Sample Negative Cluster IDs: {metadata['cluster_analysis']['negative_cluster_ids']}\n")
            f.write(f"   - Anish Transformation: {metadata['cluster_analysis']['anish_transformation']}\n\n")

        f.write(f"Analysis completed in {metadata['processing_time_seconds']:.2f} seconds\n")

    print(f"Metadata analysis for {file_description} completed and saved to {output_path}")
    return metadata


def analyze_dataset_compatibility(metadata_list, output_path):
    """
    Analyze the compatibility of the datasets for integration with MOFA+.

    Parameters:
    - metadata_list: List of metadata dictionaries for each dataset.
    - output_path: Path to save the compatibility analysis.
    """
    compatibility = {
        "data_nature": "non-parametric",
        "sample_alignment": {},
        "feature_counts": {},
        "variance_considerations": {},
        "integration_recommendations": {},
    }

    failed_datasets = [meta for meta in metadata_list if meta.get("load_status") == "failed"]
    if failed_datasets:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("MOFA+ Dataset Compatibility Analysis\n")
            f.write("==================================\n\n")
            f.write("ERROR: The following datasets failed to load:\n")
            for meta in failed_datasets:
                f.write(f"- {meta['file_name']}: {meta.get('error', 'Unknown error')}\n")
            f.write("\nPlease resolve these issues before proceeding with MOFA+ implementation.\n")
        return compatibility

    sample_counts = [meta["rows"] for meta in metadata_list]
    tissues = [meta["file_description"].split()[0].lower() for meta in metadata_list]
    data_types = [meta["file_type"] for meta in metadata_list]

    compatibility["sample_alignment"]["all_datasets_same_samples"] = len(set(sample_counts)) == 1
    compatibility["sample_alignment"]["sample_counts"] = dict(zip([meta["file_name"] for meta in metadata_list], sample_counts))

    expected_rows = 1344
    compatibility["sample_alignment"]["matches_expected_augmentation"] = all(count == expected_rows for count in sample_counts)

    feature_counts = [meta["feature_columns"] for meta in metadata_list]
    compatibility["feature_counts"]["counts"] = dict(zip([meta["file_name"] for meta in metadata_list], feature_counts))
    compatibility["feature_counts"]["total_features"] = sum(feature_counts)

    compatibility["variance_considerations"]["high_variance_counts"] = {
        meta["file_name"]: meta["variance_analysis"]["high_variance_features_count"]
        for meta in metadata_list
    }
    compatibility["variance_considerations"]["low_variance_counts"] = {
        meta["file_name"]: meta["variance_analysis"]["low_variance_features_count"]
        for meta in metadata_list
    }

    compatibility["non_parametric_considerations"] = {
        "data_preprocessing": "Data is already appropriately pre-processed (no further transformation required)",
        "scaling_recommendation": "Unit variance scaling within each modality only",
        "statistical_testing": "Use non-parametric tests (permutation testing, bootstrap validation)",
        "factor_determination": "Rely on ARD and bootstrap stability rather than parametric significance tests",
        "mofa_advantage": "MOFA+ is well-suited for non-parametric data as it doesn't make strong distributional assumptions"
    }

    compatibility["integration_recommendations"]["recommended_scaling"] = "unit_variance_within_modality"
    compatibility["integration_recommendations"]["mofa_views"] = list(set([f"{tissue}_{data_type}" for tissue, data_type in zip(tissues, data_types)]))
    compatibility["integration_recommendations"]["suggested_feature_selection_strategy"] = "bootstrap_stability"

    avg_feature_per_view = sum(feature_counts) / len(feature_counts) if feature_counts else 0
    est_factors = min(20, int(avg_feature_per_view / 100)) if avg_feature_per_view > 0 else 10

    compatibility["mofa_model_estimates"] = {
        "suggested_factor_range": (10, 20),
        "estimated_initial_factors": est_factors,
        "estimated_memory_usage_gb": float(sum([meta["memory_usage_mb"] for meta in metadata_list]) / 1024 * 1.5),
    }

    compatibility["mofa_parameter_recommendations"] = {
        "factor_count": {
            "range": (10, 20),
            "determination_method": "Explained variance elbow plots & bootstrap stability analysis",
            "notes": "Start higher (20) and let ARD prune unnecessary factors"
        },
        "sparsity_level": {
            "method": "5-fold cross-validation",
            "notes": "Begin with default sparsity setting"
        },
        "model_selection": {
            "method": "ARD (Automatic Relevance Determination)",
            "notes": "Enables automatic factor pruning"
        },
        "cpu_cores": 40,
        "ram_gb": 196,
    }

    compatibility["critical_workflow_elements"] = {
        "mofa_to_transformer_feature_selection": {
            "priority": "Highest",
            "approach": "Bootstrap stability validation & adaptive thresholding",
            "target_features_per_view": 150,
            "notes": "Most critical technical element of implementation"
        },
        "sample_alignment": {
            "priority": "High",
            "approach": "Maintain consistent sample ordering across all datasets",
            "notes": "Essential for proper integration"
        },
        "feature_selection": {
            "priority": "High",
            "approach": "Importance scores from relevant factors",
            "notes": "Focus on features from biologically relevant factors"
        },
        "statistical_validation": {
            "priority": "High",
            "approach": "Bootstrap (n=100), permutation testing, FDR correction",
            "notes": "Apply to all results for statistical rigor"
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("MOFA+ Integration Analysis for Multi-Omic Plant Stress Response Data\n")
        f.write("==================================================================\n\n")

        f.write("1. Dataset Overview\n")
        f.write(f"   - Data Nature: {compatibility['data_nature']} (no further transformation required)\n")
        for i, meta in enumerate(metadata_list):
            f.write(f"   {i+1}. {meta['file_description']} ({meta['file_name']})\n")
            f.write(f"      - Type: {meta['file_type']}\n")
            f.write(f"      - Samples: {meta['rows']}\n")
            f.write(f"      - Features: {meta['feature_columns']}\n")
        f.write("\n")

        f.write("2. Non-Parametric Data Considerations\n")
        for key, value in compatibility["non_parametric_considerations"].items():
            f.write(f"   - {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")

        f.write("3. Sample Alignment\n")
        f.write(f"   - All datasets have the same number of samples: {compatibility['sample_alignment']['all_datasets_same_samples']}\n")
        f.write(f"   - Matches expected augmentation: {compatibility['sample_alignment']['matches_expected_augmentation']}\n")
        f.write("   - Sample counts per dataset:\n")
        for name, count in compatibility['sample_alignment']['sample_counts'].items():
            f.write(f"      - {name}: {count}\n")
        f.write("\n")

        f.write("4. Feature Information\n")
        f.write("   - Feature counts per dataset:\n")
        for name, count in compatibility['feature_counts']['counts'].items():
            f.write(f"      - {name}: {count}\n")
        f.write(f"   - Total features across all datasets: {compatibility['feature_counts']['total_features']}\n")
        f.write("\n")

        f.write("5. Variance Analysis\n")
        f.write("   - High variance features (75th percentile and above):\n")
        for name, count in compatibility['variance_considerations']['high_variance_counts'].items():
            f.write(f"      - {name}: {count}\n")
        f.write("   - Low variance features (25th percentile and below):\n")
        for name, count in compatibility['variance_considerations']['low_variance_counts'].items():
            f.write(f"      - {name}: {count}\n")
        f.write("\n")

        f.write("6. MOFA+ Implementation Recommendations\n")
        f.write(f"   - Recommended scaling: {compatibility['integration_recommendations']['recommended_scaling']}\n")
        f.write("   - MOFA+ views:\n")
        for view in compatibility['integration_recommendations']['mofa_views']:
            f.write(f"      - {view}\n")
        f.write(f"   - Suggested feature selection strategy: {compatibility['integration_recommendations']['suggested_feature_selection_strategy']}\n")
        f.write("\n")

        f.write("7. MOFA+ Model Parameters\n")
        f.write(f"   - Suggested factor range: {compatibility['mofa_parameter_recommendations']['factor_count']['range']}\n")
        f.write("   - Factor determination method: {compatibility['mofa_parameter_recommendations']['factor_count']['determination_method']}\n")
        f.write(f"   - Sparsity level optimization: {compatibility['mofa_parameter_recommendations']['sparsity_level']['method']}\n")
        f.write(f"   - Model selection: {compatibility['mofa_parameter_recommendations']['model_selection']['method']}\n")
        f.write(f"   - Computational resources: {compatibility['mofa_parameter_recommendations']['cpu_cores']} cores, {compatibility['mofa_parameter_recommendations']['ram_gb']} GB RAM\n")
        f.write(f"   - Estimated memory usage: {compatibility['mofa_model_estimates']['estimated_memory_usage_gb']:.2f} GB\n")
        f.write("\n")

        f.write("8. Critical Workflow Elements\n")
        for element, details in compatibility['critical_workflow_elements'].items():
            f.write(f"   - {element.replace('_', ' ').title()}\n")
            f.write(f"     * Priority: {details['priority']}\n")
            f.write(f"     * Approach: {details['approach']}\n")
            if 'target_features_per_view' in details:
                f.write(f"     * Target features per view: {details['target_features_per_view']}\n")
            f.write(f"     * Notes: {details['notes']}\n")
        f.write("\n")

        f.write("9. MOFA+ Expected Output Files\n")
        f.write("   1. mofa_model.hdf5 - Complete trained model\n")
        f.write("   2. mofa_latent_factors.csv - Sample scores on factors\n")
        f.write("   3. mofa_variance_explained.csv - Per-view variance breakdown\n")
        f.write("   4. mofa_feature_weights_{view}.csv - Feature-factor loadings\n")
        f.write("   5. mofa_factor_metadata_associations.csv - Factor correlations with experimental variables\n")
        f.write("   6. mofa_top_features_{view}.csv - Feature ranking for each tissue/data type\n")
        f.write("   7. mofa_bootstrap_stability.csv - Feature selection consistency metrics\n")
        f.write("   8. transformer_input_{view}.csv - Processed data with selected features for Transformer input\n")
        f.write("\n")

        f.write("10. Implementation Challenges and Solutions for Non-Parametric Data\n")
        f.write("   A. MOFA+ to Transformer Feature Selection (Most Critical Element)\n")
        f.write("      - Challenge: Ensuring robust and biologically meaningful feature selection\n")
        f.write("      - Solution: Use bootstrap validation, adaptive thresholds, preserve feature clusters\n")
        f.write("      - Implementation: Focus on features associated with biologically relevant factors\n\n")

        f.write("   B. Integration Between Methods\n")
        f.write("      - Challenge: Maintaining compatible feature spaces and sample alignment\n")
        f.write("      - Solution: Common data preprocessing, clear feature selection strategy\n")
        f.write("      - Implementation: Use identical sample ordering in all datasets\n\n")

        f.write("   C. Consensus Scoring Framework\n")
        f.write("      - Challenge: Reconciling different importance metrics\n")
        f.write("      - Solution: Weight MOFA+ scores (0.6) higher than Transformer scores (0.4)\n")
        f.write("      - Implementation: Z-normalize within method, apply weighted combination\n\n")

        f.write("   D. Statistical Validation Framework\n")
        f.write("      - Challenge: Ensuring statistical rigor with non-parametric data\n")
        f.write("      - Solution: Bootstrap validation, permutation testing, FDR correction\n")
        f.write("      - Implementation: Apply consistent statistical framework to all results\n\n")

    print("Dataset compatibility analysis completed and saved to", output_path)
    return compatibility


def create_mofa_view_summary(metadata_list, output_path):
    """
    Create a summary of the MOFA+ views structure based on the metadata.

    Parameters:
    - metadata_list: List of metadata dictionaries for each dataset.
    - output_path: Path to save the MOFA+ views summary.
    """
    tissues = [meta["file_description"].split()[0].lower() for meta in metadata_list]
    data_types = [meta["file_type"] for meta in metadata_list]
    file_names = [meta["file_name"] for meta in metadata_list]

    views = {}
    for tissue, data_type, file_name in zip(tissues, data_types, file_names):
        view_name = f"{tissue}_{data_type}"
        if view_name not in views:
            views[view_name] = {
                "name": view_name,
                "file": file_name,
                "tissue": tissue,
                "data_type": data_type,
                "feature_count": metadata_list[file_names.index(file_name)]["feature_columns"],
                "data_nature": "non-parametric"
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("MOFA+ View Structure for Multi-Omic Plant Stress Response Data\n")
        f.write("==========================================================\n\n")

        f.write("MOFA+ requires clearly defined 'views' that represent different data modalities.\n")
        f.write("Based on the dataset analysis, the following views are recommended:\n\n")

        for i, (view_name, view_info) in enumerate(views.items()):
            f.write(f"View {i+1}: {view_name}\n")
            f.write(f"  - Data Source: {view_info['file']}\n")
            f.write(f"  - Tissue: {view_info['tissue']}\n")
            f.write(f"  - Data Type: {view_info['data_type']}\n")
            f.write(f"  - Data Nature: {view_info['data_nature']}\n")
            f.write(f"  - Feature Count: {view_info['feature_count']}\n\n")

        f.write("For MOFA+ Implementation with Non-Parametric Data:\n")
        f.write("  - Each view should be stored as a separate matrix\n")
        f.write("  - Sample order must be identical across all views\n")
        f.write("  - Each view will have unit variance scaling applied within the view\n")
        f.write("  - No further transformation is required as data is already pre-processed\n")
        f.write("  - For metabolite data, Anish transformation has already been applied\n")
        f.write("  - For spectral data, preprocessing has already been performed\n\n")

        f.write("Expected Output Files per View:\n")
        for view_name in views.keys():
            f.write(f"  - mofa_feature_weights_{view_name}.csv\n")
            f.write(f"  - mofa_top_features_{view_name}.csv\n")
            f.write(f"  - transformer_input_{view_name}.csv\n")
        f.write("\n")

    print("MOFA+ view structure summary created and saved to", output_path)
    return views


def create_implementation_workflow(output_path):
    """
    Create a summary of the recommended MOFA+ implementation workflow.

    Parameters:
    - output_path: Path to save the implementation workflow.
    """
    workflow = {
        "phase1_data_preprocessing": {
            "steps": [
                "Load spectral and metabolite data (already appropriately pre-processed for non-parametric analysis)",
                "Verify sample alignment across datasets",
                "Unit variance scaling within each modality (appropriate for non-parametric data)",
                "Create matched sample matrices with consistent ordering",
                "Split data for cross-validation (80% train, 10% validate, 10% test)"
            ],
            "outputs": [
                "preprocessed_data_{view}.csv - Scaled data per view",
                "train_test_split.json - Sample indices for CV"
            ]
        },
        "phase2_mofa_optimization": {
            "steps": [
                "Implement parameter optimization framework",
                "Test factor count range (10-20)",
                "Optimize sparsity via 5-fold CV",
                "Implement ARD for automatic factor pruning",
                "Perform bootstrap stability analysis (100 initializations)",
                "Select optimal parameters based on explained variance and stability"
            ],
            "outputs": [
                "mofa_parameter_optimization.csv - Results of parameter search",
                "mofa_bootstrap_results.csv - Bootstrap stability metrics",
                "mofa_optimal_parameters.json - Selected parameters"
            ]
        },
        "phase3_mofa_analysis": {
            "steps": [
                "Run MOFA+ with optimized parameters",
                "Extract and analyze factors and loadings",
                "Calculate factor-experimental condition associations",
                "Identify significant features for each factor",
                "Rank features by importance within each view",
                "Generate bootstrap confidence intervals"
            ],
            "outputs": [
                "mofa_model.hdf5 - Complete trained model",
                "mofa_latent_factors.csv - Sample scores on factors",
                "mofa_variance_explained.csv - Per-view variance breakdown",
                "mofa_feature_weights_{view}.csv - Feature-factor loadings",
                "mofa_factor_metadata_associations.csv - Factor correlations with experimental variables",
                "mofa_top_features_{view}.csv - Feature ranking for each tissue/data type",
                "mofa_bootstrap_stability.csv - Feature selection consistency metrics"
            ],
            "non_parametric_considerations": [
                "Use non-parametric correlation measures for factor-condition associations",
                "Rely on bootstrap validation rather than parametric significance tests",
                "Apply permutation testing to establish baseline significance",
                "Ensure all statistical tests are appropriate for non-parametric data"
            ]
        },
        "phase4_feature_selection": {
            "steps": [
                "Identify biologically relevant factors",
                "Calculate importance scores for each feature",
                "Apply adaptive threshold for feature selection",
                "Extract top 150 features per view",
                "Create filtered data matrices with selected features",
                "Normalize filtered matrices for Transformer input"
            ],
            "outputs": [
                "transformer_input_{view}.csv - Processed data with selected features"
            ]
        },
        "phase5_statistical_validation": {
            "steps": [
                "Implement bootstrap significance testing (n=100)",
                "Perform permutation testing for factor significance",
                "Apply FDR correction for multiple comparisons",
                "Calculate confidence intervals for key metrics",
                "Validate feature selection stability"
            ],
            "outputs": [
                "mofa_statistical_significance.csv - Statistical validation results",
                "mofa_permutation_results.csv - Permutation test results",
                "mofa_confidence_intervals.csv - Confidence intervals"
            ]
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("MOFA+ Implementation Workflow for Non-Parametric Multi-Omic Plant Stress Response Analysis\n")
        f.write("====================================================================================\n\n")

        f.write("The following implementation workflow is recommended for the non-parametric data:\n\n")

        for phase, details in workflow.items():
            phase_name = phase.replace('_', ' ').title()
            f.write(f"{phase_name}\n")
            f.write(f"{'-' * len(phase_name)}\n\n")

            f.write("Steps:\n")
            for i, step in enumerate(details["steps"]):
                f.write(f"  {i+1}. {step}\n")
            f.write("\n")

            if "non_parametric_considerations" in details:
                f.write("Non-Parametric Data Considerations:\n")
                for consideration in details["non_parametric_considerations"]:
                    f.write(f"  - {consideration}\n")
                f.write("\n")

            f.write("Expected Outputs:\n")
            for output_file in details["outputs"]:
                f.write(f"  - {output_file}\n")
            f.write("\n\n")

        f.write("Critical Implementation Notes for Non-Parametric Data:\n")
        f.write("-----------------------------------------------\n\n")

        f.write("1. MOFA+ to Transformer Feature Selection\n")
        f.write("   This is the single most critical technical element. Implement a robust pipeline that:\n")
        f.write("   - Validates feature importance stability via bootstrapping\n")
        f.write("   - Uses adaptive threshold based on feature weight distribution\n")
        f.write("   - Preserves feature clusters that may operate together\n")
        f.write("   - Maintains tissue-specific features\n\n")

        f.write("2. Consensus Scoring Framework\n")
        f.write("   Implement this well-defined consensus metric combining evidence from both methods:\n")
        f.write("   ```python\n")
        f.write("   def calculate_consensus_score(mofa_score, transformer_score):\n")
        f.write("       # Z-normalize within method\n")
        f.write("       mofa_z = (mofa_score - np.mean(mofa_scores)) / np.std(mofa_scores)\n")
        f.write("       transformer_z = (transformer_score - np.mean(transformer_scores)) / np.std(transformer_scores)\n")
        f.write("       \n")
        f.write("       # Weighted combination (MOFA gets higher weight due to statistical robustness)\n")
        f.write("       consensus = (0.6 * mofa_z) + (0.4 * transformer_z)\n")
        f.write("       \n")
        f.write("       # Assign confidence tier\n")
        f.write("       if mofa_z > threshold_high and transformer_z > threshold_high:\n")
        f.write("           confidence = \"high\"\n")
        f.write("       elif mofa_z > threshold_med and transformer_z > threshold_med:\n")
        f.write("           confidence = \"medium\"\n")
        f.write("       else:\n")
        f.write("           confidence = \"low\"\n")
        f.write("           \n")
        f.write("       return consensus, confidence\n")
        f.write("   ```\n\n")

        f.write("3. Statistical Validation Framework for Non-Parametric Data\n")
        f.write("   Implement a comprehensive statistical framework that includes:\n")
        f.write("   - Bootstrap validation (n=100) for all feature importance scores\n")
        f.write("   - Permutation testing to establish baseline significance\n")
        f.write("   - FDR correction (Benjamini-Hochberg) for all multiple comparisons\n")
        f.write("   - Confidence intervals for all reported metrics\n")
        f.write("   - Effect size calculations beyond p-values\n")
        f.write("   - Avoid tests that assume normality or other parametric distributions\n\n")

        f.write("4. Transformer Implementation Simplification\n")
        f.write("   Keep the Transformer implementation minimal:\n")
        f.write("   - Use only 2 layers to reduce complexity\n")
        f.write("   - Focus on cross-attention between modalities\n")
        f.write("   - Implement baseline models first (Random Forest, SVM)\n")
        f.write("   - Use early stopping (patience=10) to prevent overfitting\n")
        f.write("   - Begin with a subset of data to validate the approach\n\n")

        f.write("5. Data Preprocessing Considerations for Non-Parametric Data\n")
        f.write("   - No need for further transformation as data is already appropriately processed\n")
        f.write("   - For metabolite data, Anish transformation has already been applied\n")
        f.write("   - Apply unit variance scaling within each modality\n")
        f.write("   - Avoid normalization methods that assume normal distribution\n")
        f.write("   - Maintain consistent sample ordering across all datasets\n\n")

    print("Implementation workflow created and saved to", output_path)
    return workflow


def merge_text_files(output_dir, output_file="mofa_complete_analysis.txt"):
    """
    Merge all generated text files into a single comprehensive file.

    Parameters:
    - output_dir: Directory containing the individual text files.
    - output_file: Name of the merged output file.
    """
    file_order = [
        "mofa_master_summary.txt",
        "dataset_compatibility.txt",
        "mofa_view_structure.txt",
        "mofa_implementation_workflow.txt",
        "leaf_spectral_metadata.txt",
        "root_spectral_metadata.txt",
        "leaf_metabolite_metadata.txt",
        "root_metabolite_metadata.txt"
    ]

    with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as outfile:
        outfile.write("MOFA+ COMPREHENSIVE ANALYSIS REPORT FOR NON-PARAMETRIC DATA\n")
        outfile.write("======================================================\n\n")
        outfile.write("This document contains all analysis results merged into a single file.\n")
        outfile.write("The data is non-parametric in nature and already appropriately processed.\n\n")

        for filename in file_order:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                outfile.write("\n\n" + "="*80 + "\n")
                outfile.write(f"SECTION: {filename}\n")
                outfile.write("="*80 + "\n\n")

                with open(filepath, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())

    print(f"All text files merged into {output_file}")


def create_metadata_summary(file_paths, output_dir):
    """
    Generate comprehensive metadata analysis for all datasets.

    Parameters:
    - file_paths: Dictionary of file paths.
    - output_dir: Directory to save outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating metadata summary in {output_dir}")

    metadata_list = []

    metadata_leaf_spectral = analyze_metadata(
        file_paths["spectral_leaf"],
        os.path.join(output_dir, "leaf_spectral_metadata.txt"),
        "Leaf Spectral Data",
        "spectral"
    )
    metadata_list.append(metadata_leaf_spectral)

    metadata_root_spectral = analyze_metadata(
        file_paths["spectral_root"],
        os.path.join(output_dir, "root_spectral_metadata.txt"),
        "Root Spectral Data",
        "spectral"
    )
    metadata_list.append(metadata_root_spectral)

    metadata_leaf_metabolite = analyze_metadata(
        file_paths["metabolite_leaf"],
        os.path.join(output_dir, "leaf_metabolite_metadata.txt"),
        "Leaf Metabolite Data",
        "metabolite"
    )
    metadata_list.append(metadata_leaf_metabolite)

    metadata_root_metabolite = analyze_metadata(
        file_paths["metabolite_root"],
        os.path.join(output_dir, "root_metabolite_metadata.txt"),
        "Root Metabolite Data",
        "metabolite"
    )
    metadata_list.append(metadata_root_metabolite)

    compatibility = analyze_dataset_compatibility(
        metadata_list,
        os.path.join(output_dir, "dataset_compatibility.txt")
    )

    views = create_mofa_view_summary(
        metadata_list,
        os.path.join(output_dir, "mofa_view_structure.txt")
    )

    workflow = create_implementation_workflow(
        os.path.join(output_dir, "mofa_implementation_workflow.txt")
    )

    all_metadata = {
        "datasets": {
            "leaf_spectral": metadata_leaf_spectral,
            "root_spectral": metadata_root_spectral,
            "leaf_metabolite": metadata_leaf_metabolite,
            "root_metabolite": metadata_root_metabolite
        },
        "data_nature": "non-parametric",
        "compatibility": compatibility,
        "views": views,
        "workflow": workflow
    }

    with open(os.path.join(output_dir, "all_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, cls=NumpyEncoder)

    with open(os.path.join(output_dir, "mofa_master_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("MOFA+ Implementation Master Summary for Non-Parametric Data\n")
        f.write("=======================================================\n\n")

        f.write("This document provides a comprehensive summary of the multi-omic\n")
        f.write("plant stress response dataset analysis and MOFA+ implementation plan.\n")
        f.write("The data is non-parametric in nature and has already been appropriately processed.\n\n")

        f.write("Contents:\n")
        f.write("1. Dataset Overview\n")
        f.write("2. Non-Parametric Data Considerations\n")
        f.write("3. MOFA+ Implementation Requirements\n")
        f.write("4. Key Challenge Areas\n")
        f.write("5. Expected Outputs\n")
        f.write("6. Next Steps\n\n")

        f.write("Non-Parametric Data Implications:\n")
        f.write("- No further transformation required as data is already pre-processed\n")
        f.write("- Unit variance scaling within each modality is appropriate\n")
        f.write("- Non-parametric statistical tests should be used for validation\n")
        f.write("- Bootstrap and permutation methods are preferred over parametric tests\n")
        f.write("- MOFA+ is well-suited as it doesn't make strong distributional assumptions\n\n")

        f.write("For detailed information, please refer to the following files:\n")
        f.write("- dataset_compatibility.txt - Analysis of compatibility between datasets\n")
        f.write("- mofa_view_structure.txt - Recommended MOFA+ view structure\n")
        f.write("- mofa_implementation_workflow.txt - Step-by-step implementation process\n")
        f.write("- [dataset]_metadata.txt - Detailed analysis of each dataset\n")
        f.write("- all_metadata.json - Comprehensive metadata in JSON format\n")
        f.write("- mofa_complete_analysis.txt - All analyses merged into a single file\n\n")

        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    merge_text_files(output_dir)

    print("Metadata summary generation completed. All files saved to", output_dir)


def main():
    """Main function to execute the metadata analysis workflow."""
    file_paths = {
        "spectral_leaf": "C:/Users/ms/Desktop/hyper/data/hyper_l_w_augmt.csv",
        "spectral_root": "C:/Users/ms/Desktop/hyper/data/hyper_r_w_augmt.csv",
        "metabolite_leaf": "C:/Users/ms/Desktop/hyper/data/n_p_l2_augmt.csv",
        "metabolite_root": "C:/Users/ms/Desktop/hyper/data/n_p_r2_augmt.csv"
    }

    output_dir = "C:/Users/ms/Desktop/hyper/output/mofa/test"

    create_metadata_summary(file_paths, output_dir)

    print("All analyses completed. Results saved to:", output_dir)
    print(f"Comprehensive merged analysis available at: {os.path.join(output_dir, 'mofa_complete_analysis.txt')}")


if __name__ == "__main__":
    main()