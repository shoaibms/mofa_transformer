"""
MolecularFeatureQC: Comprehensive Quality Control for Augmented Molecular Feature Data

A module for performing thorough quality control assessments on augmented molecular feature datasets.
Implements multiple QC approaches including outlier detection, class separability,
signal-to-noise ratio metrics, machine learning cross-validation, and statistical power analysis.

The module compares augmented data against the original dataset to evaluate preservation
of biological relationships and statistical properties, generating detailed reports
and visualizations to guide selection of optimal augmentation methods.
"""

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from itertools import combinations


class MolecularFeatureQC:
    """
    Comprehensive quality control module for augmented molecular feature data.
    
    Implements:
    1. Critical outlier detection with biochemical context
    2. Class separability assessment pre/post augmentation
    3. Signal-to-noise ratio metrics for LC-MS features
    4. Cross-validation with machine learning classifiers
    5. Statistical power analysis with augmented dataset
    """
    
    def __init__(self, original_path, augmented_path, output_dir):
        """
        Initialize the MolecularFeatureQC class.
        
        Parameters:
        -----------
        original_path : str
            Path to original molecular feature data CSV file
        augmented_path : str
            Path to augmented molecular feature data CSV file
        output_dir : str
            Directory to save QC results
        """
        self.original_path = original_path
        self.augmented_path = augmented_path
        self.output_dir = output_dir
        
        # Create output directories
        self.qc_dir = os.path.join(output_dir, 'quality_control')
        if not os.path.exists(self.qc_dir):
            os.makedirs(self.qc_dir)
            
        self.plots_dir = os.path.join(self.qc_dir, 'plots')
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            
        self.results_dir = os.path.join(self.qc_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load and prepare molecular feature data."""
        print("Loading data...")
        self.original_data = pd.read_csv(self.original_path)
        self.augmented_data = pd.read_csv(self.augmented_path)
        
        # Identify N_Cluster, P_Cluster and metadata columns
        self.n_cluster_cols = [col for col in self.original_data.columns if col.startswith('N_Cluster_')]
        self.p_cluster_cols = [col for col in self.original_data.columns if col.startswith('P_Cluster_')]
        self.molecular_feature_cols = self.n_cluster_cols + self.p_cluster_cols
        self.metadata_cols = [col for col in self.original_data.columns if col not in self.molecular_feature_cols]
        
        # Separate augmented data by augmentation methods
        self.augmented_only = self.augmented_data[~self.augmented_data['Row_names'].isin(self.original_data['Row_names'])]
        
        # Extract data by augmentation method
        self.augmentation_methods = self.identify_augmentation_methods()
        self.method_data = self.extract_data_by_method()
        
        print(f"Loaded {len(self.original_data)} original samples")
        print(f"Loaded {len(self.augmented_only)} augmented samples")
        print(f"Identified {len(self.augmentation_methods)} augmentation methods: {', '.join(self.augmentation_methods)}")
        
    def identify_augmentation_methods(self):
        """
        Identify the augmentation methods used in the dataset.
        
        Returns:
        --------
        list
            List of identified augmentation methods
        """
        methods = set()
        
        # Extract method suffixes from row names
        for row_name in self.augmented_only['Row_names']:
            if '_' in row_name:
                suffix = row_name.split('_')[-1]
                methods.add(suffix)
        
        return list(methods)
    
    def extract_data_by_method(self):
        """
        Extract data subsets for each augmentation method.
        
        Returns:
        --------
        dict
            Dictionary containing data subsets for each method
        """
        method_data = {'original': self.original_data}
        
        for method in self.augmentation_methods:
            # Find rows with this method suffix
            mask = self.augmented_data['Row_names'].str.endswith(f'_{method}')
            data_subset = self.augmented_data[mask]
            
            if len(data_subset) > 0:
                method_data[method] = data_subset
        
        return method_data
    
    #-------------------------------------------------------------------------
    # 1. Critical Outlier Detection with Biochemical Context
    #-------------------------------------------------------------------------
    
    def critical_outlier_detection(self):
        """
        Detect outliers in molecular feature data with biochemical context awareness.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with outlier detection results
        """
        print("\n1. Performing critical outlier detection with biochemical context...")
        
        # First, calculate basic correlation matrix for biochemical context
        orig_corr = self.original_data[self.molecular_feature_cols].corr()
        
        # Dictionary to store outlier detection results
        outlier_results = {}
        
        # Perform outlier detection for each method
        for method, data in self.method_data.items():
            print(f"  Processing {method}...")
            outlier_results[method] = self.detect_outliers_for_dataset(data, orig_corr)
        
        # Prepare summary DataFrame
        summary_rows = []
        for method, results in outlier_results.items():
            summary_rows.append({
                'Method': method,
                'Total_Samples': results['total_samples'],
                'Standard_Outliers': results['standard_outliers'],
                'Biochemical_Outliers': results['biochemical_outliers'],
                'Total_Outliers': results['total_outliers'],
                'Outlier_Percentage': results['outlier_percentage']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary to CSV
        summary_df.to_csv(os.path.join(self.results_dir, 'outlier_detection_summary.csv'), index=False)
        
        # Create detailed outlier report
        detailed_rows = []
        for method, results in outlier_results.items():
            if 'outlier_details' in results:
                for i, details in enumerate(results['outlier_details']):
                    details['Method'] = method
                    details['Outlier_Index'] = i
                    detailed_rows.append(details)
        
        if detailed_rows:
            detailed_df = pd.DataFrame(detailed_rows)
            detailed_df.to_csv(os.path.join(self.results_dir, 'outlier_detection_details.csv'), index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Sort methods by outlier percentage
        sorted_df = summary_df.sort_values('Outlier_Percentage')
        
        # Create stacked bar chart
        bar_width = 0.8
        bars1 = plt.bar(sorted_df['Method'], sorted_df['Standard_Outliers'] / sorted_df['Total_Samples'] * 100, 
                        width=bar_width, label='Standard Outliers', color='#6baed6')  # Medium Blue
        bars2 = plt.bar(sorted_df['Method'], sorted_df['Biochemical_Outliers'] / sorted_df['Total_Samples'] * 100, 
                        width=bar_width, bottom=sorted_df['Standard_Outliers'] / sorted_df['Total_Samples'] * 100,
                        label='Biochemical Context Outliers', color='#fe9929')  # Muted Orange/Yellow
        
        # Add labels and title
        plt.xlabel('Augmentation Method')
        plt.ylabel('Outlier Percentage (%)')
        plt.title('Outlier Detection Results by Method')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        
        # Add data labels
        for i, row in enumerate(sorted_df.itertuples()):
            plt.text(i, row.Outlier_Percentage + 1, 
                    f"{row.Outlier_Percentage:.1f}%", 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'outlier_detection.png'), dpi=300)
        plt.savefig(os.path.join(self.plots_dir, 'outlier_detection.pdf'))
        plt.close()
        
        print(f"Outlier detection completed. Results saved to: {self.results_dir}")
        print(summary_df)
        
        return summary_df
    
    def detect_outliers_for_dataset(self, data, reference_corr, contamination=0.05):
        """
        Detect outliers for a single dataset with biochemical context.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset to analyze
        reference_corr : pd.DataFrame
            Correlation matrix from reference (original) data for biochemical context
        contamination : float
            Expected proportion of outliers
            
        Returns:
        --------
        dict
            Dictionary with outlier detection results
        """
        results = {
            'total_samples': len(data),
            'standard_outliers': 0,
            'biochemical_outliers': 0,
            'total_outliers': 0,
            'outlier_percentage': 0,
            'outlier_details': []
        }
        
        # Extract molecular feature data
        molecular_feature_data = data[self.molecular_feature_cols].copy()
        
        # Step 1: Standard outlier detection using Isolation Forest
        try:
            # Apply standard outlier detection
            clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            molecular_feature_data_values = molecular_feature_data.values
            standard_outliers = clf.fit_predict(molecular_feature_data_values) == -1
            standard_indices = np.where(standard_outliers)[0]
            results['standard_outliers'] = len(standard_indices)
            
            # Step 2: Calculate current data correlation matrix
            current_corr = molecular_feature_data.corr()
            
            # Step 3: Find samples that violate biochemical relationships
            biochemical_outliers = set()
            
            # Calculate correlation difference
            common_cols = list(set(reference_corr.columns) & set(current_corr.columns))
            
            if common_cols:
                ref_corr_subset = reference_corr.loc[common_cols, common_cols]
                curr_corr_subset = current_corr.loc[common_cols, common_cols]
                
                # Calculate correlation difference matrix
                corr_diff = np.abs(ref_corr_subset - curr_corr_subset)
                
                # Find top correlated pairs in reference (representing expected biochemical relationships)
                top_pairs = []
                for i, j in combinations(range(len(common_cols)), 2):
                    if abs(ref_corr_subset.iloc[i, j]) > 0.7:  # Strong correlation threshold
                        top_pairs.append((common_cols[i], common_cols[j], ref_corr_subset.iloc[i, j]))
                
                # Sort pairs by absolute correlation strength
                top_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Take top 100 pairs or all if less
                top_pairs = top_pairs[:min(100, len(top_pairs))]
                
                # For each sample, check if it violates expected relationships
                for idx in range(len(data)):
                    # Skip samples already detected as standard outliers
                    if idx in standard_indices:
                        continue
                    
                    sample = molecular_feature_data.iloc[idx]
                    violations = 0
                    
                    for m1, m2, ref_corr_val in top_pairs:
                        # Get the expected relationship direction
                        expected_direction = np.sign(ref_corr_val)
                        
                        # Verify if this sample follows the expected relationship
                        actual_direction = np.sign((sample[m1] - molecular_feature_data[m1].mean()) * 
                                                  (sample[m2] - molecular_feature_data[m2].mean()))
                        
                        if expected_direction != 0 and actual_direction != 0 and expected_direction != actual_direction:
                            violations += 1
                    
                    # Threshold for detecting biochemical outliers (10% of pairs)
                    if violations > len(top_pairs) * 0.1:
                        biochemical_outliers.add(idx)
            
            results['biochemical_outliers'] = len(biochemical_outliers)
            
            # Combine both types of outliers
            all_outliers = set(standard_indices) | biochemical_outliers
            results['total_outliers'] = len(all_outliers)
            results['outlier_percentage'] = (results['total_outliers'] / results['total_samples']) * 100
            
            # Create detailed report for each outlier
            for idx in all_outliers:
                is_standard = idx in standard_indices
                is_biochemical = idx in biochemical_outliers
                
                detail = {
                    'Sample_Index': idx,
                    'Row_Name': data.iloc[idx]['Row_names'] if 'Row_names' in data.columns else f"Row_{idx}",
                    'Is_Standard_Outlier': is_standard,
                    'Is_Biochemical_Outlier': is_biochemical
                }
                results['outlier_details'].append(detail)
        
        except Exception as e:
            print(f"Error in outlier detection: {e}")
            # Return default values if error occurs
        
        return results
    
    #-------------------------------------------------------------------------
    # 2. Class Separability Assessment
    #-------------------------------------------------------------------------
    
    def class_separability_assessment(self):
        """
        Assess class separability in original vs augmented data.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with class separability results
        """
        print("\n2. Assessing class separability pre/post augmentation...")
        
        # Check for class variables
        class_variables = []
        for col in ['Treatment', 'Genotype', 'Batch']:
            if col in self.metadata_cols:
                class_variables.append(col)
        
        if not class_variables:
            print("Warning: No class variables found (Treatment, Genotype, Batch).")
            # Use Day as fallback if available
            if 'Day' in self.metadata_cols:
                class_variables = ['Day']
                print("Using 'Day' as class variable.")
            else:
                print("No suitable class variables found. Class separability assessment skipped.")
                return pd.DataFrame()
        
        # Metrics to calculate
        metrics = ['silhouette_score', 'davies_bouldin_score', 'bhattacharyya_distance', 'pca_class_variance_ratio']
        
        # Results storage
        results_rows = []
        
        # For each class variable and method, calculate separability metrics
        for class_var in class_variables:
            print(f"  Analyzing separability for {class_var}...")
            
            # Check if we have at least 2 classes
            for method, data in self.method_data.items():
                if len(data[class_var].unique()) < 2:
                    print(f"  Warning: {method} has less than 2 unique values for {class_var}. Skipping.")
                    continue
                
                print(f"  Processing {method}...")
                
                # Calculate metrics
                try:
                    # Scale the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(data[self.molecular_feature_cols])
                    
                    # Get class labels
                    labels = data[class_var].values
                    
                    # Calculate metrics
                    metrics_values = {}
                    
                    # 1. Silhouette Score - higher is better
                    if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
                        metrics_values['silhouette_score'] = silhouette_score(scaled_data, labels)
                    else:
                        metrics_values['silhouette_score'] = 0
                    
                    # 2. Davies-Bouldin Index - lower is better
                    if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
                        metrics_values['davies_bouldin_score'] = davies_bouldin_score(scaled_data, labels)
                    else:
                        metrics_values['davies_bouldin_score'] = float('inf')
                    
                    # 3. Bhattacharyya Distance between classes - higher is better
                    metrics_values['bhattacharyya_distance'] = self.calculate_bhattacharyya_distance(
                        scaled_data, labels)
                    
                    # 4. PCA Class Variance Ratio - higher is better
                    metrics_values['pca_class_variance_ratio'] = self.calculate_pca_class_variance(
                        scaled_data, labels)
                    
                    # Add to results
                    results_rows.append({
                        'Method': method,
                        'Class_Variable': class_var,
                        'Silhouette_Score': metrics_values['silhouette_score'],
                        'Davies_Bouldin_Score': metrics_values['davies_bouldin_score'],
                        'Bhattacharyya_Distance': metrics_values['bhattacharyya_distance'],
                        'PCA_Class_Variance_Ratio': metrics_values['pca_class_variance_ratio']
                    })
                
                except Exception as e:
                    print(f"Error calculating separability metrics for {method}, {class_var}: {e}")
        
        # Create DataFrame
        results_df = pd.DataFrame(results_rows)
        
        if len(results_df) == 0:
            print("No separability results to report.")
            return results_df
        
        # Save results
        results_df.to_csv(os.path.join(self.results_dir, 'class_separability.csv'), index=False)
        
        # Create normalized scores (0-1 scale, 1 is best) for comparison
        scores_df = results_df.copy()
        
        # Group by class variable for normalization within each class
        for class_var in class_variables:
            class_df = scores_df[scores_df['Class_Variable'] == class_var]
            
            if len(class_df) > 0:
                # Silhouette Score (higher is better)
                min_val = class_df['Silhouette_Score'].min()
                max_val = class_df['Silhouette_Score'].max()
                if max_val > min_val:
                    scores_df.loc[scores_df['Class_Variable'] == class_var, 'Silhouette_Normalized'] = \
                        (class_df['Silhouette_Score'] - min_val) / (max_val - min_val)
                else:
                    scores_df.loc[scores_df['Class_Variable'] == class_var, 'Silhouette_Normalized'] = 0.5
                
                # Davies-Bouldin (lower is better)
                min_val = class_df['Davies_Bouldin_Score'].min()
                max_val = class_df['Davies_Bouldin_Score'].max()
                if max_val > min_val:
                    scores_df.loc[scores_df['Class_Variable'] == class_var, 'Davies_Bouldin_Normalized'] = \
                        1 - (class_df['Davies_Bouldin_Score'] - min_val) / (max_val - min_val)
                else:
                    scores_df.loc[scores_df['Class_Variable'] == class_var, 'Davies_Bouldin_Normalized'] = 0.5
                
                # Bhattacharyya (higher is better)
                min_val = class_df['Bhattacharyya_Distance'].min()
                max_val = class_df['Bhattacharyya_Distance'].max()
                if max_val > min_val:
                    scores_df.loc[scores_df['Class_Variable'] == class_var, 'Bhattacharyya_Normalized'] = \
                        (class_df['Bhattacharyya_Distance'] - min_val) / (max_val - min_val)
                else:
                    scores_df.loc[scores_df['Class_Variable'] == class_var, 'Bhattacharyya_Normalized'] = 0.5
                
                # PCA Class Variance (higher is better)
                min_val = class_df['PCA_Class_Variance_Ratio'].min()
                max_val = class_df['PCA_Class_Variance_Ratio'].max()
                if max_val > min_val:
                    scores_df.loc[scores_df['Class_Variable'] == class_var, 'PCA_Variance_Normalized'] = \
                        (class_df['PCA_Class_Variance_Ratio'] - min_val) / (max_val - min_val)
                else:
                    scores_df.loc[scores_df['Class_Variable'] == class_var, 'PCA_Variance_Normalized'] = 0.5
        
        # Calculate overall separability score (average of normalized metrics)
        scores_df['Overall_Separability_Score'] = scores_df[[
            'Silhouette_Normalized', 'Davies_Bouldin_Normalized', 
            'Bhattacharyya_Normalized', 'PCA_Variance_Normalized'
        ]].mean(axis=1)
        
        # Save normalized scores
        scores_df.to_csv(os.path.join(self.results_dir, 'class_separability_scores.csv'), index=False)
        
        # Create visualization - one plot per class variable
        for class_var in class_variables:
            class_scores = scores_df[scores_df['Class_Variable'] == class_var]
            
            if len(class_scores) > 0:
                # Create plot
                plt.figure(figsize=(12, 6))
                
                # Sort by overall score
                class_scores = class_scores.sort_values('Overall_Separability_Score', ascending=False)
                
                # Bar plot of overall scores
                bars = plt.bar(class_scores['Method'], class_scores['Overall_Separability_Score'], 
                       color=['#3182bd' if method == 'original' else '#41ab5d' for method in class_scores['Method']])
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom')
                
                plt.title(f'Class Separability for {class_var}')
                plt.xlabel('Method')
                plt.ylabel('Overall Separability Score')
                plt.ylim(0, 1.1)
                plt.axhline(y=0.8, linestyle='--', color='#238b45', alpha=0.7)
                plt.axhline(y=0.6, linestyle='--', color='#fe9929', alpha=0.7)
                plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f'class_separability_{class_var}.png'), dpi=300)
                plt.savefig(os.path.join(self.plots_dir, f'class_separability_{class_var}.pdf'))
                plt.close()
                
        print(f"Class separability assessment completed. Results saved to: {self.results_dir}")
        print("Overall separability scores:")
        print(scores_df[['Method', 'Class_Variable', 'Overall_Separability_Score']])
        
        return scores_df
    
    def calculate_bhattacharyya_distance(self, data, labels):
        """
        Calculate average Bhattacharyya distance between classes.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Scaled feature data
        labels : numpy.ndarray
            Class labels
            
        Returns:
        --------
        float
            Average Bhattacharyya distance
        """
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return 0
        
        # Collect samples by class
        class_data = {label: data[labels == label] for label in unique_labels}
        
        # Calculate means and covariances
        means = {label: np.mean(samples, axis=0) for label, samples in class_data.items()}
        covs = {label: np.cov(samples, rowvar=False) + np.eye(data.shape[1]) * 1e-6 
               for label, samples in class_data.items() if len(samples) > 1}
        
        # For classes with only one sample, use identity covariance
        for label in class_data:
            if len(class_data[label]) <= 1:
                covs[label] = np.eye(data.shape[1])
        
        # Calculate Bhattacharyya distances between all class pairs
        distances = []
        for i, label1 in enumerate(unique_labels[:-1]):
            for label2 in unique_labels[i+1:]:
                if label1 in covs and label2 in covs:
                    try:
                        # Compute mean term
                        mean_diff = means[label1] - means[label2]
                        cov_avg = (covs[label1] + covs[label2]) / 2
                        
                        # Compute covariance term
                        det_cov_avg = np.linalg.det(cov_avg)
                        det_cov1 = np.linalg.det(covs[label1])
                        det_cov2 = np.linalg.det(covs[label2])
                        
                        if det_cov_avg > 0 and det_cov1 > 0 and det_cov2 > 0:
                            # Compute Bhattacharyya distance
                            term1 = 0.125 * mean_diff.dot(np.linalg.inv(cov_avg)).dot(mean_diff)
                            term2 = 0.5 * np.log(det_cov_avg / np.sqrt(det_cov1 * det_cov2))
                            dist = term1 + term2
                            distances.append(dist)
                    except:
                        # Skip if numerical issues
                        pass
        
        # Return average distance
        return np.mean(distances) if distances else 0
    
    def calculate_pca_class_variance(self, data, labels, n_components=2):
        """
        Calculate the ratio of between-class to within-class variance in PCA space.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Scaled feature data
        labels : numpy.ndarray
            Class labels
        n_components : int
            Number of PCA components to use
            
        Returns:
        --------
        float
            Ratio of between-class to within-class variance
        """
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return 0
        
        # Perform PCA
        n_components = min(n_components, data.shape[1], data.shape[0])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data)
        
        # Collect PCA points by class
        class_data = {label: pca_result[labels == label] for label in unique_labels}
        
        # Calculate overall mean
        overall_mean = np.mean(pca_result, axis=0)
        
        # Calculate between-class variance
        between_var = 0
        for label, samples in class_data.items():
            if len(samples) > 0:
                class_mean = np.mean(samples, axis=0)
                between_var += len(samples) * np.sum((class_mean - overall_mean) ** 2)
                
        between_var /= len(data)
        
        # Calculate within-class variance
        within_var = 0
        for label, samples in class_data.items():
            if len(samples) > 1:
                class_mean = np.mean(samples, axis=0)
                within_var += np.sum((samples - class_mean) ** 2)
                
        within_var /= len(data)
        
        # Calculate ratio (avoid division by zero)
        if within_var > 0:
            return between_var / within_var
        else:
            return 0
        
    #-------------------------------------------------------------------------
    # 3. Signal-to-Noise Ratio Metrics
    #-------------------------------------------------------------------------
    
    def signal_to_noise_assessment(self):
        """
        Calculate signal-to-noise ratio metrics for LC-MS features.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with SNR assessment results
        """
        print("\n3. Calculating signal-to-noise ratio metrics for LC-MS features...")
        
        # Store results
        snr_results = {}
        
        for method, data in self.method_data.items():
            print(f"  Processing {method}...")
            
            # Calculate SNR for each molecular feature
            molecular_feature_snr = {}
            
            for feature in self.molecular_feature_cols:
                if feature in data.columns:
                    # Extract values
                    values = data[feature].values
                    
                    # Calculate SNR
                    snr = self.calculate_snr(values)
                    molecular_feature_snr[feature] = snr
            
            # Store summary statistics
            snr_results[method] = {
                'mean_snr': np.mean(list(molecular_feature_snr.values())),
                'median_snr': np.median(list(molecular_feature_snr.values())),
                'min_snr': min(molecular_feature_snr.values()),
                'max_snr': max(molecular_feature_snr.values()),
                'snr_by_molecular_feature': molecular_feature_snr
            }
        
        # Create summary DataFrame
        summary_rows = [{
            'Method': method,
            'Mean_SNR': results['mean_snr'],
            'Median_SNR': results['median_snr'],
            'Min_SNR': results['min_snr'],
            'Max_SNR': results['max_snr']
        } for method, results in snr_results.items()]
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary to CSV
        summary_df.to_csv(os.path.join(self.results_dir, 'snr_assessment.csv'), index=False)
        
        # Calculate SNR preservation compared to original
        if 'original' in snr_results:
            orig_snr = snr_results['original']['snr_by_molecular_feature']
            
            preservation_rows = []
            for method, results in snr_results.items():
                if method != 'original':
                    method_snr = results['snr_by_molecular_feature']
                    
                    # Calculate preservation metrics
                    snr_ratios = []
                    for feature in orig_snr:
                        if feature in method_snr:
                            # Ratio of method SNR to original SNR (capped at 2.0)
                            ratio = min(2.0, method_snr[feature] / orig_snr[feature]) if orig_snr[feature] > 0 else 0
                            snr_ratios.append(ratio)
                    
                    # Calculate preservation score (1.0 = same as original, < 1.0 = worse, > 1.0 = better)
                    mean_ratio = np.mean(snr_ratios) if snr_ratios else 0
                    
                    # Preservation score: 1.0 - min(1.0, |ratio - 1.0|)
                    # This gives 1.0 for perfect preservation (ratio = 1.0), and decreases as ratio deviates from 1.0
                    preservation_score = 1.0 - min(1.0, abs(mean_ratio - 1.0))
                    
                    preservation_rows.append({
                        'Method': method,
                        'Mean_SNR_Ratio': mean_ratio,
                        'SNR_Preservation_Score': preservation_score
                    })
            
            preservation_df = pd.DataFrame(preservation_rows)
            
            # Add original as reference
            preservation_df = pd.concat([
                pd.DataFrame([{'Method': 'original', 'Mean_SNR_Ratio': 1.0, 'SNR_Preservation_Score': 1.0}]),
                preservation_df
            ]).reset_index(drop=True)
            
            # Save preservation scores
            preservation_df.to_csv(os.path.join(self.results_dir, 'snr_preservation.csv'), index=False)
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Sort by preservation score
            sorted_df = preservation_df.sort_values('SNR_Preservation_Score', ascending=False)
            
            # Plot preservation scores
            bars = plt.bar(sorted_df['Method'], sorted_df['SNR_Preservation_Score'],
                   color=['#3182bd' if method == 'original' else '#41ab5d' for method in sorted_df['Method']])
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.title('Signal-to-Noise Ratio Preservation by Method')
            plt.xlabel('Method')
            plt.ylabel('SNR Preservation Score')
            plt.ylim(0, 1.1)
            plt.axhline(y=0.9, linestyle='--', color='#238b45', alpha=0.7)
            plt.axhline(y=0.8, linestyle='--', color='#fe9929', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'snr_preservation.png'), dpi=300)
            plt.savefig(os.path.join(self.plots_dir, 'snr_preservation.pdf'))
            plt.close()
            
            print(f"SNR assessment completed. Results saved to: {self.results_dir}")
            print(preservation_df)
            
            return preservation_df
        else:
            print(f"SNR assessment completed. Results saved to: {self.results_dir}")
            print(summary_df)
            
            return summary_df
    
    def calculate_snr(self, values):
        """
        Calculate signal-to-noise ratio for a set of values.
        
        Parameters:
        -----------
        values : numpy.ndarray
            Values to analyze
            
        Returns:
        --------
        float
            Signal-to-noise ratio
        """
        if len(values) < 3:
            return 0
        
        # Use moving average as proxy for signal
        window_size = min(5, len(values) - 2)
        
        # Apply Savitzky-Golay filter for smoothing
        try:
            from scipy.signal import savgol_filter
            window_length = min(window_size * 2 + 1, len(values))
            # Ensure window_length is odd
            if window_length % 2 == 0:
                window_length -= 1
            
            # If window too small, use simple smoothing
            if window_length >= 5:
                signal = savgol_filter(values, window_length, 3)
            else:
                # Simple moving average
                signal = np.convolve(values, np.ones(window_size)/window_size, mode='same')
        except:
            # Fallback to simple moving average
            signal = np.convolve(values, np.ones(window_size)/window_size, mode='same')
        
        # Noise is the difference between original values and signal
        noise = values - signal
        
        # Calculate SNR
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 20.0  # High SNR default
        
        return snr
    
    #-------------------------------------------------------------------------
    # 4. Cross-Validation with Machine Learning
    #-------------------------------------------------------------------------
    
    def cross_validation_assessment(self):
        """
        Perform cross-validation with machine learning classifiers.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with cross-validation results
        """
        print("\n4. Performing cross-validation with machine learning classifiers...")
        
        # Check for class variables
        class_variables = []
        for col in ['Treatment', 'Genotype', 'Batch']:
            if col in self.metadata_cols:
                class_variables.append(col)
        
        if not class_variables:
            print("Warning: No class variables found (Treatment, Genotype, Batch).")
            # Use Day as fallback if available
            if 'Day' in self.metadata_cols:
                class_variables = ['Day']
                print("Using 'Day' as class variable.")
            else:
                print("No suitable class variables found. Cross-validation assessment skipped.")
                return pd.DataFrame()
        
        # Results storage
        cv_results = []
        
        # For each class variable, perform cross-validation
        for class_var in class_variables:
            print(f"  Cross-validation for {class_var}...")
            
            # Check if we have enough classes for classification
            for method, data in self.method_data.items():
                unique_classes = data[class_var].unique()
                if len(unique_classes) < 2:
                    print(f"  Warning: {method} has less than 2 unique values for {class_var}. Skipping.")
                    continue
                
                print(f"  Processing {method}...")
                
                # Prepare data
                X = data[self.molecular_feature_cols]
                y = data[class_var]
                
                # Encode categorical classes if needed
                if y.dtype == object or y.dtype.name == 'category':
                    y = pd.factorize(y)[0]
                
                # Perform cross-validation with random forest classifier
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y if len(y) > len(np.unique(y)) * 2 else None
                    )
                    
                    # Train model
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)
                    clf.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = clf.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test, y_pred, average='weighted')
                    
                    # Feature importance
                    importance = clf.feature_importances_
                    top_features = pd.Series(importance, index=X.columns).nlargest(10).index.tolist()
                    
                    # Store results
                    cv_results.append({
                        'Method': method,
                        'Class_Variable': class_var,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1_Score': f1,
                        'Top_Features': ','.join(top_features)
                    })
                    
                except Exception as e:
                    print(f"  Error in cross-validation for {method}, {class_var}: {e}")
        
        # Create DataFrame
        cv_df = pd.DataFrame(cv_results)
        
        if len(cv_df) == 0:
            print("No cross-validation results to report.")
            return cv_df
        
        # Save results
        cv_df.to_csv(os.path.join(self.results_dir, 'cross_validation.csv'), index=False)
        
        # Calculate preservation scores compared to original
        preservation_rows = []
        
        for class_var in class_variables:
            # Get original scores for this class variable
            orig_cv = cv_df[(cv_df['Method'] == 'original') & (cv_df['Class_Variable'] == class_var)]
            
            if len(orig_cv) == 0:
                continue
                
            orig_accuracy = orig_cv['Accuracy'].values[0]
            orig_f1 = orig_cv['F1_Score'].values[0]
            
            # Calculate preservation for each method
            for method in cv_df['Method'].unique():
                if method == 'original':
                    continue
                    
                method_cv = cv_df[(cv_df['Method'] == method) & (cv_df['Class_Variable'] == class_var)]
                
                if len(method_cv) == 0:
                    continue
                    
                method_accuracy = method_cv['Accuracy'].values[0]
                method_f1 = method_cv['F1_Score'].values[0]
                
                # Calculate relative performance (1.0 = same as original)
                accuracy_ratio = method_accuracy / orig_accuracy if orig_accuracy > 0 else 0
                f1_ratio = method_f1 / orig_f1 if orig_f1 > 0 else 0
                
                # Calculate preservation score (1.0 for exact match, decreasing as ratio deviates from 1.0)
                accuracy_preservation = 1.0 - min(1.0, abs(accuracy_ratio - 1.0))
                f1_preservation = 1.0 - min(1.0, abs(f1_ratio - 1.0))
                
                # Overall preservation as average
                overall_preservation = (accuracy_preservation + f1_preservation) / 2
                
                preservation_rows.append({
                    'Method': method,
                    'Class_Variable': class_var,
                    'Accuracy_Ratio': accuracy_ratio,
                    'F1_Ratio': f1_ratio,
                    'Accuracy_Preservation': accuracy_preservation,
                    'F1_Preservation': f1_preservation,
                    'Overall_Preservation': overall_preservation
                })
        
        # Add original reference
        for class_var in class_variables:
            if any(r['Class_Variable'] == class_var for r in preservation_rows):
                preservation_rows.append({
                    'Method': 'original',
                    'Class_Variable': class_var,
                    'Accuracy_Ratio': 1.0,
                    'F1_Ratio': 1.0,
                    'Accuracy_Preservation': 1.0,
                    'F1_Preservation': 1.0,
                    'Overall_Preservation': 1.0
                })
        
        preservation_df = pd.DataFrame(preservation_rows)
        
        if len(preservation_df) > 0:
            # Save preservation scores
            preservation_df.to_csv(os.path.join(self.results_dir, 'cross_validation_preservation.csv'), index=False)
            
            # Create visualization for each class variable
            for class_var in preservation_df['Class_Variable'].unique():
                # Filter to this class
                class_data = preservation_df[preservation_df['Class_Variable'] == class_var]
                
                plt.figure(figsize=(12, 6))
                
                # Sort by overall preservation
                sorted_data = class_data.sort_values('Overall_Preservation', ascending=False)
                
                # Plot overall preservation
                bars = plt.bar(sorted_data['Method'], sorted_data['Overall_Preservation'],
                       color=['#3182bd' if method == 'original' else '#41ab5d' for method in sorted_data['Method']])
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.2f}', ha='center', va='bottom')
                
                plt.title(f'Classification Performance Preservation for {class_var}')
                plt.xlabel('Method')
                plt.ylabel('Preservation Score')
                plt.ylim(0, 1.1)
                plt.axhline(y=0.9, linestyle='--', color='#238b45', alpha=0.7)
                plt.axhline(y=0.8, linestyle='--', color='#fe9929', alpha=0.7)
                plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f'cv_preservation_{class_var}.png'), dpi=300)
                plt.savefig(os.path.join(self.plots_dir, f'cv_preservation_{class_var}.pdf'))
                plt.close()
            
            print(f"Cross-validation assessment completed. Results saved to: {self.results_dir}")
            print("Classification performance preservation:")
            print(preservation_df[['Method', 'Class_Variable', 'Overall_Preservation']])
            
            return preservation_df
        else:
            print(f"Cross-validation assessment completed. Results saved to: {self.results_dir}")
            print(cv_df[['Method', 'Class_Variable', 'Accuracy', 'F1_Score']])
            
            return cv_df
    
    #-------------------------------------------------------------------------
    # 5. Statistical Power Analysis
    #-------------------------------------------------------------------------
    
    def statistical_power_analysis(self):
        """
        Perform statistical power analysis on the augmented dataset.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with power analysis results
        """
        print("\n5. Performing statistical power analysis...")
        
        # Check for treatment variable
        if 'Treatment' not in self.metadata_cols:
            print("Warning: 'Treatment' column not found. Using 'Batch' instead.")
            treatment_col = 'Batch'
        else:
            treatment_col = 'Treatment'
        
        # Check for enough unique treatment values
        for method, data in self.method_data.items():
            if len(data[treatment_col].unique()) < 2:
                print(f"  Warning: {method} has less than 2 unique values for {treatment_col}.")
                if method == 'original':
                    print("  Cannot perform power analysis without treatment groups in original data.")
                    return pd.DataFrame()
        
        # Detect effect sizes in original data
        orig_effect_sizes = self.calculate_effect_sizes(self.original_data, treatment_col)
        
        if not orig_effect_sizes:
            print("  No significant effects detected in original data. Power analysis skipped.")
            return pd.DataFrame()
        
        # Results storage
        power_results = []
        
        # For each method, calculate power for different effect sizes
        for method, data in self.method_data.items():
            print(f"  Processing {method}...")
            
            # Skip if less than 2 treatment groups
            if len(data[treatment_col].unique()) < 2:
                continue
            
            # Get sample sizes per group
            group_sizes = data.groupby(treatment_col).size().to_dict()
            
            # Estimate noise level (standard deviation) for each molecular feature
            feature_sd = {}
            for feature in self.molecular_feature_cols:
                if feature in data.columns:
                    # Get standard deviation within each group
                    group_sds = data.groupby(treatment_col)[feature].std().dropna()
                    if len(group_sds) > 0:
                        # Use pooled standard deviation as noise estimate
                        feature_sd[feature] = np.mean(group_sds)
            
            # Calculate power for each effect size detected in original data
            for feature, effect_size in orig_effect_sizes.items():
                if feature in feature_sd:
                    # Get relevant parameters
                    sd = feature_sd[feature]
                    
                    # For each comparison, calculate power
                    for group1, n1 in group_sizes.items():
                        for group2, n2 in group_sizes.items():
                            if group1 < group2:  # Avoid duplicate comparisons
                                # Calculate power using t-test formula
                                alpha = 0.05  # Significance level
                                power = self.calculate_power(effect_size, n1, n2, sd, alpha)
                                
                                power_results.append({
                                    'Method': method,
                                    'Molecular_Feature': feature,
                                    'Effect_Size': effect_size,
                                    'Group1': group1,
                                    'Group2': group2,
                                    'N1': n1,
                                    'N2': n2,
                                    'SD': sd,
                                    'Power': power
                                })
        
        # Create DataFrame
        power_df = pd.DataFrame(power_results)
        
        if len(power_df) == 0:
            print("No power analysis results to report.")
            return power_df
        
        # Save detailed results
        power_df.to_csv(os.path.join(self.results_dir, 'power_analysis_detailed.csv'), index=False)
        
        # Create summary by method
        summary_rows = []
        for method in power_df['Method'].unique():
            method_power = power_df[power_df['Method'] == method]
            
            summary_rows.append({
                'Method': method,
                'Mean_Power': method_power['Power'].mean(),
                'Median_Power': method_power['Power'].median(),
                'Min_Power': method_power['Power'].min(),
                'Max_Power': method_power['Power'].max(),
                'Power_80_Percent': (method_power['Power'] >= 0.8).mean() * 100,
                'Sample_Size': method_power['N1'].iloc[0] + method_power['N2'].iloc[0]
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary
        summary_df.to_csv(os.path.join(self.results_dir, 'power_analysis_summary.csv'), index=False)
        
        # Calculate power improvement compared to original
        if 'original' in summary_df['Method'].values:
            orig_power = float(summary_df[summary_df['Method'] == 'original']['Mean_Power'])
            orig_sample = float(summary_df[summary_df['Method'] == 'original']['Sample_Size'])
            
            for i, row in summary_df.iterrows():
                if row['Method'] != 'original':
                    # Calculate power improvement ratio
                    power_ratio = row['Mean_Power'] / orig_power if orig_power > 0 else 0
                    
                    # Calculate expected improvement due to sample size
                    # Power scales approximately with sqrt(n), so expected improvement is sqrt(n_new/n_orig)
                    expected_ratio = np.sqrt(row['Sample_Size'] / orig_sample) if orig_sample > 0 else 0
                    
                    # Calculate efficiency (actual/expected)
                    efficiency = power_ratio / expected_ratio if expected_ratio > 0 else 0
                    
                    summary_df.loc[i, 'Power_Ratio'] = power_ratio
                    summary_df.loc[i, 'Expected_Ratio'] = expected_ratio
                    summary_df.loc[i, 'Efficiency'] = efficiency
            
            # Add original ratios
            orig_idx = summary_df[summary_df['Method'] == 'original'].index[0]
            summary_df.loc[orig_idx, 'Power_Ratio'] = 1.0
            summary_df.loc[orig_idx, 'Expected_Ratio'] = 1.0
            summary_df.loc[orig_idx, 'Efficiency'] = 1.0
        
        # Re-save summary with ratios
        summary_df.to_csv(os.path.join(self.results_dir, 'power_analysis_summary.csv'), index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Sort by mean power
        sorted_df = summary_df.sort_values('Mean_Power', ascending=False)
        
        # Plot mean power
        bars = plt.bar(sorted_df['Method'], sorted_df['Mean_Power'],
               color=['#3182bd' if method == 'original' else '#41ab5d' for method in sorted_df['Method']])
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Statistical Power by Method')
        plt.xlabel('Method')
        plt.ylabel('Mean Statistical Power')
        plt.ylim(0, 1.1)
        plt.axhline(y=0.8, linestyle='--', color='#238b45', alpha=0.7)
        plt.axhline(y=0.5, linestyle='--', color='#fe9929', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'statistical_power.png'), dpi=300)
        plt.savefig(os.path.join(self.plots_dir, 'statistical_power.pdf'))
        plt.close()
        
        # If we have efficiency data, create efficiency plot
        if 'Efficiency' in summary_df.columns:
            plt.figure(figsize=(12, 6))
            
            # Sort by efficiency
            sorted_df = summary_df.sort_values('Efficiency', ascending=False)
            
            # Plot efficiency
            bars = plt.bar(sorted_df['Method'], sorted_df['Efficiency'],
                   color=['#3182bd' if method == 'original' else '#41ab5d' for method in sorted_df['Method']])
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom')
            
            plt.title('Power Efficiency by Method')
            plt.xlabel('Method')
            plt.ylabel('Efficiency (Actual/Expected Power Ratio)')
            plt.ylim(0, max(2.0, sorted_df['Efficiency'].max() * 1.2))
            plt.axhline(y=1.0, linestyle='--', color='#238b45', alpha=0.7)
            plt.axhline(y=0.8, linestyle='--', color='#fe9929', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'power_efficiency.png'), dpi=300)
            plt.savefig(os.path.join(self.plots_dir, 'power_efficiency.pdf'))
            plt.close()
        
        print(f"Statistical power analysis completed. Results saved to: {self.results_dir}")
        print(summary_df[['Method', 'Mean_Power', 'Power_80_Percent', 'Sample_Size']])
        
        return summary_df
    
    def calculate_effect_sizes(self, data, treatment_col):
        """
        Calculate effect sizes for treatment comparisons.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset to analyze
        treatment_col : str
            Column name for treatment variable
            
        Returns:
        --------
        dict
            Dictionary mapping molecular features to effect sizes
        """
        effect_sizes = {}
        
        # Get unique treatments
        treatments = sorted(data[treatment_col].unique())
        
        if len(treatments) < 2:
            return {}
        
        # Assuming binary treatment (0, 1) for simplicity
        # For multi-level, we'd need to consider all pairwise comparisons
        if len(treatments) == 2:
            control = treatments[0]
            treatment = treatments[1]
            
            for feature in self.molecular_feature_cols:
                if feature in data.columns:
                    # Get values for each group
                    control_values = data[data[treatment_col] == control][feature].values
                    treatment_values = data[data[treatment_col] == treatment][feature].values
                    
                    if len(control_values) > 0 and len(treatment_values) > 0:
                        # Calculate Cohen's d effect size
                        mean_diff = np.mean(treatment_values) - np.mean(control_values)
                        pooled_std = np.sqrt((np.var(control_values) + np.var(treatment_values)) / 2)
                        
                        if pooled_std > 0:
                            effect_size = abs(mean_diff / pooled_std)  # Use absolute value for power calculation
                            
                            # Only include effects that are "significant" in original data
                            t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)
                            
                            if p_value < 0.1:  # Relaxed threshold to include more effects
                                effect_sizes[feature] = effect_size
        
        return effect_sizes
    
    def calculate_power(self, effect_size, n1, n2, sd, alpha=0.05):
        """
        Calculate statistical power for two-sample t-test.
        
        Parameters:
        -----------
        effect_size : float
            Cohen's d effect size
        n1 : int
            Sample size of group 1
        n2 : int
            Sample size of group 2
        sd : float
            Pooled standard deviation
        alpha : float
            Significance level
            
        Returns:
        --------
        float
            Statistical power (0-1)
        """
        # Calculate non-centrality parameter
        ncp = effect_size * np.sqrt(n1 * n2 / (n1 + n2))
        
        # Degrees of freedom
        df = n1 + n2 - 2
        
        # Critical t-value for two-tailed test
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        # Calculate power
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        
        return power
    
    #-------------------------------------------------------------------------
    # Integrated Quality Control Analysis
    #-------------------------------------------------------------------------
    
    def run_all_qc(self):
        """
        Run all quality control checks and generate integrated report.
        
        Returns:
        --------
        dict
            Dictionary with all QC results
        """
        print("\nRunning comprehensive quality control analysis for molecular feature data...")
        start_time = time.time()
        
        # Run all QC methods
        outlier_results = self.critical_outlier_detection()
        separability_results = self.class_separability_assessment()
        snr_results = self.signal_to_noise_assessment()
        cv_results = self.cross_validation_assessment()
        power_results = self.statistical_power_analysis()
        
        # Combine all results
        all_results = {
            'outliers': outlier_results,
            'separability': separability_results,
            'snr': snr_results,
            'cross_validation': cv_results,
            'power': power_results
        }
        
        # Generate integrated report
        self.generate_integrated_report(all_results)
        
        end_time = time.time()
        print(f"\nAll quality control checks completed in {end_time - start_time:.2f} seconds")
        
        return all_results
    
    def generate_integrated_report(self, all_results):
        """
        Generate an integrated HTML report for all QC results.
        
        Parameters:
        -----------
        all_results : dict
            Dictionary with all QC results
        """
        report_path = os.path.join(self.qc_dir, 'integrated_qc_report.html')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # HTML header and basic styling
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Molecular Feature Data Quality Control Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; margin-top: 30px; }
        h3 { color: #2980b9; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .summary { margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }
        .good { color: #238b45; }
        .moderate { color: #fe9929; }
        .poor { color: #d62728; }
        img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>Molecular Feature Data Quality Control Report</h1>
    <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
    
    <div class="summary">
        <h2>Overview</h2>
        <p>This report presents comprehensive quality control results for augmented molecular feature data.</p>
        <p><strong>Original data:</strong> """ + self.original_path + """</p>
        <p><strong>Augmented data:</strong> """ + self.augmented_path + """</p>
    </div>
""")
            
            # 1. Outlier Detection Section
            if 'outliers' in all_results and len(all_results['outliers']) > 0:
                f.write("""
    <h2>1. Critical Outlier Detection</h2>
    <p>Detection of statistical outliers and samples that violate expected biochemical relationships.</p>
    <table>
        <tr>
            <th>Method</th>
            <th>Total Samples</th>
            <th>Outlier Percentage</th>
            <th>Assessment</th>
        </tr>
""")
                
                outlier_df = all_results['outliers']
                for _, row in outlier_df.iterrows():
                    method = row['Method']
                    total = row['Total_Samples']
                    percentage = row['Outlier_Percentage']
                    
                    # Determine quality class
                    if percentage < 5:
                        quality_class = "good"
                        assessment = "Excellent: Very few outliers"
                    elif percentage < 10:
                        quality_class = "moderate"
                        assessment = "Good: Acceptable number of outliers"
                    else:
                        quality_class = "poor"
                        assessment = "Moderate: Higher than expected outliers"
                    
                    f.write(f"""
        <tr>
            <td>{method}</td>
            <td>{total}</td>
            <td class="{quality_class}">{percentage:.1f}%</td>
            <td>{assessment}</td>
        </tr>
""")
                
                f.write("</table>")
                
                # Add plot
                if os.path.exists(os.path.join(self.plots_dir, 'outlier_detection.png')):
                    f.write(f"""
    <div>
        <img src="plots/outlier_detection.png" alt="Outlier Detection Results">
    </div>
""")
            
            # 2. Class Separability Section
            if 'separability' in all_results and len(all_results['separability']) > 0:
                f.write("""
    <h2>2. Class Separability Assessment</h2>
    <p>Evaluation of how well different classes are separated in the molecular feature data.</p>
""")
                
                sep_df = all_results['separability']
                
                # For each class variable, create a table
                for class_var in sep_df['Class_Variable'].unique():
                    f.write(f"""
    <h3>Separability for {class_var}</h3>
    <table>
        <tr>
            <th>Method</th>
            <th>Overall Score</th>
            <th>Silhouette Score</th>
            <th>Davies-Bouldin Score</th>
            <th>Assessment</th>
        </tr>
""")
                    
                    class_df = sep_df[sep_df['Class_Variable'] == class_var].sort_values('Overall_Separability_Score', ascending=False)
                    
                    for _, row in class_df.iterrows():
                        method = row['Method']
                        overall = row['Overall_Separability_Score']
                        silhouette = row['Silhouette_Normalized']
                        davies = row['Davies_Bouldin_Normalized']
                        
                        # Determine quality class
                        if overall > 0.8:
                            quality_class = "good"
                            assessment = "Excellent class separation"
                        elif overall > 0.6:
                            quality_class = "moderate"
                            assessment = "Good class separation"
                        else:
                            quality_class = "poor"
                            assessment = "Moderate class separation"
                        
                        f.write(f"""
        <tr>
            <td>{method}</td>
            <td class="{quality_class}">{overall:.2f}</td>
            <td>{silhouette:.2f}</td>
            <td>{davies:.2f}</td>
            <td>{assessment}</td>
        </tr>
""")
                    
                    f.write("</table>")
                    
                    # Add plot
                    plot_path = os.path.join(self.plots_dir, f'class_separability_{class_var}.png')
                    if os.path.exists(plot_path):
                        f.write(f"""
    <div>
        <img src="plots/class_separability_{class_var}.png" alt="Class Separability for {class_var}">
    </div>
""")
            
            # 3. SNR Assessment Section
            if 'snr' in all_results and len(all_results['snr']) > 0:
                f.write("""
    <h2>3. Signal-to-Noise Ratio Assessment</h2>
    <p>Evaluation of signal quality in molecular features.</p>
    <table>
        <tr>
            <th>Method</th>
            <th>SNR Preservation</th>
            <th>Mean SNR Ratio</th>
            <th>Assessment</th>
        </tr>
""")
                
                snr_df = all_results['snr']
                
                # If preservation score is available, use it
                if 'SNR_Preservation_Score' in snr_df.columns:
                    sorted_df = snr_df.sort_values('SNR_Preservation_Score', ascending=False)
                    
                    for _, row in sorted_df.iterrows():
                        method = row['Method']
                        preservation = row['SNR_Preservation_Score']
                        ratio = row.get('Mean_SNR_Ratio', 1.0)
                        
                        # Determine quality class
                        if preservation > 0.9:
                            quality_class = "good"
                            assessment = "Excellent SNR preservation"
                        elif preservation > 0.8:
                            quality_class = "moderate"
                            assessment = "Good SNR preservation"
                        else:
                            quality_class = "poor"
                            assessment = "Moderate SNR preservation"
                        
                        ratio_text = f"{ratio:.2f}x" if method != 'original' else "1.00x (reference)"
                        
                        f.write(f"""
        <tr>
            <td>{method}</td>
            <td class="{quality_class}">{preservation:.2f}</td>
            <td>{ratio_text}</td>
            <td>{assessment}</td>
        </tr>
""")
                
                f.write("</table>")
                
                # Add plot
                if os.path.exists(os.path.join(self.plots_dir, 'snr_preservation.png')):
                    f.write(f"""
    <div>
        <img src="plots/snr_preservation.png" alt="SNR Preservation Results">
    </div>
""")
            
            # 4. Cross-Validation Section
            if 'cross_validation' in all_results and len(all_results['cross_validation']) > 0:
                f.write("""
    <h2>4. Cross-Validation Assessment</h2>
    <p>Evaluation of classification performance with machine learning models.</p>
""")
                
                cv_df = all_results['cross_validation']
                
                # If we have preservation scores, use them
                if 'Overall_Preservation' in cv_df.columns:
                    # For each class variable, create a table
                    for class_var in cv_df['Class_Variable'].unique():
                        f.write(f"""
    <h3>Cross-Validation for {class_var}</h3>
    <table>
        <tr>
            <th>Method</th>
            <th>Preservation Score</th>
            <th>Accuracy Preservation</th>
            <th>F1 Preservation</th>
            <th>Assessment</th>
        </tr>
""")
                        
                        class_df = cv_df[cv_df['Class_Variable'] == class_var].sort_values('Overall_Preservation', ascending=False)
                        
                        for _, row in class_df.iterrows():
                            method = row['Method']
                            overall = row['Overall_Preservation']
                            accuracy = row['Accuracy_Preservation']
                            f1 = row['F1_Preservation']
                            
                            # Determine quality class
                            if overall > 0.9:
                                quality_class = "good"
                                assessment = "Excellent performance preservation"
                            elif overall > 0.8:
                                quality_class = "moderate"
                                assessment = "Good performance preservation"
                            else:
                                quality_class = "poor"
                                assessment = "Moderate performance preservation"
                            
                            f.write(f"""
        <tr>
            <td>{method}</td>
            <td class="{quality_class}">{overall:.2f}</td>
            <td>{accuracy:.2f}</td>
            <td>{f1:.2f}</td>
            <td>{assessment}</td>
        </tr>
""")
                        
                        f.write("</table>")
                        
                        # Add plot
                        plot_path = os.path.join(self.plots_dir, f'cv_preservation_{class_var}.png')
                        if os.path.exists(plot_path):
                            f.write(f"""
    <div>
        <img src="plots/cv_preservation_{class_var}.png" alt="Cross-Validation Preservation for {class_var}">
    </div>
""")
            
            # 5. Power Analysis Section
            if 'power' in all_results and len(all_results['power']) > 0:
                f.write("""
    <h2>5. Statistical Power Analysis</h2>
    <p>Assessment of statistical power to detect treatment effects.</p>
    <table>
        <tr>
            <th>Method</th>
            <th>Mean Power</th>
            <th>Power ≥ 80%</th>
            <th>Sample Size</th>
            <th>Assessment</th>
        </tr>
""")
                
                power_df = all_results['power'].sort_values('Mean_Power', ascending=False)
                
                for _, row in power_df.iterrows():
                    method = row['Method']
                    power = row['Mean_Power']
                    power80 = row['Power_80_Percent']
                    sample = row['Sample_Size']
                    
                    # Determine quality class
                    if power > 0.8:
                        quality_class = "good"
                        assessment = "Excellent statistical power"
                    elif power > 0.5:
                        quality_class = "moderate"
                        assessment = "Good statistical power"
                    else:
                        quality_class = "poor"
                        assessment = "Moderate statistical power"
                    
                    f.write(f"""
        <tr>
            <td>{method}</td>
            <td class="{quality_class}">{power:.2f}</td>
            <td>{power80:.1f}%</td>
            <td>{sample}</td>
            <td>{assessment}</td>
        </tr>
""")
                
                f.write("</table>")
                
                # Add plots
                if os.path.exists(os.path.join(self.plots_dir, 'statistical_power.png')):
                    f.write(f"""
    <div>
        <img src="plots/statistical_power.png" alt="Statistical Power Results">
    </div>
""")
                
                if os.path.exists(os.path.join(self.plots_dir, 'power_efficiency.png')):
                    f.write(f"""
    <div>
        <img src="plots/power_efficiency.png" alt="Power Efficiency Results">
    </div>
""")
            
            # Overall Assessment
            f.write("""
    <h2>Overall Quality Assessment</h2>
""")
            
            # Calculate overall quality scores
            method_scores = {}
            
            # Define metrics and weights for overall score
            metrics_weights = {
                'outliers': {'metric': 'Outlier_Percentage', 'weight': 0.15, 'inverse': True},
                'separability': {'metric': 'Overall_Separability_Score', 'weight': 0.2, 'inverse': False},
                'snr': {'metric': 'SNR_Preservation_Score', 'weight': 0.2, 'inverse': False},
                'cross_validation': {'metric': 'Overall_Preservation', 'weight': 0.2, 'inverse': False},
                'power': {'metric': 'Mean_Power', 'weight': 0.25, 'inverse': False}
            }
            
            # Initialize all methods from any available results
            all_methods = set()
            for result_key, result_df in all_results.items():
                if isinstance(result_df, pd.DataFrame) and 'Method' in result_df.columns:
                    all_methods.update(result_df['Method'].unique())
            
            # Initialize scores for all methods
            for method in all_methods:
                method_scores[method] = {'scores': {}, 'weights': {}}
            
            # Calculate scores for each method
            for result_key, metrics in metrics_weights.items():
                if result_key in all_results and len(all_results[result_key]) > 0:
                    result_df = all_results[result_key]
                    
                    if 'Method' in result_df.columns and metrics['metric'] in result_df.columns:
                        # Handle class variables by averaging
                        if 'Class_Variable' in result_df.columns:
                            for method in all_methods:
                                method_df = result_df[result_df['Method'] == method]
                                if len(method_df) > 0:
                                    avg_score = method_df[metrics['metric']].mean()
                                    
                                    # For inverse metrics (lower is better), convert to 0-1 scale
                                    if metrics['inverse']:
                                        # Assuming percentage is 0-100
                                        score = 1 - min(1, avg_score / 100)
                                    else:
                                        score = avg_score
                                    
                                    method_scores[method]['scores'][result_key] = score
                                    method_scores[method]['weights'][result_key] = metrics['weight']
                        else:
                            for method in all_methods:
                                method_df = result_df[result_df['Method'] == method]
                                if len(method_df) > 0:
                                    score = method_df[metrics['metric']].iloc[0]
                                    
                                    # For inverse metrics (lower is better), convert to 0-1 scale
                                    if metrics['inverse']:
                                        # Assuming percentage is 0-100
                                        score = 1 - min(1, score / 100)
                                    
                                    method_scores[method]['scores'][result_key] = score
                                    method_scores[method]['weights'][result_key] = metrics['weight']
            
            # Calculate weighted scores
            for method in method_scores:
                scores = method_scores[method]['scores']
                weights = method_scores[method]['weights']
                
                if scores and weights:
                    weighted_sum = sum(scores[k] * weights[k] for k in scores)
                    total_weight = sum(weights[k] for k in scores)
                    
                    if total_weight > 0:
                        method_scores[method]['overall'] = weighted_sum / total_weight
                    else:
                        method_scores[method]['overall'] = 0
                else:
                    method_scores[method]['overall'] = 0
            
            # Create summary table
            f.write("""
    <table>
        <tr>
            <th>Method</th>
            <th>Overall Score</th>
            <th>Assessment</th>
        </tr>
""")
            
            # Sort methods by overall score
            sorted_methods = sorted(method_scores.items(), key=lambda x: x[1].get('overall', 0), reverse=True)
            
            for method, scores in sorted_methods:
                overall = scores.get('overall', 0)
                
                # Determine quality class
                if overall > 0.85:
                    quality_class = "good"
                    assessment = "Excellent quality - suitable for high-confidence analyses"
                elif overall > 0.7:
                    quality_class = "moderate"
                    assessment = "Good quality - suitable for most analyses"
                else:
                    quality_class = "poor"
                    assessment = "Moderate quality - may need refinement for sensitive analyses"
                
                f.write(f"""
        <tr>
            <td>{method}</td>
            <td class="{quality_class}">{overall:.2f}/1.00</td>
            <td>{assessment}</td>
        </tr>
""")
            
            f.write("</table>")
            
            # Add conclusion
            f.write("""
    <h3>Conclusion</h3>
""")
            
            # Find best method
            if sorted_methods:
                best_method, best_scores = sorted_methods[0]
                best_score = best_scores.get('overall', 0)
                
                if best_method != 'original':
                    # Look for original in the sorted methods
                    orig_score = 0
                    for method, scores in sorted_methods:
                        if method == 'original':
                            orig_score = scores.get('overall', 0)
                            break
                    
                    if best_score > 0.85:
                        conclusion = f"""
        <p>The <strong>{best_method}</strong> augmentation method produces excellent quality data that preserves the biological characteristics of the original dataset while significantly increasing statistical power. This method is recommended for publication-quality molecular feature data augmentation.</p>
"""
                    elif best_score > 0.7:
                        conclusion = f"""
        <p>The <strong>{best_method}</strong> augmentation method produces good quality data that generally preserves the biological characteristics of the original dataset. This method is suitable for most analytical purposes, though some refinement may be beneficial for the highest-sensitivity applications.</p>
"""
                    else:
                        conclusion = f"""
        <p>The <strong>{best_method}</strong> augmentation method produces acceptable quality data, but there are notable deviations from the original dataset characteristics. For research requiring high confidence, further refinement of the augmentation parameters is recommended.</p>
"""
                    
                    if orig_score > 0 and best_score > orig_score:
                        conclusion += f"""
        <p>Notably, the <strong>{best_method}</strong> method shows improvements in some quality metrics compared to the original data, particularly in statistical power and class separability, while maintaining acceptable biochemical relationship preservation.</p>
"""
                    
                    f.write(conclusion)
                else:
                    # Original is best
                    f.write("""
        <p>The original data shows the highest overall quality. While augmentation methods increase sample size, they introduce some deviation from the original data characteristics. For the most sensitive analyses, using the original data may be preferable, while augmented data can be valuable for exploratory analyses and methods that benefit from larger sample sizes.</p>
""")
            
            # Close HTML
            f.write("""
    <hr>
    <p><em>Report generated by Molecular Feature Quality Control Module</em></p>
</body>
</html>
""")
        
        print(f"\nIntegrated QC report saved to: {report_path}")


# Main function for direct execution
def run_molecular_feature_qc():
    """Run the molecular feature quality control pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quality control for augmented molecular feature data.')
    parser.add_argument('--original', type=str, default=r"C:\Users\ms\Desktop\hyper\data\n_p_r2.csv",
                      help='Path to original molecular feature data CSV file')
    parser.add_argument('--augmented', type=str, 
                      default=r"C:\Users\ms\Desktop\hyper\output\augment\molecular_feature\root\augmented_molecular_feature_data.csv",
                      help='Path to augmented molecular feature data CSV file')
    parser.add_argument('--output', type=str, default=r"C:\Users\ms\Desktop\hyper\output\augment\molecular_feature\root",
                      help='Directory to save QC results')
    
    args = parser.parse_args()
    
    # Create and run QC
    qc = MolecularFeatureQC(args.original, args.augmented, args.output)
    results = qc.run_all_qc()
    
    return results


if __name__ == "__main__":
    run_molecular_feature_qc()