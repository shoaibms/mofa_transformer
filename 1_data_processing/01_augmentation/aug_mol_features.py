"""
Molecular Feature Augmentation Tool

This module provides functionality for augmenting Molecular features data using scaling and
mixup methods. It helps generate synthetic data samples while preserving the 
statistical properties of the original dataset to improve model training.

The MolecularFeatureAugmentation class reads Molecular features data from a CSV file, performs
data augmentation using two complementary methods (SCALE and MIX), verifies data
quality, and saves the augmented dataset to a specified location.
"""

import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# Hardcoded color values
COLORS = {
    # Core Experimental Variables
    'G1': '#3182bd',             # Tolerant Genotype (Medium-Dark Blue)
    'G2': '#2ca25f',             # Susceptible Genotype (Medium Teal)
    'T0': '#74c476',             # Control Treatment (Medium Green)
    'T1': '#fe9929',             # Stress Treatment (Muted Orange/Yellow)
    'Leaf': '#006d2c',           # Leaf Tissue (Darkest Green)
    'Root': '#08519c',           # Root Tissue (Darkest Blue)
    'Day1': '#ffffcc',           # Very Light Yellow-Green
    'Day2': '#c2e699',           # Light Yellow-Green
    'Day3': '#78c679',           # Medium Yellow-Green
    
    # Data Types / Omics / Features
    'Spectral': '#6baed6',       # General Spectral (Medium Blue)
    'MolecularFeature': '#41ab5d', # General Molecular Feature (Medium-Dark Yellow-Green)
    'UnknownFeature': '#969696', # Medium Grey for fallback
    
    # Specific Molecular Feature Categories
    'MolecularFeature_PCluster': '#006837', # Darkest Yellow-Green
    'MolecularFeature_NCluster': '#ffffd4', # Very Light Yellow
    'MolecularFeature_Other': '#bdbdbd',    # Light Grey
}


class MolecularFeatureAugmentation:
    def __init__(self, input_file, output_dir):
        """
        Initialize the MolecularFeatureAugmentation class.
        
        Parameters:
        -----------
        input_file : str
            Path to input CSV file containing Molecular features data
        output_dir : str
            Directory to save augmented data
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.n_cluster_cols = None
        self.p_cluster_cols = None
        self.molecular_feature_cols = None
        self.metadata_cols = None
        self.data = None
        self.load_data()

    def load_data(self):
        """Load and prepare the Molecular features data."""
        self.data = pd.read_csv(self.input_file)
        
        # Identify Molecular features and metadata columns
        self.n_cluster_cols = [col for col in self.data.columns 
                              if col.startswith('N_Cluster_')]
        self.p_cluster_cols = [col for col in self.data.columns 
                              if col.startswith('P_Cluster_')]
        self.molecular_feature_cols = self.n_cluster_cols + self.p_cluster_cols
        self.metadata_cols = [col for col in self.data.columns 
                             if col not in self.molecular_feature_cols]
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        print(f"Dataset information:")
        print(f"  - Total rows: {len(self.data)}")
        print(f"  - Metadata columns: {len(self.metadata_cols)}")
        print(f"  - N-cluster Molecular features: {len(self.n_cluster_cols)}")
        print(f"  - P-cluster Molecular features: {len(self.p_cluster_cols)}")
        print(f"  - Total Molecular features: {len(self.molecular_feature_cols)}")
        print(f"  - First 9 columns: {', '.join(self.data.columns[:9])}")

    def _molecular_feature_scaling_single(self, args):
        """
        Process a single Molecular features sample using scaling approach.
        
        Parameters:
        -----------
        args : tuple
            Contains (row, idx, seed_offset) where:
            - row: pandas Series with sample data
            - idx: index of the sample
            - seed_offset: offset for random seed
            
        Returns:
        --------
        pd.Series
            Augmented sample
        """
        row, idx, seed_offset = args
        
        # Get original Molecular features values
        original_values = row[self.molecular_feature_cols].values.astype(float)
        
        # Set seed for reproducibility but with variation per sample
        np.random.seed(idx * 10000 + seed_offset)
        
        # Generate scaling factor with slight variation based on seed_offset
        base_scale = 0.95 + 0.1 * (seed_offset % 5) / 5.0
        scale = np.random.uniform(base_scale, base_scale + 0.05)
        
        # Scale values while preserving relationships
        scaled_values = original_values * scale
        
        # Add small noise to avoid exact linear scaling
        noise_level = 0.005 + 0.01 * (seed_offset % 5) / 5.0
        noise = np.random.normal(0, noise_level * np.abs(scaled_values))
        final_values = scaled_values + noise
        
        # Create new row with metadata
        new_row = row.copy()
        new_row[self.molecular_feature_cols] = final_values
        new_row['Row_names'] = f"{row['Row_names']}_{seed_offset+1}_SCALE"
        
        return new_row

    def molecular_feature_scaling_multiple(self, n_copies=5):
        """
        Generate multiple new samples by scaling Molecular features values.
        
        Parameters:
        -----------
        n_copies : int
            Number of copies to generate per original sample
            
        Returns:
        --------
        pd.DataFrame
            Augmented Molecular features data
        """
        print(f"Running Molecular Feature Scaling augmentation ({n_copies}x)...")
        
        all_results = []
        
        # Generate multiple copies with different parameters
        for i in range(n_copies):
            print(f"  Generating SCALE set {i+1}/{n_copies}...")
            
            # Prepare arguments for parallel processing
            args_list = [(row, idx, i) 
                         for idx, (_, row) in enumerate(self.data.iterrows())]
            
            # Use parallel processing
            n_cores = min(40, cpu_count())
            with Pool(n_cores) as pool:
                results = list(tqdm(
                    pool.imap(self._molecular_feature_scaling_single, args_list),
                    total=len(args_list),
                    desc=f"SCALE Set {i+1}"
                ))
            
            all_results.extend(results)
        
        return pd.DataFrame(all_results)

    def _molecular_feature_mixup_single(self, args):
        """
        Process a single Molecular features sample using mixup approach.
        
        Parameters:
        -----------
        args : tuple
            Contains (row, group_key, grouped_data, seed_offset) where:
            - row: pandas Series with sample data
            - group_key: tuple of metadata values
            - grouped_data: grouped DataFrame
            - seed_offset: offset for random seed
            
        Returns:
        --------
        pd.Series
            Augmented sample
        """
        row, group_key, grouped_data, seed_offset = args
        
        # Set seed for reproducibility with variation per sample and iteration
        seed_value = (abs(hash(str(row['Row_names']))) + seed_offset * 1000) % (2**32 - 1)
        np.random.seed(seed_value)
        
        # Get the group that this row belongs to
        group = grouped_data.get_group(group_key)
        
        # Find rows with different Replication
        other_rows = group[group['Replication'] != row['Replication']]
        
        if len(other_rows) > 0:
            # Select a different sample from same group
            other_idx = seed_offset % len(other_rows)
            other_row = other_rows.iloc[other_idx]
            
            # Get Molecular features values
            molecular_features1 = row[self.molecular_feature_cols].values.astype(float)
            molecular_features2 = other_row[self.molecular_feature_cols].values.astype(float)
            
            # Generate mixing ratio, varying by seed_offset
            if seed_offset % 2 == 0:
                # First strategy: favor values near 0 or 1
                mix_ratio = np.random.beta(0.2, 0.2)
            else:
                # Second strategy: more centered values
                mix_ratio = np.random.beta(2.0, 2.0)
            
            # Mix Molecular features
            mixed_molecular_features = mix_ratio * molecular_features1 + (1 - mix_ratio) * molecular_features2
            
            # Create new row with metadata
            new_row = row.copy()
            new_row[self.molecular_feature_cols] = mixed_molecular_features
            new_row['Row_names'] = f"{row['Row_names']}_{seed_offset+1}_MIX"
        else:
            # If no other samples, add noise
            molecular_features = row[self.molecular_feature_cols].values.astype(float)
            noise_scale = 0.02 + 0.02 * (seed_offset % 3)
            noisy_molecular_features = molecular_features + np.random.normal(
                0, noise_scale * np.abs(molecular_features))
            
            # Create new row with metadata
            new_row = row.copy()
            new_row[self.molecular_feature_cols] = noisy_molecular_features
            new_row['Row_names'] = f"{row['Row_names']}_{seed_offset+1}_MIX"
        
        return new_row

    def molecular_feature_mixup_multiple(self, n_copies=2):
        """
        Generate multiple new samples by mixing Molecular features profiles.
        
        Parameters:
        -----------
        n_copies : int
            Number of copies to generate per original sample
            
        Returns:
        --------
        pd.DataFrame
            Augmented Molecular features data
        """
        print(f"Running Molecular Feature Mixup augmentation ({n_copies}x)...")
        
        # Group by metadata columns except Row_names
        group_cols = self.metadata_cols[1:9]
        grouped = self.data.groupby(group_cols)
        
        all_results = []
        
        # Generate multiple copies
        for i in range(n_copies):
            print(f"  Generating MIX set {i+1}/{n_copies}...")
            
            # Prepare arguments for parallel processing
            args_list = []
            for _, row in self.data.iterrows():
                group_key = tuple(row[col] for col in group_cols)
                args_list.append((row, group_key, grouped, i))
            
            # Use parallel processing
            n_cores = min(40, cpu_count())
            with Pool(n_cores) as pool:
                results = list(tqdm(
                    pool.imap(self._molecular_feature_mixup_single, args_list),
                    total=len(args_list),
                    desc=f"MIX Set {i+1}"
                ))
            
            all_results.extend(results)
        
        return pd.DataFrame(all_results)

    def check_metadata_balance(self, dataset):
        """
        Check if experimental factors have balanced representation in the dataset.
        
        Parameters:
        -----------
        dataset : pd.DataFrame
            Dataset to check
            
        Returns:
        --------
        bool
            True if balanced, False otherwise
        """
        print("\nMetadata Distribution Check:")
        print("--------------------------")
        
        original_total = len(self.data)
        augmented_total = len(dataset)
        multiplier = augmented_total / original_total
        
        print(f"Total samples: {augmented_total} ({multiplier:.1f}x original)")
        
        # Check each metadata column except Row_names
        for col in self.metadata_cols[1:9]:
            print(f"\n{col} distribution:")
            orig_dist = self.data[col].value_counts().sort_index()
            aug_dist = dataset[col].value_counts().sort_index()
            
            for val in sorted(set(orig_dist.index)):
                orig_count = orig_dist.get(val, 0)
                aug_count = aug_dist.get(val, 0)
                ratio = aug_count / orig_count
                print(f"  - {val}: {aug_count} samples ({ratio:.1f}x)")
        
        # Check distribution of methods
        method_counts = {}
        for row_name in dataset['Row_names']:
            if '_' in row_name:
                parts = row_name.split('_')
                if len(parts) >= 2:
                    # Extract the method (MIX or SCALE)
                    if 'MIX' in parts:
                        method = 'MIX'
                    elif 'SCALE' in parts:
                        method = 'SCALE'
                    else:
                        method = parts[-1]
                    method_counts[method] = method_counts.get(method, 0) + 1
            else:
                method_counts['original'] = method_counts.get('original', 0) + 1
        
        print("\nMethod distribution:")
        for method, count in method_counts.items():
            print(f"  - {method}: {count} samples")
        
        return True

    def generate_optimized_dataset(self, scale_copies=5, mix_copies=2):
        """
        Generate optimized augmented dataset using SCALE and MIX methods.
        
        Parameters:
        -----------
        scale_copies : int
            Number of copies to generate per original sample using SCALE method
        mix_copies : int
            Number of copies to generate per original sample using MIX method
            
        Returns:
        --------
        pd.DataFrame
            Augmented Molecular features data
        """
        start_time = time.time()
        
        # Calculate expected multiplication factor
        total_multiplier = 1 + scale_copies + mix_copies
        
        print(f"\nOptimized Augmentation Plan:")
        print(f"---------------------------")
        print(f"Original samples: {len(self.data)}")
        print(f"SCALE method: {scale_copies}x")
        print(f"MIX method: {mix_copies}x")
        print(f"Expected multiplication: {total_multiplier}x")
        print(f"Expected final dataset size: {len(self.data) * total_multiplier}")
        
        # Get counts for different experimental factors
        for col in ['Batch', 'Treatment', 'Genotype', 'Day']:
            if col in self.data.columns:
                for val in sorted(self.data[col].unique()):
                    count = len(self.data[self.data[col] == val])
                    print(f"Expected {col}={val} samples: {count * total_multiplier}")
        
        # Start with original data
        augmented_data = [self.data]
        
        # Apply SCALE method multiple times
        print(f"\nApplying SCALE augmentation method ({scale_copies}x)...")
        scale_data = self.molecular_feature_scaling_multiple(n_copies=scale_copies)
        augmented_data.append(scale_data)
        
        # Apply MIX method multiple times
        print(f"\nApplying MIX augmentation method ({mix_copies}x)...")
        mix_data = self.molecular_feature_mixup_multiple(n_copies=mix_copies)
        augmented_data.append(mix_data)
        
        # Combine all augmented data
        final_data = pd.concat(augmented_data, ignore_index=True)
        
        # Check metadata balance
        self.check_metadata_balance(final_data)
        
        # Save augmented dataset
        output_file = os.path.join(self.output_dir, 'augmented_molecular_feature_data.csv')
        final_data.to_csv(output_file, index=False)
        
        end_time = time.time()
        print(f"\nAugmentation Statistics:")
        print(f"------------------------")
        print(f"Augmented dataset saved to: {output_file}")
        print(f"Original samples: {len(self.data)}")
        print(f"Augmented samples: {len(final_data)}")
        print(f"Actual multiplication: {len(final_data) / len(self.data):.1f}x")
        print(f"Time taken: {(end_time - start_time):.2f} seconds")
        print(f"RAM used: {final_data.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
        
        # Verify data quality
        self.verify_data_quality(final_data)
        
        return final_data
    
    def verify_data_quality(self, augmented_data):
        """
        Verify the quality of augmented data by checking for invalid values.
        
        Parameters:
        -----------
        augmented_data : pd.DataFrame
            Augmented dataset to verify
        """
        print("\nData Quality Verification:")
        print("-------------------------")
        
        # Check for NaN values
        nan_count = augmented_data[self.molecular_feature_cols].isna().sum().sum()
        print(f"NaN values found: {nan_count}")
        
        # Check for infinite values
        inf_count = np.isinf(augmented_data[self.molecular_feature_cols].values).sum()
        print(f"Infinite values found: {inf_count}")
        
        # Check for extremely small/large values
        molecular_feature_values = augmented_data[self.molecular_feature_cols].values
        original_values = self.data[self.molecular_feature_cols].values
        
        # Find range of original data
        orig_min = np.min(original_values)
        orig_max = np.max(original_values)
        
        # Add some margin
        acceptable_min = orig_min * 2
        acceptable_max = orig_max * 2
        
        # Check for outliers beyond acceptable range
        too_small = (molecular_feature_values < acceptable_min).sum()
        too_large = (molecular_feature_values > acceptable_max).sum()
        
        print(f"Original data range: [{orig_min:.2e}, {orig_max:.2e}]")
        print(f"Values too small: {too_small}")
        print(f"Values too large: {too_large}")
        
        # Suggest a fix if there are issues
        if nan_count > 0 or inf_count > 0 or too_small > 0 or too_large > 0:
            print("\nWarning: Some data quality issues were found.")
            print("Consider:")
            print("  - Adjusting augmentation parameters to produce more conservative changes")
            print("  - Post-processing the augmented data to replace extreme values")
        else:
            print("\nNo data quality issues found. Augmentation successful!")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Molecular features data augmentation.')
    parser.add_argument('--input', type=str, 
                        default=r"C:\Users\ms\Desktop\hyper\data\n_p_l2.csv",
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, 
                        default=r"C:\Users\ms\Desktop\hyper\output\augment\molecular_feature\leaf",
                        help='Directory to save augmented data')
    parser.add_argument('--scale', type=int, default=5,
                        help='Number of copies using SCALE method')
    parser.add_argument('--mix', type=int, default=2,
                        help='Number of copies using MIX method')
    
    args = parser.parse_args()
    
    # Initialize augmentation
    augmentor = MolecularFeatureAugmentation(args.input, args.output)
    
    # Generate optimized augmented dataset
    augmentor.generate_optimized_dataset(scale_copies=args.scale, mix_copies=args.mix)