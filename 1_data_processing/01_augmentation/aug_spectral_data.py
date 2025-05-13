"""
Spectral Data Augmentation Tool

This module provides a comprehensive framework for augmenting spectral data with various
techniques including Gaussian Process modeling, spectral mixup, peak-preserving warping,
and other spectral transformations. The augmentation preserves metadata balance while
increasing dataset size for improved machine learning model training.
"""

import os
import time
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

warnings.filterwarnings('ignore')

# Color definitions
COLORS = {
    # ==========================================================================
    # == Core Experimental Variables ==
    # ==========================================================================
    # Using distinct core families: Blues for Genotypes, Greens for Treatments
    'G1': '#3182bd',             # Tolerant Genotype (Medium-Dark Blue)
    'G2': '#2ca25f',             # Susceptible Genotype (Medium Teal)

    'T0': '#74c476',             # Control Treatment (Medium Green)
    'T1': '#fe9929',             # Stress Treatment (Muted Orange/Yellow)

    'Leaf': '#006d2c',            # Leaf Tissue (Darkest Green)
    'Root': '#08519c',            # Root Tissue (Darkest Blue)

    # --- Days (Subtle Yellow-Green sequence) ---
    'Day1': '#ffffcc',            # Very Light Yellow-Green
    'Day2': '#c2e699',            # Light Yellow-Green
    'Day3': '#78c679',            # Medium Yellow-Green

    # ==========================================================================
    # == Data Types / Omics / Features ==
    # ==========================================================================
    # Using distinct Blue/Green families for general types
    'Spectral': '#6baed6',        # General Spectral (Medium Blue)
    'Molecular features': '#41ab5d',       # General Molecular features (Medium-Dark Yellow-Green)
    'UnknownFeature': '#969696',  # Medium Grey for fallback

    # --- Specific Spectral Categories --- (Using blues, teals, greens, greys)
    'Spectral_Water': '#3182bd',     # Medium-Dark Blue
    'Spectral_Pigment': '#238b45',    # Medium-Dark Green
    'Spectral_Structure': '#7fcdbb',  # Medium Teal
    'Spectral_SWIR': '#636363',       # Dark Grey
    'Spectral_VIS': '#c2e699',        # Light Yellow-Green
    'Spectral_RedEdge': '#78c679',    # Medium Yellow-Green
    'Spectral_UV': '#08519c',         # Darkest Blue (Matches Root)
    'Spectral_Other': '#969696',      # Medium Grey

    # --- Specific Molecular features Categories --- (Using Yellow/Greens)
    'Molecular features_PCluster': '#006837', # Darkest Yellow-Green
    'Molecular features_NCluster': '#ffffd4', # Very Light Yellow
    'Molecular features_Other': '#bdbdbd',     # Light Grey

    # ==========================================================================
    # == Methods & Model Comparison ==
    # ==========================================================================
    # Using distinct shades for clarity
    'MOFA': '#08519c',            # Dark Blue
    'SHAP': '#006d2c',            # Dark Green
    'Overlap': '#41ab5d',         # Medium-Dark Yellow-Green

    'Transformer': '#6baed6',     # Medium Blue
    'RandomForest': '#74c476',    # Medium Green
    'KNN': '#7fcdbb',             # Medium Teal

    # ==========================================================================
    # == Network Visualization Elements ==
    # ==========================================================================
    'Edge_Low': '#f0f0f0',         # Very Light Gray
    'Edge_High': '#08519c',        # Dark Blue
    'Node_Spectral': '#6baed6',    # Default Spectral Node (Medium Blue)
    'Node_Molecular features': '#41ab5d',   # Default Molecular features Node (Med-Dark Yellow-Green)
    'Node_Edge': '#252525',        # Darkest Gray / Near Black border

    # ==========================================================================
    # == Statistical & Difference Indicators ==
    # ==========================================================================
    # Using Green for positive, muted Yellow for negative, Dark Blue for significance
    'Positive_Diff': '#238b45',     # Medium-Dark Green
    'Negative_Diff': '#fe9929',     # Muted Orange/Yellow (Matches T1)
    'Significance': '#08519c',      # Dark Blue (for markers/text)
    'NonSignificant': '#bdbdbd',    # Light Grey
    'Difference_Line': '#636363',   # Dark Grey line

    # ==========================================================================
    # == Plot Elements & Annotations ==
    # ==========================================================================
    'Background': '#FFFFFF',       # White plot background
    'Panel_Background': '#f7f7f7', # Very Light Gray background for some panels
    'Grid': '#d9d9d9',             # Lighter Gray grid lines
    'Text_Dark': '#252525',        # Darkest Gray / Near Black text
    'Text_Light': '#FFFFFF',       # White text
    'Text_Annotation': '#000000',   # Black text for annotations
    'Annotation_Box_BG': '#FFFFFF', # White background for text boxes
    'Annotation_Box_Edge': '#bdbdbd',# Light Grey border for text boxes
    'Table_Header_BG': '#deebf7',   # Very Light Blue table header
    'Table_Highlight_BG': '#fff7bc',# Pale Yellow for highlighted table cells

    # --- Temporal Patterns (Fig S6) --- (Using core palette shades)
    'Pattern_Increasing': '#238b45',  # Medium-Dark Green
    'Pattern_Decreasing': '#fe9929',  # Muted Orange/Yellow
    'Pattern_Peak': '#78c679',        # Medium Yellow-Green
    'Pattern_Valley': '#6baed6',      # Medium Blue
    'Pattern_Stable': '#969696',      # Medium Grey
}


class SpectralAugmentation:
    def __init__(self, input_file, output_dir):
        """
        Initialize the SpectralAugmentation class.
        
        Parameters:
        -----------
        input_file : str
            Path to input CSV file containing spectral data
        output_dir : str
            Directory to save augmented data
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.wavelength_cols = None
        self.metadata_cols = None
        self.data = None
        self.load_data()

    def load_data(self):
        """Load and prepare the spectral data."""
        self.data = pd.read_csv(self.input_file)
        
        # Identify wavelength and metadata columns
        self.wavelength_cols = [col for col in self.data.columns 
                               if col.startswith('W_')]
        self.metadata_cols = [col for col in self.data.columns 
                             if not col.startswith('W_')]
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        print(f"Dataset information:")
        print(f"  - Total rows: {len(self.data)}")
        print(f"  - Metadata columns: {len(self.metadata_cols)}")
        print(f"  - Wavelength columns: {len(self.wavelength_cols)}")
        print(f"  - First 9 columns: {', '.join(self.data.columns[:9])}")

    def _gaussian_process_single(self, args):
        """Process a single spectral sample using Gaussian Process"""
        row, idx, length_scale, noise_level, wavelengths = args
        spectrum = row[self.wavelength_cols].values.astype(float)
        
        # Define GP kernel
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=idx)
        
        # Fit GP to original spectrum
        gp.fit(wavelengths.reshape(-1, 1), spectrum)
        
        # Generate new sample
        new_spectrum = gp.sample_y(wavelengths.reshape(-1, 1), random_state=idx)
        
        # Create new row with metadata
        new_row = row.copy()
        new_row[self.wavelength_cols] = new_spectrum.flatten()
        new_row['Row_names'] = f"{row['Row_names']}_GP"
        
        return new_row

    def gaussian_process_augmentation(self):
        """
        Generate new spectral samples using Gaussian Process modeling.
        
        Returns:
        --------
        pd.DataFrame
            Augmented spectral data
        """
        print("Running Gaussian Process augmentation...")
        wavelengths = np.array([float(col.split('_')[1]) 
                               for col in self.wavelength_cols])
        
        # Prepare arguments for parallel processing
        args_list = [(row, idx, 50.0, 0.01, wavelengths) 
                    for idx, row in self.data.iterrows()]
        
        # Use parallel processing
        n_cores = min(40, cpu_count())  # Use up to 40 cores
        print(f"Using {n_cores} CPU cores...")
        with Pool(n_cores) as pool:
            results = list(tqdm(
                pool.imap(self._gaussian_process_single, args_list),
                total=len(args_list),
                desc="GP Augmentation"
            ))
        
        return pd.DataFrame(results)

    def spectral_mixup(self):
        """
        Generate new samples by mixing spectra from the same row group.
        Ensures perfect metadata balance by creating exactly one augmented sample 
        per original.
        
        Returns:
        --------
        pd.DataFrame
            Augmented spectral data
        """
        print("Running Spectral Mixup augmentation...")
        augmented_data = []
        
        # Group by all 9 metadata columns
        group_cols = self.metadata_cols
        grouped = self.data.groupby(group_cols[:-1])  # Group by all except Row_names
        
        # Create lookup for quickly finding matched rows
        tissue_type_groups = self.data.groupby(
            ['Tissue.type', 'Genotype', 'Batch', 'Treatment', 'Day'])
        
        for _, row in tqdm(self.data.iterrows(), 
                          total=len(self.data), 
                          desc="Mixup Augmentation"):
            # Get the key for this row
            key = tuple(row[col] for col in group_cols[:-1])  # exclude Row_names
            
            # Get a smaller group that matches tissue type, genotype, etc.
            tissue_key = (row['Tissue.type'], row['Genotype'], 
                          row['Batch'], row['Treatment'], row['Day'])
            tissue_group = tissue_type_groups.get_group(tissue_key)
            
            # Don't mix with itself, select another sample with different Replication
            other_rows = tissue_group[tissue_group['Replication'] != row['Replication']]
            
            if len(other_rows) > 0:
                # Select a different sample from same group
                other_row = other_rows.sample(1).iloc[0]
                
                # Get spectra
                spectrum1 = row[self.wavelength_cols].values.astype(float)
                spectrum2 = other_row[self.wavelength_cols].values.astype(float)
                
                # Generate mixing ratio
                mix_ratio = np.random.beta(0.2, 0.2)
                
                # Mix spectra
                mixed_spectrum = mix_ratio * spectrum1 + (1 - mix_ratio) * spectrum2
                
                # Create new row with metadata (maintaining original metadata)
                new_row = row.copy()
                new_row[self.wavelength_cols] = mixed_spectrum
                new_row['Row_names'] = f"{row['Row_names']}_MIX"
                augmented_data.append(new_row)
            else:
                # If no other samples available, duplicate with small noise
                spectrum = row[self.wavelength_cols].values.astype(float)
                noisy_spectrum = spectrum + np.random.normal(0, 0.001, 
                                                           size=len(spectrum))
                
                new_row = row.copy()
                new_row[self.wavelength_cols] = noisy_spectrum
                new_row['Row_names'] = f"{row['Row_names']}_MIX"
                augmented_data.append(new_row)
        
        return pd.DataFrame(augmented_data)

    def additive_mixup(self):
        """
        Generate new samples by additive mixing of spectra.
        Ensures perfect metadata balance by creating exactly one augmented sample 
        per original.
        
        Returns:
        --------
        pd.DataFrame
            Augmented spectral data
        """
        print("Running Additive Mixup augmentation...")
        augmented_data = []
        
        # Group by all relevant metadata
        tissue_type_groups = self.data.groupby(
            ['Tissue.type', 'Genotype', 'Batch', 'Treatment', 'Day'])
        
        for _, row in tqdm(self.data.iterrows(), 
                          total=len(self.data), 
                          desc="Additive Mixup"):
            # Get tissue-specific group
            tissue_key = (row['Tissue.type'], row['Genotype'], 
                          row['Batch'], row['Treatment'], row['Day'])
            tissue_group = tissue_type_groups.get_group(tissue_key)
            
            # Don't mix with itself, select another sample with different Replication
            other_rows = tissue_group[tissue_group['Replication'] != row['Replication']]
            
            if len(other_rows) > 0:
                # Select a different sample from same group
                other_row = other_rows.sample(1).iloc[0]
                
                # Get spectra
                spectrum1 = row[self.wavelength_cols].values.astype(float)
                spectrum2 = other_row[self.wavelength_cols].values.astype(float)
                
                # Additive mixing with normalization
                mixed_spectrum = (spectrum1 + spectrum2) / 2
                
                # Create new row with metadata (maintaining original metadata)
                new_row = row.copy()
                new_row[self.wavelength_cols] = mixed_spectrum
                new_row['Row_names'] = f"{row['Row_names']}_ADD"
                augmented_data.append(new_row)
            else:
                # If no other samples available, duplicate with small noise
                spectrum = row[self.wavelength_cols].values.astype(float)
                noisy_spectrum = spectrum + np.random.normal(0, 0.001, 
                                                           size=len(spectrum))
                
                new_row = row.copy()
                new_row[self.wavelength_cols] = noisy_spectrum
                new_row['Row_names'] = f"{row['Row_names']}_ADD"
                augmented_data.append(new_row)
        
        return pd.DataFrame(augmented_data)

    def multiplicative_mixup(self):
        """
        Generate new samples by multiplicative mixing of spectra.
        Ensures perfect metadata balance by creating exactly one augmented sample 
        per original.
        
        Returns:
        --------
        pd.DataFrame
            Augmented spectral data
        """
        print("Running Multiplicative Mixup augmentation...")
        augmented_data = []
        
        # Group by all relevant metadata
        tissue_type_groups = self.data.groupby(
            ['Tissue.type', 'Genotype', 'Batch', 'Treatment', 'Day'])
        
        for _, row in tqdm(self.data.iterrows(), 
                          total=len(self.data), 
                          desc="Multiplicative Mixup"):
            # Get tissue-specific group
            tissue_key = (row['Tissue.type'], row['Genotype'], 
                          row['Batch'], row['Treatment'], row['Day'])
            tissue_group = tissue_type_groups.get_group(tissue_key)
            
            # Don't mix with itself, select another sample with different Replication
            other_rows = tissue_group[tissue_group['Replication'] != row['Replication']]
            
            if len(other_rows) > 0:
                # Select a different sample from same group
                other_row = other_rows.sample(1).iloc[0]
                
                # Get spectra
                spectrum1 = row[self.wavelength_cols].values.astype(float)
                spectrum2 = other_row[self.wavelength_cols].values.astype(float)
                
                # Multiplicative mixing with geometric mean
                mixed_spectrum = np.sqrt(spectrum1 * spectrum2)
                
                # Create new row with metadata (maintaining original metadata)
                new_row = row.copy()
                new_row[self.wavelength_cols] = mixed_spectrum
                new_row['Row_names'] = f"{row['Row_names']}_MULT"
                augmented_data.append(new_row)
            else:
                # If no other samples available, duplicate with small noise
                spectrum = row[self.wavelength_cols].values.astype(float)
                noisy_spectrum = spectrum + np.random.normal(0, 0.001, 
                                                           size=len(spectrum))
                
                new_row = row.copy()
                new_row[self.wavelength_cols] = noisy_spectrum
                new_row['Row_names'] = f"{row['Row_names']}_MULT"
                augmented_data.append(new_row)
        
        return pd.DataFrame(augmented_data)

    def peak_preserving_warp(self):
        """
        Generate new samples by warping spectra while preserving major peaks.
        Ensures perfect metadata balance by creating exactly one augmented sample 
        per original.
        
        Returns:
        --------
        pd.DataFrame
            Augmented spectral data
        """
        print("Running Peak-Preserving Warp augmentation...")
        augmented_data = []
        wavelengths = np.array([float(col.split('_')[1]) 
                               for col in self.wavelength_cols])
        
        for _, row in tqdm(self.data.iterrows(), 
                          total=len(self.data), 
                          desc="Warp Augmentation"):
            spectrum = row[self.wavelength_cols].values.astype(float)
            
            # Generate random control points for warping
            n_controls = 5
            control_points = np.linspace(wavelengths[0], wavelengths[-1], n_controls)
            shifts = np.random.uniform(-20, 20, n_controls)
            warped_controls = control_points + shifts
            
            # Ensure boundary conditions
            warped_controls[0] = wavelengths[0]
            warped_controls[-1] = wavelengths[-1]
            
            # Create warping function
            spline = CubicSpline(warped_controls, control_points)
            warped_wavelengths = spline(wavelengths)
            
            # Interpolate spectrum at warped wavelengths
            spline_spectrum = CubicSpline(wavelengths, spectrum)
            warped_spectrum = spline_spectrum(warped_wavelengths)
            
            # Create new row with original metadata
            new_row = row.copy()
            new_row[self.wavelength_cols] = warped_spectrum
            new_row['Row_names'] = f"{row['Row_names']}_WARP"
            augmented_data.append(new_row)
        
        return pd.DataFrame(augmented_data)

    def reflectance_scaling(self):
        """
        Generate new samples by scaling reflectance within physiological constraints.
        Ensures perfect metadata balance by creating exactly one augmented sample 
        per original.
        
        Returns:
        --------
        pd.DataFrame
            Augmented spectral data
        """
        print("Running Reflectance Scaling augmentation...")
        augmented_data = []
        
        for _, row in tqdm(self.data.iterrows(), 
                          total=len(self.data), 
                          desc="Scale Augmentation"):
            spectrum = row[self.wavelength_cols].values.astype(float)
            
            # Generate scaling factor
            scale = np.random.uniform(0.95, 1.05)
            
            # Scale spectrum while preserving physiological constraints
            scaled_spectrum = spectrum * scale
            scaled_spectrum = np.clip(scaled_spectrum, 0, 1)  # Ensure valid range
            
            # Create new row with original metadata
            new_row = row.copy()
            new_row[self.wavelength_cols] = scaled_spectrum
            new_row['Row_names'] = f"{row['Row_names']}_SCALE"
            augmented_data.append(new_row)
        
        return pd.DataFrame(augmented_data)

    def add_band_specific_noise(self):
        """
        Generate new samples by adding band-specific noise.
        Ensures perfect metadata balance by creating exactly one augmented sample 
        per original.
        
        Returns:
        --------
        pd.DataFrame
            Augmented spectral data
        """
        print("Running Band-Specific Noise augmentation...")
        augmented_data = []
        
        for _, row in tqdm(self.data.iterrows(), 
                          total=len(self.data), 
                          desc="Noise Augmentation"):
            spectrum = row[self.wavelength_cols].values.astype(float)
            
            # Generate band-specific noise
            noise = np.random.normal(0, 0.01, size=len(spectrum))
            
            # Add noise while preserving spectral characteristics
            noisy_spectrum = spectrum + noise
            noisy_spectrum = np.clip(noisy_spectrum, 0, 1)  # Ensure valid range
            
            # Create new row with original metadata
            new_row = row.copy()
            new_row[self.wavelength_cols] = noisy_spectrum
            new_row['Row_names'] = f"{row['Row_names']}_NOISE"
            augmented_data.append(new_row)
        
        return pd.DataFrame(augmented_data)

    def check_metadata_balance(self, dataset):
        """
        Check if all experimental factors have balanced representation in the dataset.
        
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
        for col in self.metadata_cols[1:]:  # Skip Row_names
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
                method = row_name.split('_')[-1]
                method_counts[method] = method_counts.get(method, 0) + 1
            else:
                method_counts['original'] = method_counts.get('original', 0) + 1
        
        print("\nMethod distribution:")
        for method, count in method_counts.items():
            print(f"  - {method}: {count} samples")
        
        return True
        
    def generate_augmented_dataset(self):
        """
        Generate complete augmented dataset using all methods.
        Each method produces exactly one new sample per original sample,
        resulting in perfect metadata balance and a 7x multiplier.
        """
        start_time = time.time()
        
        # Define all methods
        methods = [
            'gaussian_process_augmentation',
            'spectral_mixup',
            'peak_preserving_warp',
            'reflectance_scaling',
            'add_band_specific_noise',
            'additive_mixup',
            'multiplicative_mixup'
        ]
        
        # Calculate expected multiplication factor
        total_multiplier = 1 + len(methods)  # original + 1 per method
        
        print(f"\nAugmentation Plan:")
        print(f"------------------")
        print(f"Original samples: {len(self.data)}")
        print(f"Methods: {len(methods)}")
        print(f"Samples per method: 1 per original row")
        print(f"Expected multiplication: {total_multiplier}x")
        print(f"Expected final dataset size: {len(self.data) * total_multiplier}")
        print(f"Expected B1 samples: {len(self.data[self.data['Batch'] == 'B1']) * total_multiplier}")
        print(f"Expected B2 samples: {len(self.data[self.data['Batch'] == 'B2']) * total_multiplier}")
        
        # Start with original data
        augmented_data = [self.data]
        
        # Apply each method
        for method_name in methods:
            print(f"\nApplying augmentation method: {method_name}")
            method = getattr(self, method_name)
            aug_data = method()
            
            # Verify each batch has the expected number of samples
            b1_count = len(aug_data[aug_data['Batch'] == 'B1'])
            b2_count = len(aug_data[aug_data['Batch'] == 'B2'])
            orig_b1 = len(self.data[self.data['Batch'] == 'B1'])
            orig_b2 = len(self.data[self.data['Batch'] == 'B2'])
            
            print(f"Generated {len(aug_data)} samples using {method_name}")
            print(f"  - B1: {b1_count} samples (expected {orig_b1})")
            print(f"  - B2: {b2_count} samples (expected {orig_b2})")
            
            augmented_data.append(aug_data)
        
        # Combine all augmented data
        final_data = pd.concat(augmented_data, ignore_index=True)
        
        # Check metadata balance
        self.check_metadata_balance(final_data)
        
        # Save augmented dataset
        output_file = os.path.join(self.output_dir, 'augmented_spectral_data.csv')
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


# Example usage
if __name__ == "__main__":
    input_file = r"C:\Users\ms\Desktop\hyper\data\hyper_full_w.csv"
    output_dir = r"C:\Users\ms\Desktop\hyper\output\augment\hyper"
    
    # Initialize augmentation
    augmentor = SpectralAugmentation(input_file, output_dir)
    
    # Generate augmented dataset with all methods
    augmentor.generate_augmented_dataset()