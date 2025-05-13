"""
Spectral Quality Control Module

This module provides tools for quality control of spectral data, particularly
for assessing and validating augmented spectral datasets against original data.
It includes functionality for outlier detection, signal-to-noise assessment, 
band-specific validation, and range checking against physical constraints.

The SpectralQC class provides a comprehensive framework for ensuring the 
quality and validity of spectral data augmentation techniques.
"""

import numpy as np
import pandas as pd
import os
import time
from scipy.stats import zscore, iqr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class SpectralQC:
    """
    Quality control module for spectral data to detect outliers, assess signal-to-noise ratio,
    perform band-specific validation, and check ranges against physiological constraints.
    """
    
    def __init__(self, original_path, augmented_path, output_dir):
        """
        Initialize the SpectralQC class.
        
        Parameters:
        -----------
        original_path : str
            Path to original spectral data CSV file
        augmented_path : str
            Path to augmented spectral data CSV file
        output_dir : str
            Directory to save QC results
        """
        self.original_path = original_path
        self.augmented_path = augmented_path
        self.output_dir = output_dir
        
        # Create output directory for QC results
        self.qc_dir = os.path.join(output_dir, 'quality_control')
        if not os.path.exists(self.qc_dir):
            os.makedirs(self.qc_dir)
            
        # Load data
        self.load_data()
        
        # Define spectral regions
        self.define_spectral_regions()
        
    def load_data(self):
        """Load and prepare the original and augmented spectral data."""
        print("Loading data...")
        self.original_data = pd.read_csv(self.original_path)
        self.augmented_data = pd.read_csv(self.augmented_path)
        
        # Identify wavelength and metadata columns
        self.wavelength_cols = [col for col in self.original_data.columns if col.startswith('W_')]
        self.metadata_cols = [col for col in self.original_data.columns if not col.startswith('W_')]
        
        # Extract wavelength values
        self.wavelengths = np.array([float(col.split('_')[1]) for col in self.wavelength_cols])
        
        # Separate augmented data
        self.augmented_only = self.augmented_data[~self.augmented_data['Row_names'].isin(self.original_data['Row_names'])]
        
        # Extract spectra
        self.original_spectra = self.original_data[self.wavelength_cols].values
        self.augmented_spectra = self.augmented_only[self.wavelength_cols].values
        
        print(f"Loaded {len(self.original_data)} original samples")
        print(f"Loaded {len(self.augmented_only)} augmented samples")
        
    def define_spectral_regions(self):
        """Define key spectral regions for specific analyses."""
        # Define regions
        self.regions = {
            'Visible': (400, 700),
            'Green': (500, 600),
            'Red': (600, 700),
            'Red_Edge': (680, 750),
            'NIR': (750, 1300),
            'SWIR1': (1300, 1800),
            'SWIR2': (1800, 2500),
            'Water_Absorption1': (1350, 1450),
            'Water_Absorption2': (1800, 1950)
        }
        
        # Get indices for each region
        self.region_indices = {}
        for region, (start, end) in self.regions.items():
            self.region_indices[region] = np.where((self.wavelengths >= start) & (self.wavelengths <= end))[0]
    
    def detect_outliers(self, method='combined', contamination=0.05):
        """
        Detect outliers in both original and augmented spectral data.
        
        Parameters:
        -----------
        method : str
            Outlier detection method: 'zscore', 'iqr', 'isolation_forest', 'lof', or 'combined'
        contamination : float
            Expected proportion of outliers (for isolation_forest and lof methods)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with outlier detection results
        """
        print(f"Detecting outliers using {method} method...")
        
        # Prepare results storage
        outlier_results = {
            'original': {'total': len(self.original_spectra), 'outliers': 0, 'outlier_indices': []},
            'augmented': {'total': len(self.augmented_spectra), 'outliers': 0, 'outlier_indices': []}
        }
        
        # Add category breakdown for augmented data
        augmentation_methods = self.identify_augmentation_methods()
        for aug_method in augmentation_methods:
            outlier_results[aug_method] = {'total': 0, 'outliers': 0, 'outlier_indices': []}
        
        # Perform outlier detection based on the specified method
        if method == 'zscore' or method == 'combined':
            # Z-score method
            orig_outliers_z = self.zscore_outliers(self.original_spectra)
            aug_outliers_z = self.zscore_outliers(self.augmented_spectra)
            
            if method == 'zscore':
                outlier_results['original']['outliers'] = sum(orig_outliers_z)
                outlier_results['original']['outlier_indices'] = np.where(orig_outliers_z)[0].tolist()
                outlier_results['augmented']['outliers'] = sum(aug_outliers_z)
                outlier_results['augmented']['outlier_indices'] = np.where(aug_outliers_z)[0].tolist()
        
        if method == 'iqr' or method == 'combined':
            # IQR method
            orig_outliers_iqr = self.iqr_outliers(self.original_spectra)
            aug_outliers_iqr = self.iqr_outliers(self.augmented_spectra)
            
            if method == 'iqr':
                outlier_results['original']['outliers'] = sum(orig_outliers_iqr)
                outlier_results['original']['outlier_indices'] = np.where(orig_outliers_iqr)[0].tolist()
                outlier_results['augmented']['outliers'] = sum(aug_outliers_iqr)
                outlier_results['augmented']['outlier_indices'] = np.where(aug_outliers_iqr)[0].tolist()
        
        if method == 'isolation_forest' or method == 'combined':
            # Isolation Forest method
            orig_outliers_if = self.isolation_forest_outliers(self.original_spectra, contamination)
            aug_outliers_if = self.isolation_forest_outliers(self.augmented_spectra, contamination)
            
            if method == 'isolation_forest':
                outlier_results['original']['outliers'] = sum(orig_outliers_if)
                outlier_results['original']['outlier_indices'] = np.where(orig_outliers_if)[0].tolist()
                outlier_results['augmented']['outliers'] = sum(aug_outliers_if)
                outlier_results['augmented']['outlier_indices'] = np.where(aug_outliers_if)[0].tolist()
        
        if method == 'lof' or method == 'combined':
            # Local Outlier Factor method
            orig_outliers_lof = self.lof_outliers(self.original_spectra, contamination)
            aug_outliers_lof = self.lof_outliers(self.augmented_spectra, contamination)
            
            if method == 'lof':
                outlier_results['original']['outliers'] = sum(orig_outliers_lof)
                outlier_results['original']['outlier_indices'] = np.where(orig_outliers_lof)[0].tolist()
                outlier_results['augmented']['outliers'] = sum(aug_outliers_lof)
                outlier_results['augmented']['outlier_indices'] = np.where(aug_outliers_lof)[0].tolist()
        
        # Combined approach for more robust outlier detection
        if method == 'combined':
            # Combine results (sample is outlier if detected by at least 2 methods)
            orig_outlier_counts = np.zeros(len(self.original_spectra))
            aug_outlier_counts = np.zeros(len(self.augmented_spectra))
            
            for outliers in [orig_outliers_z, orig_outliers_iqr, orig_outliers_if, orig_outliers_lof]:
                orig_outlier_counts[outliers] += 1
                
            for outliers in [aug_outliers_z, aug_outliers_iqr, aug_outliers_if, aug_outliers_lof]:
                aug_outlier_counts[outliers] += 1
            
            # Mark as outlier if detected by at least 2 methods
            orig_outliers = orig_outlier_counts >= 2
            aug_outliers = aug_outlier_counts >= 2
            
            outlier_results['original']['outliers'] = sum(orig_outliers)
            outlier_results['original']['outlier_indices'] = np.where(orig_outliers)[0].tolist()
            outlier_results['augmented']['outliers'] = sum(aug_outliers)
            outlier_results['augmented']['outlier_indices'] = np.where(aug_outliers)[0].tolist()
        
        # Analyze outliers by augmentation method
        if method == 'combined' or method == 'isolation_forest':  # Use the selected method for breakdown
            outlier_method = aug_outliers if method == 'combined' else aug_outliers_if
            
            # Get augmentation method for each sample
            for i, row_name in enumerate(self.augmented_only['Row_names']):
                for aug_method in augmentation_methods:
                    if aug_method.upper() in row_name:
                        outlier_results[aug_method]['total'] += 1
                        if outlier_method[i]:
                            outlier_results[aug_method]['outliers'] += 1
                            outlier_results[aug_method]['outlier_indices'].append(i)
                        break
        
        # Calculate percentages
        for category in outlier_results:
            if outlier_results[category]['total'] > 0:
                outlier_results[category]['percentage'] = (
                    outlier_results[category]['outliers'] / outlier_results[category]['total'] * 100
                )
            else:
                outlier_results[category]['percentage'] = 0
        
        # Create summary DataFrame
        summary_data = []
        for category, results in outlier_results.items():
            if results['total'] > 0:
                summary_data.append({
                    'Category': category,
                    'Total_Samples': results['total'],
                    'Outliers': results['outliers'],
                    'Percentage': results['percentage']
                })
        
        outlier_summary = pd.DataFrame(summary_data)
        
        # Save results
        outlier_file = os.path.join(self.qc_dir, f'outlier_detection_{method}.csv')
        outlier_summary.to_csv(outlier_file, index=False)
        
        print(f"Outlier detection results saved to: {outlier_file}")
        print(outlier_summary)
        
        return outlier_results
    
    def zscore_outliers(self, spectra, threshold=3.0):
        """
        Detect outliers using Z-score method.
        
        Parameters:
        -----------
        spectra : numpy.ndarray
            Spectral data array
        threshold : float
            Z-score threshold for outlier detection
            
        Returns:
        --------
        numpy.ndarray
            Boolean array indicating outliers
        """
        # Calculate mean spectrum
        mean_spectrum = np.mean(spectra, axis=0)
        
        # Calculate distances from mean
        distances = np.sqrt(np.sum((spectra - mean_spectrum)**2, axis=1))
        
        # Calculate z-scores of distances
        z = zscore(distances)
        
        # Identify outliers (samples with z-score above threshold)
        outliers = z > threshold
        
        return outliers
    
    def iqr_outliers(self, spectra, k=1.5):
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
        -----------
        spectra : numpy.ndarray
            Spectral data array
        k : float
            Multiplier for IQR
            
        Returns:
        --------
        numpy.ndarray
            Boolean array indicating outliers
        """
        # Calculate mean spectrum
        mean_spectrum = np.mean(spectra, axis=0)
        
        # Calculate distances from mean
        distances = np.sqrt(np.sum((spectra - mean_spectrum)**2, axis=1))
        
        # Calculate IQR
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr_val = q3 - q1
        
        # Define bounds
        lower_bound = q1 - k * iqr_val
        upper_bound = q3 + k * iqr_val
        
        # Identify outliers
        outliers = (distances < lower_bound) | (distances > upper_bound)
        
        return outliers
    
    def isolation_forest_outliers(self, spectra, contamination=0.05):
        """
        Detect outliers using Isolation Forest.
        
        Parameters:
        -----------
        spectra : numpy.ndarray
            Spectral data array
        contamination : float
            Expected proportion of outliers
            
        Returns:
        --------
        numpy.ndarray
            Boolean array indicating outliers
        """
        # Create and fit Isolation Forest model
        clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        
        # Reduce dimensionality for performance if too many features
        if spectra.shape[1] > 100:
            # Sample wavelengths
            indices = np.linspace(0, spectra.shape[1]-1, 100, dtype=int)
            reduced_spectra = spectra[:, indices]
        else:
            reduced_spectra = spectra
            
        clf.fit(reduced_spectra)
        
        # Predict outliers (-1 for outliers, 1 for inliers)
        predictions = clf.predict(reduced_spectra)
        
        # Convert to boolean array (True for outliers)
        outliers = predictions == -1
        
        return outliers
    
    def lof_outliers(self, spectra, contamination=0.05):
        """
        Detect outliers using Local Outlier Factor.
        
        Parameters:
        -----------
        spectra : numpy.ndarray
            Spectral data array
        contamination : float
            Expected proportion of outliers
            
        Returns:
        --------
        numpy.ndarray
            Boolean array indicating outliers
        """
        # Create and fit LOF model
        clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
        
        # Reduce dimensionality for performance if too many features
        if spectra.shape[1] > 100:
            # Sample wavelengths
            indices = np.linspace(0, spectra.shape[1]-1, 100, dtype=int)
            reduced_spectra = spectra[:, indices]
        else:
            reduced_spectra = spectra
            
        # Predict outliers (-1 for outliers, 1 for inliers)
        predictions = clf.fit_predict(reduced_spectra)
        
        # Convert to boolean array (True for outliers)
        outliers = predictions == -1
        
        return outliers
    
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
    
    def signal_to_noise_assessment(self):
        """
        Assess signal-to-noise ratio in original and augmented spectral data.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with SNR assessment results
        """
        print("Performing signal-to-noise assessment...")
        
        # Storage for results
        snr_results = {
            'original': {},
            'augmented': {}
        }
        
        # Add categories for augmentation methods
        augmentation_methods = self.identify_augmentation_methods()
        for method in augmentation_methods:
            snr_results[method] = {}
        
        # Define SNR calculation function
        def calculate_snr(spectra):
            # Method 1: Use Savitzky-Golay filter to separate signal and noise
            smoothed = np.array([savgol_filter(spectrum, 11, 3) for spectrum in spectra])
            noise = spectra - smoothed
            
            # Calculate SNR for each sample (ratio of signal power to noise power)
            signal_power = np.mean(smoothed**2, axis=1)
            noise_power = np.mean(noise**2, axis=1)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))  # Add small epsilon to avoid division by zero
            
            return snr
        
        # Calculate SNR for original data
        original_snr = calculate_snr(self.original_spectra)
        snr_results['original']['mean_snr'] = np.mean(original_snr)
        snr_results['original']['std_snr'] = np.std(original_snr)
        snr_results['original']['min_snr'] = np.min(original_snr)
        snr_results['original']['max_snr'] = np.max(original_snr)
        
        # Calculate SNR for all augmented data
        augmented_snr = calculate_snr(self.augmented_spectra)
        snr_results['augmented']['mean_snr'] = np.mean(augmented_snr)
        snr_results['augmented']['std_snr'] = np.std(augmented_snr)
        snr_results['augmented']['min_snr'] = np.min(augmented_snr)
        snr_results['augmented']['max_snr'] = np.max(augmented_snr)
        
        # Calculate SNR by augmentation method
        for method in augmentation_methods:
            # Get indices for this method
            method_indices = []
            for i, row_name in enumerate(self.augmented_only['Row_names']):
                if method.upper() in row_name:
                    method_indices.append(i)
            
            if method_indices:
                method_snr = augmented_snr[method_indices]
                snr_results[method]['mean_snr'] = np.mean(method_snr)
                snr_results[method]['std_snr'] = np.std(method_snr)
                snr_results[method]['min_snr'] = np.min(method_snr)
                snr_results[method]['max_snr'] = np.max(method_snr)
                snr_results[method]['samples'] = len(method_indices)
        
        # Create summary DataFrame
        summary_data = []
        for category, results in snr_results.items():
            if results:
                summary_data.append({
                    'Category': category,
                    'Mean_SNR': results.get('mean_snr', 0),
                    'Std_SNR': results.get('std_snr', 0),
                    'Min_SNR': results.get('min_snr', 0),
                    'Max_SNR': results.get('max_snr', 0),
                    'Samples': results.get('samples', len(original_snr) if category == 'original' else len(augmented_snr) if category == 'augmented' else 0)
                })
        
        snr_summary = pd.DataFrame(summary_data)
        
        # Save results
        snr_file = os.path.join(self.qc_dir, 'signal_to_noise_assessment.csv')
        snr_summary.to_csv(snr_file, index=False)
        
        print(f"Signal-to-noise assessment results saved to: {snr_file}")
        print(snr_summary)
        
        return snr_results
    
    def band_specific_validation(self):
        """
        Perform band-specific validation to ensure spectral features are preserved.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with band-specific validation results
        """
        print("Performing band-specific validation...")
        
        # Storage for results
        band_results = {}
        
        # Calculate region statistics for original data
        original_region_stats = {}
        for region, indices in self.region_indices.items():
            if len(indices) > 0:
                region_data = self.original_spectra[:, indices]
                
                original_region_stats[region] = {
                    'mean': np.mean(region_data),
                    'std': np.std(region_data),
                    'min': np.min(region_data),
                    'max': np.max(region_data),
                    'range': np.max(region_data) - np.min(region_data)
                }
        
        # Calculate key vegetation indices for original data
        ndvi_orig = self.calculate_ndvi(self.original_spectra)
        pri_orig = self.calculate_pri(self.original_spectra)
        reip_orig = self.calculate_reip(self.original_spectra)
        
        original_indices = {
            'NDVI': {
                'mean': np.mean(ndvi_orig),
                'std': np.std(ndvi_orig),
                'min': np.min(ndvi_orig),
                'max': np.max(ndvi_orig)
            },
            'PRI': {
                'mean': np.mean(pri_orig),
                'std': np.std(pri_orig),
                'min': np.min(pri_orig),
                'max': np.max(pri_orig)
            },
            'REIP': {
                'mean': np.mean(reip_orig),
                'std': np.std(reip_orig),
                'min': np.min(reip_orig),
                'max': np.max(reip_orig)
            }
        }
        
        # Process augmented data by method
        augmentation_methods = self.identify_augmentation_methods()
        
        for aug_method in ['augmented'] + augmentation_methods:
            # Get spectra for this method
            if aug_method == 'augmented':
                method_spectra = self.augmented_spectra
            else:
                # Get indices for this method
                method_indices = []
                for i, row_name in enumerate(self.augmented_only['Row_names']):
                    if aug_method.upper() in row_name:
                        method_indices.append(i)
                
                if not method_indices:
                    continue
                
                method_spectra = self.augmented_spectra[method_indices]
            
            # Calculate region statistics
            method_region_stats = {}
            for region, indices in self.region_indices.items():
                if len(indices) > 0:
                    region_data = method_spectra[:, indices]
                    
                    method_region_stats[region] = {
                        'mean': np.mean(region_data),
                        'std': np.std(region_data),
                        'min': np.min(region_data),
                        'max': np.max(region_data),
                        'range': np.max(region_data) - np.min(region_data),
                        'diff_from_orig': np.mean(region_data) - original_region_stats[region]['mean'],
                        'pct_diff': (np.mean(region_data) - original_region_stats[region]['mean']) / original_region_stats[region]['mean'] * 100
                    }
            
            # Calculate vegetation indices
            ndvi = self.calculate_ndvi(method_spectra)
            pri = self.calculate_pri(method_spectra)
            reip = self.calculate_reip(method_spectra)
            
            method_indices_stats = {
                'NDVI': {
                    'mean': np.mean(ndvi),
                    'std': np.std(ndvi),
                    'min': np.min(ndvi),
                    'max': np.max(ndvi),
                    'diff_from_orig': np.mean(ndvi) - original_indices['NDVI']['mean'],
                    'pct_diff': (np.mean(ndvi) - original_indices['NDVI']['mean']) / original_indices['NDVI']['mean'] * 100
                },
                'PRI': {
                    'mean': np.mean(pri),
                    'std': np.std(pri),
                    'min': np.min(pri),
                    'max': np.max(pri),
                    'diff_from_orig': np.mean(pri) - original_indices['PRI']['mean'],
                    'pct_diff': (np.mean(pri) - original_indices['PRI']['mean']) / (original_indices['PRI']['mean'] + 1e-10) * 100
                },
                'REIP': {
                    'mean': np.mean(reip),
                    'std': np.std(reip),
                    'min': np.min(reip),
                    'max': np.max(reip),
                    'diff_from_orig': np.mean(reip) - original_indices['REIP']['mean'],
                    'pct_diff': (np.mean(reip) - original_indices['REIP']['mean']) / original_indices['REIP']['mean'] * 100
                }
            }
            
            # Store results
            band_results[aug_method] = {
                'regions': method_region_stats,
                'indices': method_indices_stats
            }
        
        # Also store original data
        band_results['original'] = {
            'regions': original_region_stats,
            'indices': original_indices
        }
        
        # Prepare summary DataFrame for regions
        region_summary = []
        for region in self.regions:
            for method in band_results:
                if region in band_results[method]['regions']:
                    region_summary.append({
                        'Method': method,
                        'Region': region,
                        'Mean': band_results[method]['regions'][region]['mean'],
                        'Std': band_results[method]['regions'][region]['std'],
                        'Min': band_results[method]['regions'][region]['min'],
                        'Max': band_results[method]['regions'][region]['max'],
                        'Diff_from_Original': band_results[method]['regions'][region].get('diff_from_orig', 0),
                        'Pct_Diff': band_results[method]['regions'][region].get('pct_diff', 0)
                    })
        
        region_df = pd.DataFrame(region_summary)
        
        # Prepare summary DataFrame for indices
        index_summary = []
        for index_name in ['NDVI', 'PRI', 'REIP']:
            for method in band_results:
                index_summary.append({
                    'Method': method,
                    'Index': index_name,
                    'Mean': band_results[method]['indices'][index_name]['mean'],
                    'Std': band_results[method]['indices'][index_name]['std'],
                    'Min': band_results[method]['indices'][index_name]['min'],
                    'Max': band_results[method]['indices'][index_name]['max'],
                    'Diff_from_Original': band_results[method]['indices'][index_name].get('diff_from_orig', 0),
                    'Pct_Diff': band_results[method]['indices'][index_name].get('pct_diff', 0)
                })
        
        index_df = pd.DataFrame(index_summary)
        
        # Save results
        region_file = os.path.join(self.qc_dir, 'band_specific_regions.csv')
        index_file = os.path.join(self.qc_dir, 'vegetation_indices.csv')
        
        region_df.to_csv(region_file, index=False)
        index_df.to_csv(index_file, index=False)
        
        print(f"Band-specific validation results saved to: {region_file} and {index_file}")
        
        return band_results
    
    def calculate_ndvi(self, spectra):
        """Calculate Normalized Difference Vegetation Index."""
        # Find closest wavelengths to 670 (Red) and 800 (NIR)
        red_idx = np.argmin(np.abs(self.wavelengths - 670))
        nir_idx = np.argmin(np.abs(self.wavelengths - 800))
        
        red = spectra[:, red_idx]
        nir = spectra[:, nir_idx]
        
        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-10)  # Add small epsilon to avoid division by zero
        
        return ndvi
    
    def calculate_pri(self, spectra):
        """Calculate Photochemical Reflectance Index."""
        # Find closest wavelengths to 531 and 570
        w531_idx = np.argmin(np.abs(self.wavelengths - 531))
        w570_idx = np.argmin(np.abs(self.wavelengths - 570))
        
        r531 = spectra[:, w531_idx]
        r570 = spectra[:, w570_idx]
        
        # Calculate PRI
        pri = (r570 - r531) / (r570 + r531 + 1e-10)
        
        return pri
    
    def calculate_reip(self, spectra):
        """Calculate Red Edge Inflection Point using linear interpolation."""
        # Find indices for red edge region (670-780nm)
        red_edge_indices = np.where((self.wavelengths >= 670) & (self.wavelengths <= 780))[0]
        
        reip_values = []
        
        for spectrum in spectra:
            red_edge = spectrum[red_edge_indices]
            wavelengths_re = self.wavelengths[red_edge_indices]
            
            # Find first derivative
            derivative = np.diff(red_edge) / np.diff(wavelengths_re)
            
            # Find maximum of first derivative
            max_deriv_idx = np.argmax(derivative)
            
            # Find wavelength corresponding to maximum derivative
            if max_deriv_idx < len(wavelengths_re) - 1:
                reip = wavelengths_re[max_deriv_idx] + (wavelengths_re[max_deriv_idx + 1] - wavelengths_re[max_deriv_idx]) / 2
            else:
                reip = wavelengths_re[max_deriv_idx]
            
            reip_values.append(reip)
        
        return np.array(reip_values)
    
    def range_checks(self):
        """
        Perform range checks to verify data falls within physically plausible limits.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with range check results
        """
        print("Performing range checks...")
        
        # Define physical limits for reflectance data
        reflectance_constraints = (0, 1)  # Valid reflectance range (0-100%)
        
        # Define spectral region constraints with operator
        region_constraints = {
            'green_peak': ('Green', 'Red', '>'),  # Green reflectance > Red reflectance
            'nir_plateau': ('NIR', 'Red', '>'),  # NIR reflectance > Red reflectance
            'water_absorption': ('Water_Absorption1', 'NIR', '<'),  # Water absorption band < NIR
            'swir_check': ('SWIR1', 'Visible', '<')  # SWIR typically less than visible
        }
        
        # Storage for results
        range_results = {}
        
        # Process original and augmented data
        for data_type, spectra in [('original', self.original_spectra), ('augmented', self.augmented_spectra)]:
            total_samples = len(spectra)
            valid_counts = {}
            
            # Check valid reflectance range
            min_vals = np.min(spectra, axis=1)
            max_vals = np.max(spectra, axis=1)
            valid_range = np.logical_and(min_vals >= reflectance_constraints[0], 
                                        max_vals <= reflectance_constraints[1])
            valid_counts['valid_reflectance'] = np.sum(valid_range)
            
            # Check specific spectral features
            for constraint, (region1, region2, operator) in region_constraints.items():
                
                if region1 in self.region_indices and region2 in self.region_indices:
                    # Calculate mean reflectance in each region
                    region1_mean = np.mean(spectra[:, self.region_indices[region1]], axis=1)
                    region2_mean = np.mean(spectra[:, self.region_indices[region2]], axis=1)
                    
                    # Apply the constraint
                    if operator == '>':
                        valid = region1_mean > region2_mean
                    elif operator == '<':
                        valid = region1_mean < region2_mean
                    else:  # Equal
                        valid = np.isclose(region1_mean, region2_mean)
                    
                    valid_counts[constraint] = np.sum(valid)
            
            # Store results with percentages
            range_results[data_type] = {
                'total_samples': total_samples,
                'constraints': {k: {'count': v, 'percentage': (v / total_samples * 100)} 
                               for k, v in valid_counts.items()}
            }
        
        # Process by augmentation method
        augmentation_methods = self.identify_augmentation_methods()
        for method in augmentation_methods:
            # Get indices for this method
            method_indices = []
            for i, row_name in enumerate(self.augmented_only['Row_names']):
                if method.upper() in row_name:
                    method_indices.append(i)
            
            if not method_indices:
                continue
            
            method_spectra = self.augmented_spectra[method_indices]
            total_samples = len(method_spectra)
            valid_counts = {}
            
            # Check valid reflectance range
            min_vals = np.min(method_spectra, axis=1)
            max_vals = np.max(method_spectra, axis=1)
            valid_range = np.logical_and(min_vals >= reflectance_constraints[0], 
                                        max_vals <= reflectance_constraints[1])
            valid_counts['valid_reflectance'] = np.sum(valid_range)
            
            # Check specific spectral features
            for constraint, (region1, region2, operator) in region_constraints.items():
                
                if region1 in self.region_indices and region2 in self.region_indices:
                    # Calculate mean reflectance in each region
                    region1_mean = np.mean(method_spectra[:, self.region_indices[region1]], axis=1)
                    region2_mean = np.mean(method_spectra[:, self.region_indices[region2]], axis=1)
                    
                    # Apply the constraint
                    if operator == '>':
                        valid = region1_mean > region2_mean
                    elif operator == '<':
                        valid = region1_mean < region2_mean
                    else:  # Equal
                        valid = np.isclose(region1_mean, region2_mean)
                    
                    valid_counts[constraint] = np.sum(valid)
            
            # Store results with percentages
            range_results[method] = {
                'total_samples': total_samples,
                'constraints': {k: {'count': v, 'percentage': (v / total_samples * 100)} 
                               for k, v in valid_counts.items()}
            }
        
        # Create summary DataFrame
        summary_data = []
        for category, results in range_results.items():
            for constraint, values in results['constraints'].items():
                summary_data.append({
                    'Category': category,
                    'Constraint': constraint,
                    'Valid_Count': values['count'],
                    'Total_Samples': results['total_samples'],
                    'Percentage': values['percentage']
                })
        
        range_summary = pd.DataFrame(summary_data)
        
        # Save results
        range_file = os.path.join(self.qc_dir, 'range_checks.csv')
        range_summary.to_csv(range_file, index=False)
        
        print(f"Range check results saved to: {range_file}")
        
        return range_results
    
    def run_all_qc(self):
        """
        Run all quality control checks and generate an integrated report.
        
        Returns:
        --------
        dict
            Dictionary containing all QC results
        """
        print("\nRunning comprehensive quality control pipeline...\n")
        start_time = time.time()
        
        # Run all QC methods
        outlier_results = self.detect_outliers(method='combined')
        snr_results = self.signal_to_noise_assessment()
        band_results = self.band_specific_validation()
        range_results = self.range_checks()
        
        # Combine all results
        all_results = {
            'outliers': outlier_results,
            'snr': snr_results,
            'band_validation': band_results,
            'range_checks': range_results
        }
        
        # Generate integrated HTML report
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
            Dictionary containing all QC results
        """
        report_file = os.path.join(self.qc_dir, 'integrated_qc_report.html')
        
        with open(report_file, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Spectral Data Quality Control Report</title>
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
        .good { color: green; }
        .moderate { color: orange; }
        .poor { color: red; }
    </style>
</head>
<body>
    <h1>Spectral Data Quality Control Report</h1>
    <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
    
    <div class="summary">
        <h2>Overview</h2>
        <p>This report presents comprehensive quality control results for augmented spectral data.</p>
    </div>
""")
            
            # Outlier Detection Section
            f.write("""
    <h2>1. Outlier Detection</h2>
    <p>Detection of statistical outliers in the spectral data.</p>
    <table>
        <tr>
            <th>Category</th>
            <th>Total Samples</th>
            <th>Outliers</th>
            <th>Percentage</th>
            <th>Assessment</th>
        </tr>
""")
            
            for category, results in all_results['outliers'].items():
                if 'total' in results:
                    percentage = results.get('percentage', 0)
                    
                    # Determine quality class
                    if percentage < 2:
                        quality_class = "good"
                        assessment = "Excellent: Very few outliers"
                    elif percentage < 5:
                        quality_class = "moderate"
                        assessment = "Good: Acceptable number of outliers"
                    else:
                        quality_class = "poor"
                        assessment = "Moderate: Higher than expected outliers"
                    
                    f.write(f"""
        <tr>
            <td>{category}</td>
            <td>{results['total']}</td>
            <td>{results['outliers']}</td>
            <td class="{quality_class}">{percentage:.2f}%</td>
            <td>{assessment}</td>
        </tr>
""")
            
            f.write("</table>")
            
            # Signal-to-Noise Assessment Section
            f.write("""
    <h2>2. Signal-to-Noise Assessment</h2>
    <p>Evaluation of signal-to-noise ratio (SNR) in the spectral data.</p>
    <table>
        <tr>
            <th>Category</th>
            <th>Mean SNR (dB)</th>
            <th>Min SNR (dB)</th>
            <th>Max SNR (dB)</th>
            <th>Assessment</th>
        </tr>
""")
            
            for category, results in all_results['snr'].items():
                if 'mean_snr' in results:
                    mean_snr = results['mean_snr']
                    
                    # Determine quality class
                    if mean_snr > 30:
                        quality_class = "good"
                        assessment = "Excellent: High SNR"
                    elif mean_snr > 20:
                        quality_class = "moderate"
                        assessment = "Good: Acceptable SNR"
                    else:
                        quality_class = "poor"
                        assessment = "Moderate: Lower SNR"
                    
                    f.write(f"""
        <tr>
            <td>{category}</td>
            <td class="{quality_class}">{mean_snr:.2f}</td>
            <td>{results['min_snr']:.2f}</td>
            <td>{results['max_snr']:.2f}</td>
            <td>{assessment}</td>
        </tr>
""")
            
            f.write("</table>")
            
            # Band-Specific Validation Section - Vegetation Indices
            f.write("""
    <h2>3. Band-Specific Validation</h2>
    <h3>3.1 Vegetation Indices</h3>
    <p>Comparison of key vegetation indices between original and augmented data.</p>
    <table>
        <tr>
            <th>Method</th>
            <th>Index</th>
            <th>Mean</th>
            <th>% Diff from Original</th>
            <th>Assessment</th>
        </tr>
""")
            
            original_indices = all_results['band_validation']['original']['indices']
            
            for method, results in all_results['band_validation'].items():
                if method != 'original':
                    for index_name, values in results['indices'].items():
                        pct_diff = values.get('pct_diff', 0)
                        
                        # Determine quality class
                        if abs(pct_diff) < 5:
                            quality_class = "good"
                            assessment = "Excellent preservation"
                        elif abs(pct_diff) < 10:
                            quality_class = "moderate"
                            assessment = "Good preservation"
                        else:
                            quality_class = "poor"
                            assessment = "Moderate preservation"
                        
                        f.write(f"""
        <tr>
            <td>{method}</td>
            <td>{index_name}</td>
            <td>{values['mean']:.4f}</td>
            <td class="{quality_class}">{pct_diff:.2f}%</td>
            <td>{assessment}</td>
        </tr>
""")
            
            f.write("</table>")
            
            # Range Checks Section
            f.write("""
    <h2>4. Range Checks</h2>
    <p>Verification that spectral data adheres to physical constraints.</p>
    <table>
        <tr>
            <th>Category</th>
            <th>Constraint</th>
            <th>Valid %</th>
            <th>Assessment</th>
        </tr>
""")
            
            for category, results in all_results['range_checks'].items():
                for constraint, values in results['constraints'].items():
                    percentage = values['percentage']
                    
                    # Determine quality class
                    if percentage > 95:
                        quality_class = "good"
                        assessment = "Excellent: High compliance"
                    elif percentage > 85:
                        quality_class = "moderate"
                        assessment = "Good: Acceptable compliance"
                    else:
                        quality_class = "poor"
                        assessment = "Moderate: Lower compliance"
                    
                    f.write(f"""
        <tr>
            <td>{category}</td>
            <td>{constraint}</td>
            <td class="{quality_class}">{percentage:.2f}%</td>
            <td>{assessment}</td>
        </tr>
""")
            
            f.write("</table>")
            
            # Overall Assessment
            f.write("""
    <h2>Overall Quality Assessment</h2>
""")
            
            # Calculate overall quality scores
            outlier_score = 0
            snr_score = 0
            band_score = 0
            range_score = 0
            
            # Outlier score
            if 'augmented' in all_results['outliers']:
                outlier_pct = all_results['outliers']['augmented'].get('percentage', 0)
                outlier_score = 100 - outlier_pct * 10  # Penalize more for higher outlier percentage
            
            # SNR score
            if 'augmented' in all_results['snr']:
                mean_snr = all_results['snr']['augmented'].get('mean_snr', 0)
                snr_score = min(100, mean_snr * 3)  # Scale SNR (typically 0-40 dB) to 0-100
            
            # Band validation score - use mean absolute percentage diff across indices
            pct_diffs = []
            for method, results in all_results['band_validation'].items():
                if method != 'original':
                    for index_name, values in results['indices'].items():
                        pct_diffs.append(abs(values.get('pct_diff', 0)))
            
            if pct_diffs:
                mean_pct_diff = np.mean(pct_diffs)
                band_score = max(0, 100 - mean_pct_diff * 5)  # Penalize based on percentage difference
            
            # Range check score - use mean of constraint percentages
            constraint_pcts = []
            if 'augmented' in all_results['range_checks']:
                for constraint, values in all_results['range_checks']['augmented']['constraints'].items():
                    constraint_pcts.append(values['percentage'])
            
            if constraint_pcts:
                range_score = np.mean(constraint_pcts)
            
            # Calculate overall score
            overall_score = np.mean([outlier_score, snr_score, band_score, range_score])
            
            # Generate assessment text
            if overall_score >= 90:
                assessment_class = "good"
                assessment_text = """
                <p class="good">The augmented spectral data is of <strong>excellent quality</strong> and meets the 
                highest standards for plant spectroscopy research. The statistical properties and 
                physical characteristics of the original spectra are preserved with high fidelity, 
                making the augmented dataset a valuable resource for machine learning and analysis.</p>
                
                <p>The data exhibits low outlier rates, high signal-to-noise ratio, excellent preservation 
                of spectral features and vegetation indices, and strong adherence to physical constraints 
                of plant reflectance spectra.</p>
                """
            elif overall_score >= 80:
                assessment_class = "moderate"
                assessment_text = """
                <p class="moderate">The augmented spectral data is of <strong>good quality</strong> and suitable for 
                most research applications. The statistical properties and physical characteristics 
                of the original spectra are well-preserved, though minor deviations exist in some aspects.</p>
                
                <p>The data shows acceptable outlier rates, good signal-to-noise ratio, good preservation 
                of spectral features, and adequate adherence to physical constraints. Any deviations are 
                within acceptable ranges for most analytical purposes.</p>
                """
            else:
                assessment_class = "poor"
                assessment_text = """
                <p class="poor">The augmented spectral data is of <strong>adequate quality</strong> but may 
                benefit from refinement for sensitive analyses. While usable for many purposes, 
                certain analyses might be affected by the observed deviations from original data properties.</p>
                
                <p>Consider adjusting the augmentation parameters to better preserve spectral characteristics, 
                particularly to improve signal-to-noise ratio and adherence to physical constraints.</p>
                """
            
            f.write(f"""
    <div class="summary">
        <h3>Overall Quality Score: <span class="{assessment_class}">{overall_score:.1f}/100</span></h3>
        {assessment_text}
        <p>Quality Scores by Component:</p>
        <ul>
            <li>Outlier Quality: {outlier_score:.1f}/100</li>
            <li>Signal-to-Noise Quality: {snr_score:.1f}/100</li>
            <li>Spectral Features Quality: {band_score:.1f}/100</li>
            <li>Physical Constraints Quality: {range_score:.1f}/100</li>
        </ul>
    </div>
""")
            
            f.write("""
    <hr>
    <p><em>Report generated by Spectral QC Module</em></p>
</body>
</html>
""")
        
        print(f"Integrated QC report saved to: {report_file}")


def run_spectral_qc():
    """Run the spectral quality control pipeline."""
    # File paths
    original_path = r"C:\Users\ms\Desktop\hyper\data\hyper_full_w.csv"
    augmented_path = r"C:\Users\ms\Desktop\hyper\output\augment\augmented_spectral_data.csv"
    output_dir = r"C:\Users\ms\Desktop\hyper\output\augment\hyper"
    
    # Create and run QC
    qc = SpectralQC(original_path, augmented_path, output_dir)
    
    # Run all QC checks
    results = qc.run_all_qc()
    
    return results


if __name__ == "__main__":
    run_spectral_qc()