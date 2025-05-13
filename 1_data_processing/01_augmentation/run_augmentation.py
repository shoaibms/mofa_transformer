"""
Comprehensive Data Augmentation Pipeline for Spectral and Molecular Features Data

This module provides a unified pipeline for augmenting and validating both spectral and
Molecular features datasets. The pipeline integrates multiple components:
1. Spectral data augmentation using various signal processing techniques
2. Molecular features data augmentation using scaling and mixing approaches
3. Cross-modality validation to ensure consistency between data types
4. Divergence analysis to evaluate the quality of augmented data

The pipeline is configurable via JSON config files or command-line arguments
and generates comprehensive reports for analysis results.
"""

import os
import sys
import time
import logging
import argparse
import json
import pandas as pd
import numpy as np
import shutil
import traceback
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Append custom module path and import modules
sys.path.append(r"C:\Users\ms\Desktop\hyper\augment")
try:
    from spectral_augmentation_V7 import SpectralAugmentation
    from metabolite_augmentation import MolecularFeaturesAugmentation
    from cross_modality_validation import CrossModalityValidation
    from divergence_analysis import DivergenceAnalysis
    print("Successfully imported all required modules")
except Exception as e:
    print(f"ERROR IMPORTING MODULES: {e}")
    traceback.print_exc()
    sys.exit(1)


def create_sample_config(output_path):
    """Create a sample configuration file with all available options."""
    config = {
        # Input paths
        "spectral_original_path": r"C:\Users\ms\Desktop\hyper\data\hyper_full_w.csv",
        "molecular_features_original_path": r"C:\Users\ms\Desktop\hyper\data\n_p_r2.csv",
        
        # Output paths
        "output_dir": r"C:\Users\ms\Desktop\hyper\output\augment\main_pipeline",
        "spectral_output_dir": r"C:\Users\ms\Desktop\hyper\output\augment\main_pipeline\spectral",
        "molecular_features_output_dir": r"C:\Users\ms\Desktop\hyper\output\augment\main_pipeline\molecular_features",
        "validation_output_dir": r"C:\Users\ms\Desktop\hyper\output\augment\main_pipeline\validation",
        
        # Augmentation parameters
        "spectral_augmentation": {
            "enabled": True,
            "methods": ["gaussian_process", "spectral_mixup", "peak_preserving_warp", 
                        "reflectance_scaling", "add_band_specific_noise", 
                        "additive_mixup", "multiplicative_mixup"]
        },
        "molecular_features_augmentation": {
            "enabled": True,
            "scale_copies": 5, 
            "mix_copies": 2
        },
        
        # Validation parameters
        "validation": {
            "enabled": True,
            "divergence_analysis": True,
            "cross_modality": True
        },
        
        # Reporting parameters
        "reporting": {
            "generate_plots": True,
            "save_intermediate": True,
            "report_format": "html"
        },
        
        # General parameters
        "random_seed": 42,
        "n_jobs": -1,  # Use all available cores
        "verbose": True
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Sample configuration file created at: {output_path}")


class MainPipeline:
    """
    Unified workflow manager for joint spectral and metabolite data augmentation.
    """
    
    def __init__(self, config_file=None, **kwargs):
        """
        Initialize the pipeline with configuration settings.
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = self._load_config(config_file, kwargs)
        self._setup_paths()
        self._setup_logging()
        self.logger.info("Pipeline initialized with configuration:")
        for key, value in self.config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}: {json.dumps(value)}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def _load_config(self, config_file, kwargs):
        """Load and merge configuration from file and keyword arguments."""
        default_config = {
            "spectral_original_path": r"C:\Users\ms\Desktop\hyper\data\hyper_full_w.csv",
            "molecular_features_original_path": r"C:\Users\ms\Desktop\hyper\data\n_p_r2.csv",
            "output_dir": r"C:\Users\ms\Desktop\hyper\output\augment\main_pipeline",
            "spectral_output_dir": r"C:\Users\ms\Desktop\hyper\output\augment\main_pipeline\spectral",
            "molecular_features_output_dir": r"C:\Users\ms\Desktop\hyper\output\augment\main_pipeline\molecular_features",
            "validation_output_dir": r"C:\Users\ms\Desktop\hyper\output\augment\main_pipeline\validation",
            "spectral_augmentation": {
                "enabled": True,
                "methods": ["gaussian_process", "spectral_mixup", "peak_preserving_warp", 
                            "reflectance_scaling", "add_band_specific_noise", 
                            "additive_mixup", "multiplicative_mixup"]
            },
            "molecular_features_augmentation": {
                "enabled": True,
                "scale_copies": 5, 
                "mix_copies": 2
            },
            "validation": {
                "enabled": True,
                "divergence_analysis": True,
                "cross_modality": True
            },
            "reporting": {
                "generate_plots": True,
                "save_intermediate": True,
                "report_format": "html"
            },
            "random_seed": 42,
            "n_jobs": -1,
            "verbose": True
        }
        
        config = default_config.copy()
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")
        
        config.update(kwargs)
        return config
    
    def _setup_paths(self):
        """Set up all required directories."""
        os.makedirs(self.config["output_dir"], exist_ok=True)
        for key in ["spectral_output_dir", "molecular_features_output_dir", "validation_output_dir"]:
            if key in self.config:
                os.makedirs(self.config[key], exist_ok=True)
                print(f"Created directory: {self.config[key]}")
        self.log_dir = os.path.join(self.config["output_dir"], "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.report_dir = os.path.join(self.config["output_dir"], "reports")
        os.makedirs(self.report_dir, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        self.logger = logging.getLogger("MainPipeline")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        console_handler = logging.StreamHandler()
        console_level = logging.INFO if self.config.get("verbose", True) else logging.WARNING
        console_handler.setLevel(console_level)
        log_file = os.path.join(self.log_dir, f"pipeline_{self.timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def save_configuration(self):
        """Save the current configuration to a JSON file."""
        config_file = os.path.join(self.config["output_dir"], f"config_{self.timestamp}.json")
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"Configuration saved to: {config_file}")
            return config_file
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return None
    
    def run_spectral_augmentation(self):
        """Run the spectral data augmentation pipeline."""
        self.logger.info("Starting spectral data augmentation...")
        try:
            start_time = time.time()
            spectral_augmentor = SpectralAugmentation(
                input_file=self.config["spectral_original_path"],
                output_dir=self.config["spectral_output_dir"]
            )
            self.logger.info("Running spectral augmentation...")
            _ = spectral_augmentor.generate_augmented_dataset()
            self.spectral_augmented_path = os.path.join(
                self.config["spectral_output_dir"], 'augmented_spectral_data.csv'
            )
            if os.path.exists(self.spectral_augmented_path):
                self.logger.info(f"Verified spectral augmented file exists at: {self.spectral_augmented_path}")
            else:
                self.logger.warning(f"Expected augmented file not found at: {self.spectral_augmented_path}")
                for file in os.listdir(self.config["spectral_output_dir"]):
                    if file.endswith('.csv'):
                        self.logger.info(f"Found alternative CSV file: {file}")
                        self.spectral_augmented_path = os.path.join(self.config["spectral_output_dir"], file)
                        break
            end_time = time.time()
            self.logger.info(f"Spectral augmentation completed in {end_time - start_time:.2f} seconds")
            self.logger.info(f"Augmented spectral data saved to: {self.spectral_augmented_path}")
            return self.spectral_augmented_path
        except Exception as e:
            self.logger.error(f"Error in spectral augmentation: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_molecular_features_augmentation(self):
        """Run the molecular features data augmentation pipeline."""
        self.logger.info("Starting molecular features data augmentation...")
        try:
            start_time = time.time()
            scale_copies = self.config["molecular_features_augmentation"].get("scale_copies", 5)
            mix_copies = self.config["molecular_features_augmentation"].get("mix_copies", 2)
            molecular_features_augmentor = MolecularFeaturesAugmentation(
                input_file=self.config["molecular_features_original_path"],
                output_dir=self.config["molecular_features_output_dir"]
            )
            self.logger.info(f"Running molecular features augmentation with scale_copies={scale_copies}, mix_copies={mix_copies}...")
            _ = molecular_features_augmentor.generate_optimized_dataset(
                scale_copies=scale_copies,
                mix_copies=mix_copies
            )
            self.molecular_features_augmented_path = os.path.join(
                self.config["molecular_features_output_dir"], 'augmented_molecular_features_data.csv'
            )
            if os.path.exists(self.molecular_features_augmented_path):
                self.logger.info(f"Verified molecular features augmented file exists at: {self.molecular_features_augmented_path}")
            else:
                self.logger.warning(f"Expected augmented file not found at: {self.molecular_features_augmented_path}")
                for file in os.listdir(self.config["molecular_features_output_dir"]):
                    if file.endswith('.csv'):
                        self.logger.info(f"Found alternative CSV file: {file}")
                        self.molecular_features_augmented_path = os.path.join(self.config["molecular_features_output_dir"], file)
                        break
            end_time = time.time()
            self.logger.info(f"Molecular features augmentation completed in {end_time - start_time:.2f} seconds")
            self.logger.info(f"Augmented molecular features data saved to: {self.molecular_features_augmented_path}")
            return self.molecular_features_augmented_path
        except Exception as e:
            self.logger.error(f"Error in molecular features augmentation: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_cross_modality_validation(self):
        """Run cross-modality validation between spectral and molecular features data."""
        self.logger.info("Starting cross-modality validation...")
        try:
            if not hasattr(self, 'spectral_augmented_path') or not hasattr(self, 'molecular_features_augmented_path'):
                self.logger.warning("Augmented data paths not found. Using existing files...")
                self.spectral_augmented_path = r"C:\Users\ms\Desktop\hyper\output\augment\hyper\augmented_spectral_data.csv"
                self.molecular_features_augmented_path = r"C:\Users\ms\Desktop\hyper\output\augment\molecular_features\root\augmented_molecular_features_data.csv"
                self.logger.info(f"Using spectral data from: {self.spectral_augmented_path}")
                self.logger.info(f"Using molecular features data from: {self.molecular_features_augmented_path}")
                if not os.path.exists(self.spectral_augmented_path):
                    self.logger.error(f"Spectral data file not found: {self.spectral_augmented_path}")
                    return None
                if not os.path.exists(self.molecular_features_augmented_path):
                    self.logger.error(f"Molecular features data file not found: {self.molecular_features_augmented_path}")
                    return None
            start_time = time.time()
            validation_dir = os.path.join(self.config["validation_output_dir"], "cross_modality")
            os.makedirs(validation_dir, exist_ok=True)
            self.logger.info("Initializing cross-modality validator...")
            validator = CrossModalityValidation(
                spectral_original_path=self.config["spectral_original_path"],
                spectral_augmented_path=self.spectral_augmented_path,
                molecular_features_original_path=self.config["molecular_features_original_path"],
                molecular_features_augmented_path=self.molecular_features_augmented_path,
                output_dir=validation_dir
            )
            self.logger.info("Running cross-modality validation...")
            validation_results = validator.run_all_validations()
            end_time = time.time()
            self.logger.info(f"Cross-modality validation completed in {end_time - start_time:.2f} seconds")
            self.logger.info(f"Validation results saved to: {validation_dir}")
            return validation_results
        except Exception as e:
            self.logger.error(f"Error in cross-modality validation: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def run_divergence_analysis(self):
        """Run divergence analysis for spectral and molecular features data."""
        self.logger.info("Starting divergence analysis...")
        try:
            if not hasattr(self, 'spectral_augmented_path') or not hasattr(self, 'molecular_features_augmented_path'):
                self.logger.warning("Augmented data paths not found. Using existing files...")
                self.spectral_augmented_path = r"C:\Users\ms\Desktop\hyper\output\augment\hyper\augmented_spectral_data.csv"
                self.molecular_features_augmented_path = r"C:\Users\ms\Desktop\hyper\output\augment\molecular_features\root\augmented_molecular_features_data.csv"
                self.logger.info(f"Using spectral data from: {self.spectral_augmented_path}")
                self.logger.info(f"Using molecular features data from: {self.molecular_features_augmented_path}")
                if not os.path.exists(self.spectral_augmented_path):
                    self.logger.error(f"Spectral data file not found: {self.spectral_augmented_path}")
                    return None
                if not os.path.exists(self.molecular_features_augmented_path):
                    self.logger.error(f"Molecular features data file not found: {self.molecular_features_augmented_path}")
                    return None
            start_time = time.time()
            divergence_dir = os.path.join(self.config["validation_output_dir"], "divergence")
            os.makedirs(divergence_dir, exist_ok=True)
            self.logger.info("Initializing divergence analyzer...")
            analyzer = DivergenceAnalysis(
                spectral_original_path=self.config["spectral_original_path"],
                spectral_augmented_path=self.spectral_augmented_path,
                molecular_features_original_path=self.config["molecular_features_original_path"],
                molecular_features_augmented_path=self.molecular_features_augmented_path,
                output_dir=divergence_dir
            )
            self.logger.info("Running divergence analysis...")
            divergence_results = analyzer.run_all_analyses()
            end_time = time.time()
            self.logger.info(f"Divergence analysis completed in {end_time - start_time:.2f} seconds")
            self.logger.info(f"Divergence results saved to: {divergence_dir}")
            return divergence_results
        except Exception as e:
            self.logger.error(f"Error in divergence analysis: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def copy_visualization_files(self):
        """Copy key visualization files to the report directory for inclusion in the final report."""
        self.logger.info("Copying visualization files...")
        viz_dir = os.path.join(self.report_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        paths_to_check = [
            os.path.join(self.config["validation_output_dir"], "cross_modality", "plots"),
            os.path.join(self.config["validation_output_dir"], "divergence", "plots"),
            os.path.join(self.config["spectral_output_dir"], "quality_control", "plots"),
            os.path.join(self.config["molecular_features_output_dir"], "quality_control", "plots")
        ]
        copied_files = []
        for path in paths_to_check:
            if os.path.exists(path):
                subdir_name = os.path.basename(os.path.dirname(path))
                target_dir = os.path.join(viz_dir, subdir_name)
                os.makedirs(target_dir, exist_ok=True)
                for file in os.listdir(path):
                    if file.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                        source = os.path.join(path, file)
                        target = os.path.join(target_dir, file)
                        shutil.copy2(source, target)
                        copied_files.append(os.path.join(subdir_name, file))
        self.logger.info(f"Copied {len(copied_files)} visualization files to report directory")
        return copied_files
    
    def generate_final_report(self, results):
        """Generate comprehensive final report."""
        self.logger.info("Generating final report...")
        
        # Hardcoded color values instead of importing from colour.py
        colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'tertiary': '#2980b9',
            'success': 'green',
            'warning': 'orange',
            'error': 'red',
            'background': '#f8f9fa',
            'border': '#ddd',
            'header_bg': '#f2f2f2',
            'row_alt': '#f9f9f9'
        }
        
        try:
            report_file = os.path.join(self.report_dir, f"final_report_{self.timestamp}.html")
            with open(report_file, 'w') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Augmentation Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: {colors['primary']}; }}
        h2 {{ color: {colors['secondary']}; margin-top: 30px; }}
        h3 {{ color: {colors['tertiary']}; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid {colors['border']}; padding: 8px; text-align: left; }}
        th {{ background-color: {colors['header_bg']}; }}
        tr:nth-child(even) {{ background-color: {colors['row_alt']}; }}
        .summary {{ margin: 20px 0; padding: 15px; background-color: {colors['background']}; border-radius: 5px; }}
        .success {{ color: {colors['success']}; }}
        .warning {{ color: {colors['warning']}; }}
        .error {{ color: {colors['error']}; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid {colors['border']}; }}
    </style>
</head>
<body>
    <h1>Comprehensive Augmentation Pipeline Report</h1>
    <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    
    <div class="summary">
        <h2>Pipeline Overview</h2>
        <p>This report presents results from the comprehensive molecular features and spectral data augmentation pipeline.</p>
""")
                f.write("<h3>Configuration</h3>\n<ul>\n")
                for key, value in self.config.items():
                    if isinstance(value, dict):
                        f.write(f"<li><strong>{key}</strong>: {json.dumps(value)}</li>\n")
                    else:
                        f.write(f"<li><strong>{key}</strong>: {value}</li>\n")
                f.write("</ul>\n")
                f.write("""
    <h2>Spectral Augmentation Summary</h2>
    <p>Summary of the spectral data augmentation process.</p>
""")
                if hasattr(self, 'spectral_augmented_path') and self.spectral_augmented_path:
                    try:
                        orig_data = pd.read_csv(self.config["spectral_original_path"])
                        aug_data = pd.read_csv(self.spectral_augmented_path)
                        f.write(f"""
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Original Samples</td>
            <td>{len(orig_data)}</td>
        </tr>
        <tr>
            <td>Augmented Samples</td>
            <td>{len(aug_data)}</td>
        </tr>
        <tr>
            <td>Multiplication Factor</td>
            <td>{len(aug_data) / len(orig_data):.2f}x</td>
        </tr>
        <tr>
            <td>Output Path</td>
            <td>{self.spectral_augmented_path}</td>
        </tr>
        <tr>
            <td>Status</td>
            <td class="success">Completed Successfully</td>
        </tr>
    </table>
""")
                    except Exception as e:
                        f.write(f"<p class='error'>Error loading augmented data: {e}</p>")
                else:
                    f.write("<p class='warning'>Using pre-existing spectral augmented data.</p>")
                    spectral_path = r"C:\Users\ms\Desktop\hyper\output\augment\hyper\augmented_spectral_data.csv"
                    if os.path.exists(spectral_path):
                        try:
                            orig_data = pd.read_csv(self.config["spectral_original_path"])
                            aug_data = pd.read_csv(spectral_path)
                            f.write(f"""
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Original Samples</td>
            <td>{len(orig_data)}</td>
        </tr>
        <tr>
            <td>Augmented Samples</td>
            <td>{len(aug_data)}</td>
        </tr>
        <tr>
            <td>Multiplication Factor</td>
            <td>{len(aug_data) / len(orig_data):.2f}x</td>
        </tr>
        <tr>
            <td>Output Path</td>
            <td>{spectral_path}</td>
        </tr>
        <tr>
            <td>Status</td>
            <td class="success">Using Existing Data</td>
        </tr>
    </table>
""")
                        except Exception as e:
                            f.write(f"<p class='error'>Error loading existing spectral data: {e}</p>")
                    else:
                        f.write(f"<p class='error'>No spectral data found at: {spectral_path}</p>")
                
                f.write("""
    <h2>Molecular Features Augmentation Summary</h2>
    <p>Summary of the molecular features data augmentation process.</p>
""")
                if hasattr(self, 'molecular_features_augmented_path') and self.molecular_features_augmented_path:
                    try:
                        orig_data = pd.read_csv(self.config["molecular_features_original_path"])
                        aug_data = pd.read_csv(self.molecular_features_augmented_path)
                        f.write(f"""
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Original Samples</td>
            <td>{len(orig_data)}</td>
        </tr>
        <tr>
            <td>Augmented Samples</td>
            <td>{len(aug_data)}</td>
        </tr>
        <tr>
            <td>Multiplication Factor</td>
            <td>{len(aug_data) / len(orig_data):.2f}x</td>
        </tr>
        <tr>
            <td>Output Path</td>
            <td>{self.molecular_features_augmented_path}</td>
        </tr>
        <tr>
            <td>Status</td>
            <td class="success">Completed Successfully</td>
        </tr>
    </table>
""")
                    except Exception as e:
                        f.write(f"<p class='error'>Error loading augmented data: {e}</p>")
                else:
                    f.write("<p class='warning'>Using pre-existing molecular features augmented data.</p>")
                    molecular_features_path = r"C:\Users\ms\Desktop\hyper\output\augment\molecular_features\root\augmented_molecular_features_data.csv"
                    if os.path.exists(molecular_features_path):
                        try:
                            orig_data = pd.read_csv(self.config["molecular_features_original_path"])
                            aug_data = pd.read_csv(molecular_features_path)
                            f.write(f"""
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Original Samples</td>
            <td>{len(orig_data)}</td>
        </tr>
        <tr>
            <td>Augmented Samples</td>
            <td>{len(aug_data)}</td>
        </tr>
        <tr>
            <td>Multiplication Factor</td>
            <td>{len(aug_data) / len(orig_data):.2f}x</td>
        </tr>
        <tr>
            <td>Output Path</td>
            <td>{molecular_features_path}</td>
        </tr>
        <tr>
            <td>Status</td>
            <td class="success">Using Existing Data</td>
        </tr>
    </table>
""")
                        except Exception as e:
                            f.write(f"<p class='error'>Error loading existing molecular features data: {e}</p>")
                    else:
                        f.write(f"<p class='error'>No molecular features data found at: {molecular_features_path}</p>")
                
                f.write("""
    <h2>Validation Summary</h2>
    <p>Summary of the cross-modality validation and divergence analysis.</p>
""")
                if 'cross_modality' in results and results['cross_modality']:
                    f.write("<h3>Cross-Modality Validation</h3>")
                    f.write("<p class='success'>Cross-modality validation completed successfully.</p>")
                    cross_modal_dir = os.path.join(self.config["validation_output_dir"], "cross_modality")
                    report_path = os.path.join(cross_modal_dir, 'cross_modality_report.html')
                    if os.path.exists(report_path):
                        f.write(f"<p>Detailed report available at: <a href='{report_path}'>{report_path}</a></p>")
                else:
                    f.write("<p class='warning'>Cross-modality validation was not run or did not complete.</p>")
                
                if 'divergence' in results and results['divergence']:
                    f.write("<h3>Divergence Analysis</h3>")
                    f.write("<p class='success'>Divergence analysis completed successfully.</p>")
                    divergence_dir = os.path.join(self.config["validation_output_dir"], "divergence")
                    report_path = os.path.join(divergence_dir, 'divergence_summary.html')
                    if os.path.exists(report_path):
                        f.write(f"<p>Detailed report available at: <a href='{report_path}'>{report_path}</a></p>")
                else:
                    f.write("<p class='warning'>Divergence analysis was not run or did not complete.</p>")
                
                success_count = sum([
                    1 if hasattr(self, 'spectral_augmented_path') and self.spectral_augmented_path else 0,
                    1 if hasattr(self, 'molecular_features_augmented_path') and self.molecular_features_augmented_path else 0,
                    1 if 'cross_modality' in results and results['cross_modality'] else 0,
                    1 if 'divergence' in results and results['divergence'] else 0
                ])
                total_components = 4
                if success_count == total_components:
                    f.write("<p class='success'>All pipeline components completed successfully.</p>")
                elif success_count >= total_components / 2:
                    f.write("<p class='warning'>Some pipeline components completed successfully, but others encountered issues.</p>")
                else:
                    f.write("<p class='error'>Multiple pipeline components encountered issues.</p>")
                
                f.write("<table>\n<tr><th>Component</th><th>Status</th></tr>\n")
                f.write(f"<tr><td>Spectral Augmentation</td><td class=\"{'success' if hasattr(self, 'spectral_augmented_path') and self.spectral_augmented_path else 'error'}\">{'Completed' if hasattr(self, 'spectral_augmented_path') and self.spectral_augmented_path else 'Failed'}</td></tr>\n")
                f.write(f"<tr><td>Molecular Features Augmentation</td><td class=\"{'success' if hasattr(self, 'molecular_features_augmented_path') and self.molecular_features_augmented_path else 'error'}\">{'Completed' if hasattr(self, 'molecular_features_augmented_path') and self.molecular_features_augmented_path else 'Failed'}</td></tr>\n")
                f.write(f"<tr><td>Cross-Modality Validation</td><td class=\"{'success' if 'cross_modality' in results and results['cross_modality'] else 'warning'}\">{'Completed' if 'cross_modality' in results and results['cross_modality'] else 'Not Run or Failed'}</td></tr>\n")
                f.write(f"<tr><td>Divergence Analysis</td><td class=\"{'success' if 'divergence' in results and results['divergence'] else 'warning'}\">{'Completed' if 'divergence' in results and results['divergence'] else 'Not Run or Failed'}</td></tr>\n")
                f.write(f"""
    <h3>Output Directories</h3>
    <table>
        <tr>
            <th>Component</th>
            <th>Output Directory</th>
        </tr>
""")
                f.write(f"<tr><td>Main Output</td><td>{self.config['output_dir']}</td></tr>\n")
                f.write(f"<tr><td>Spectral Output</td><td>{self.config['spectral_output_dir']}</td></tr>\n")
                f.write(f"<tr><td>Molecular Features Output</td><td>{self.config['molecular_features_output_dir']}</td></tr>\n")
                f.write(f"<tr><td>Validation Output</td><td>{self.config['validation_output_dir']}</td></tr>\n")
                f.write(f"<tr><td>Report Directory</td><td>{self.report_dir}</td></tr>\n")
                f.write(f"<tr><td>Log Directory</td><td>{self.log_dir}</td></tr>\n")
                f.write("</table>\n")
                f.write(f"""
    <hr>
    <p><em>Report generated by Comprehensive Augmentation Pipeline</em></p>
</body>
</html>
""")
            self.logger.info(f"Final report generated: {report_file}")
            return report_file
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")
            self.logger.error(traceback.format_exc())
            return None


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description='Run the comprehensive augmentation pipeline')
    
    # Main parameters
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    parser.add_argument('--output-dir', type=str, 
                        default=r"C:\Users\ms\Desktop\hyper\output\augment\main_pipeline",
                        help='Main output directory')
    parser.add_argument('--create-config', type=str, default=None,
                        help='Create a sample configuration file at the specified path and exit')
    
    # Input paths
    parser.add_argument('--spectral-original', type=str,
                        default=r"C:\Users\ms\Desktop\hyper\data\hyper_full_w.csv",
                        help='Path to original spectral data')
    parser.add_argument('--molecular-features-original', type=str,
                        default=r"C:\Users\ms\Desktop\hyper\data\n_p_r2.csv",
                        help='Path to original molecular features data')
    
    # Existing augmented data paths (to skip augmentation steps)
    parser.add_argument('--spectral-augmented', type=str, default=None,
                        help='Path to existing augmented spectral data (skips augmentation step)')
    parser.add_argument('--molecular-features-augmented', type=str, default=None,
                        help='Path to existing augmented molecular features data (skips augmentation step)')
    
    # Component control
    parser.add_argument('--skip-spectral', action='store_true',
                        help='Skip spectral augmentation')
    parser.add_argument('--skip-molecular-features', action='store_true',
                        help='Skip molecular features augmentation')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip all validation steps')
    parser.add_argument('--validation-only', action='store_true',
                        help='Skip augmentation and only run validation steps on existing data')
    parser.add_argument('--force-paths', action='store_true',
                        help='Force using hardcoded paths for augmented data files')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config(args.create_config)
        return 0
    
    config = {}
    config["output_dir"] = args.output_dir
    config["spectral_output_dir"] = os.path.join(args.output_dir, "spectral")
    config["molecular_features_output_dir"] = os.path.join(args.output_dir, "molecular_features")
    config["validation_output_dir"] = os.path.join(args.output_dir, "validation")
    
    config["spectral_original_path"] = args.spectral_original
    config["molecular_features_original_path"] = args.molecular_features_original
    
    config["spectral_augmentation"] = {"enabled": not (args.skip_spectral or args.validation_only)}
    config["molecular_features_augmentation"] = {"enabled": not (args.skip_molecular_features or args.validation_only)}
    config["validation"] = {"enabled": not args.skip_validation}
    
    if args.validation_only or args.force_paths:
        config["spectral_augmented_path"] = r"C:\Users\ms\Desktop\hyper\output\augment\hyper\augmented_spectral_data.csv"
        config["molecular_features_augmented_path"] = r"C:\Users\ms\Desktop\hyper\output\augment\molecular_features\root\augmented_molecular_features_data.csv"
        print("Using existing augmented data paths:")
        print(f"  Spectral: {config['spectral_augmented_path']}")
        print(f"  Molecular Features: {config['molecular_features_augmented_path']}")
    
    if args.spectral_augmented:
        config["spectral_augmented_path"] = args.spectral_augmented
        if not args.skip_spectral:
            config["spectral_augmentation"]["enabled"] = False
            print("Using existing augmented spectral data. Spectral augmentation will be skipped.")
    
    if args.molecular_features_augmented:
        config["molecular_features_augmented_path"] = args.molecular_features_augmented
        if not args.skip_molecular_features:
            config["molecular_features_augmentation"]["enabled"] = False
            print("Using existing augmented molecular features data. Molecular features augmentation will be skipped.")
    
    try:
        pipeline = MainPipeline(config_file=args.config, **config)
        results = {}
        if pipeline.config["spectral_augmentation"].get("enabled", False):
            results['spectral'] = pipeline.run_spectral_augmentation()
        if pipeline.config["molecular_features_augmentation"].get("enabled", False):
            results['molecular_features'] = pipeline.run_molecular_features_augmentation()
        if pipeline.config["validation"].get("enabled", False):
            results['cross_modality'] = pipeline.run_cross_modality_validation()
            results['divergence'] = pipeline.run_divergence_analysis()
        report_path = pipeline.generate_final_report(results)
        return 0 if all(results.values()) else 1
    except Exception as e:
        print(f"Error running pipeline: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
