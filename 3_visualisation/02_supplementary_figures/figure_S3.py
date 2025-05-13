"""
This script performs a comprehensive pre-analysis of hyperspectral data.
It includes several data quality assessment steps:
1.  Data loading and preparation: Reads hyperspectral data from a CSV file,
    separates metadata and spectral data, handles missing values, and extracts
    wavelength information.
2.  Core quality assessment: Performs outlier detection using IQR, Modified
    Z-Score, and optionally Local Outlier Factor (LOF). Calculates basic
    quality metrics like standard deviation, SNR approximation, and inter-band
    correlation.
3.  Normality assessment: Tests for normality in each spectral band using the
    Shapiro-Wilk test.
4.  Baseline and stability assessment: Analyzes baseline trends using Theil-Sen
    slopes and Mann-Whitney U test to assess the need for baseline correction.
5.  Derivative analysis: Calculates Savitzky-Golay derivatives (1st and 2nd)
    to inspect spectral features.
6.  Composite figure generation: Creates a multi-panel figure summarizing the key
    findings from the quality assessment steps.

The script is configurable through constants defined in the CONFIGURATION
SECTION, allowing users to specify input/output paths, enable/disable
specific analysis steps, and customize plotting parameters.
It also includes robust logging and saves a detailed JSON report of the
pipeline results.
"""
# --- Standard Libraries ---
import os
import json
import logging
from datetime import datetime
import warnings

# --- Data Handling ---
import pandas as pd
import numpy as np

# --- Scientific Computing ---
from scipy import stats
from scipy import signal
from scipy.stats import iqr, theilslopes, mannwhitneyu

# --- Machine Learning ---
from sklearn.neighbors import LocalOutlierFactor

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Optional Progress Bar ---
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

######################################################
############ CONFIGURATION SECTION
######################################################

# --- Input File ---
INPUT_FILE_HYPER = r"C:\\Users\\ms\\Desktop\\hyper\\data\\hyper_full_w.csv"
SPECTRAL_DATA_START_COL = 10

# --- Output Directory ---
OUTPUT_DIR = r"C:\\Users\\ms\\Desktop\\hyper\\output\\data_quality"

# --- Analysis Control Flags ---
RUN_LOF_OUTLIERS = True
RUN_NORMALITY_TESTS = True
RUN_BASELINE_ASSESSMENT = True
RUN_DERIVATIVE_ANALYSIS = True

# --- Plotting Styles ---
PLOT_CMAP = 'BuGn'
PLOT_TITLE_FONTSIZE = 18
PLOT_AXIS_FONTSIZE = 16
PLOT_LEGEND_FONTSIZE = 14
PLOT_DPI = 300

# --- Constants ---
ALPHA = 0.05
MOD_ZSCORE_THRESHOLD = 3.5
IQR_MULTIPLIER = 1.5
BASELINE_SLOPE_THRESHOLD_FACTOR = 1.5
BASELINE_ABS_DERIV_THRESHOLD = 0.01
BASELINE_DERIV_IQR_THRESHOLD = 0.005
BASELINE_SIG_TREND_PERC_THRESHOLD = 50.0
DERIVATIVE_WINDOW_LENGTH = 5
DERIVATIVE_POLYORDER = 2

######################################################
############ LOGGING SETUP & HELPERS
######################################################

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir,
        f"analysis_log_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)-8s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info("Logging initialized.")
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="scipy.stats._morestats"
    )
    return logger

def set_plot_styles():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.titlesize': PLOT_TITLE_FONTSIZE + 4,
        'axes.titlesize': PLOT_TITLE_FONTSIZE,
        'axes.labelsize': PLOT_AXIS_FONTSIZE,
        'xtick.labelsize': PLOT_AXIS_FONTSIZE - 1,
        'ytick.labelsize': PLOT_AXIS_FONTSIZE - 1,
        'legend.fontsize': PLOT_LEGEND_FONTSIZE,
        'lines.linewidth': 1.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    logging.debug("Plot styles updated.")

def save_report_to_json(report_dict, filename):
    try:
        def default_converter(o):
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32,
                                np.float64)):
                return float(f"{o:.4g}")
            elif isinstance(o, np.ndarray):
                return (f"Array shape {o.shape}, "
                        f"Median {np.median(o):.3g}, IQR {iqr(o):.3g}")
            elif isinstance(o, (pd.Timestamp, datetime)):
                return o.isoformat()
            elif isinstance(o, np.bool_):
                return bool(o)
            elif pd.isna(o):
                return None
            elif isinstance(o, (pd.Series, pd.DataFrame)):
                return f"Pandas {type(o).__name__} shape {o.shape}"
            raise TypeError(
                f"Object of type {o.__class__.__name__} is not JSON serializable"
            )
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=4, default=default_converter)
        logging.info(f"Report saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Failed to save report to {filename}: {e}")

def extract_wavelengths_from_wide_format(df_columns):
    wavelengths = []
    col_names_str = [str(c) for c in df_columns]
    for col in col_names_str:
        try:
            potential_num = col.split('_')[-1]
            potential_num = potential_num.replace('nm', '').strip()
            wavelengths.append(float(potential_num))
        except ValueError:
            logging.warning(
                f"Could not convert '{col}'. Wavelength axis might use index."
            )
            wavelengths = list(range(len(df_columns)))
            break
    return np.array(sorted(wavelengths))

def get_iterator(iterable, desc="Processing"):
    return tqdm(iterable, desc=desc, unit="item", leave=False) \
        if TQDM_AVAILABLE else iterable

#############################################################################
# --- STEP 1: LOAD & PREPARE DATA ---
#############################################################################
def load_and_prepare_data(file_path, spec_start_col):
    """Loads data, separates metadata/spectral, handles NaNs, extracts wavelengths."""
    logging.info(f"--- 1. Loading and Preparing Data from: {file_path} ---")
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Original data shape: {data.shape}")
        if data.shape[1] <= spec_start_col:
            raise ValueError(
                f"Data has only {data.shape[1]} columns. "
                f"Check SPECTRAL_DATA_START_COL ({spec_start_col})."
            )

        metadata = data.iloc[:, :spec_start_col]
        spectral_data_raw = data.iloc[:, spec_start_col:]
        logging.info(
            f"Separated: Metadata {metadata.shape}, Spectral {spectral_data_raw.shape}"
        )

        initial_nan_count = spectral_data_raw.isna().sum().sum()
        if initial_nan_count > 0:
            logging.warning(f"Found {initial_nan_count} NaN values in spectral data.")
            original_cols = spectral_data_raw.shape[1]
            spectral_data = spectral_data_raw.dropna(
                axis=1, thresh=int(0.9 * len(spectral_data_raw))
            )
            dropped_cols = original_cols - spectral_data.shape[1]
            if dropped_cols > 0:
                logging.warning(
                    f"Dropped {dropped_cols} spectral columns due to >10% NaNs."
                )
            if spectral_data.isna().sum().sum() > 0:
                logging.warning(
                    f"Filling remaining {spectral_data.isna().sum().sum()} "
                    f"NaNs with column medians."
                )
                spectral_data = spectral_data.fillna(spectral_data.median())
            logging.info(f"NaNs handled. Final spectral data shape: {spectral_data.shape}")
            if spectral_data.empty:
                raise ValueError("Spectral data is empty after handling NaNs.")
            nan_handling_info = (
                f"{initial_nan_count} NaNs initially. "
                f"Dropped {dropped_cols} cols (>10% NaN). "
                f"Remaining filled with median."
            )
        else:
            spectral_data = spectral_data_raw.copy()
            nan_handling_info = "No NaNs found in spectral data."
            logging.info(nan_handling_info)

        wavelengths = extract_wavelengths_from_wide_format(spectral_data.columns)

        prep_info = {
            "original_shape": list(data.shape),
            "metadata_shape": list(metadata.shape),
            "spectral_shape_raw": list(spectral_data_raw.shape),
            "spectral_shape_final": list(spectral_data.shape),
            "nan_handling": nan_handling_info,
            "num_wavelengths": len(wavelengths)
        }

        logging.info("--- Data Loading and Preparation Finished ---")
        return metadata, spectral_data, wavelengths, prep_info

    except FileNotFoundError:
        logging.error(f"FATAL: Input file not found at {file_path}")
        raise
    except ValueError as ve:
        logging.error(f"FATAL: ValueError during data preparation: {ve}")
        raise
    except Exception as e:
        logging.error(
            "FATAL: An unexpected error occurred during data "
            f"loading/preparation: {e}", exc_info=True
        )
        raise

#############################################################################
# --- STEP 2: CORE QUALITY ASSESSMENT ---
#############################################################################

def perform_core_quality_assessment(spectral_data, wavelengths, output_dir_base):
    """Performs outlier detection, calculates basic metrics, returns data for plotting."""
    logging.info("--- 2. Performing Core Quality Assessment ---")
    output_dir_core = os.path.join(output_dir_base, "Core_Quality_Assessment")
    os.makedirs(output_dir_core, exist_ok=True)
    results = {'outliers': {}, 'basic_metrics': {}, 'plotting_data': {}, 'files': {}}
    outlier_indices = set()

    logging.info("--- 2a. Outlier Detection ---")
    outlier_summary = {}
    methods_run = []
    valid_samples = spectral_data.shape[0]
    if valid_samples < 2:
        logging.warning("Skipping outlier detection: Insufficient samples.")
        results['outliers']['status'] = "Skipped - Insufficient samples"
    else:
        try:
            methods_run.append("IQR")
            Q1 = spectral_data.quantile(0.25)
            Q3 = spectral_data.quantile(0.75)
            IQR_vals = Q3 - Q1
            IQR_vals[IQR_vals == 0] = 1e-9
            iqr_mask = ((spectral_data < (Q1 - IQR_MULTIPLIER * IQR_vals)) |
                        (spectral_data > (Q3 + IQR_MULTIPLIER * IQR_vals))).any(axis=1)
            iqr_indices = set(spectral_data.index[iqr_mask])
            outlier_summary['iqr'] = {
                'count': len(iqr_indices),
                'indices': sorted(list(iqr_indices))
            }
            outlier_indices.update(iqr_indices)
            logging.info(f"IQR Outlier Detection found {len(iqr_indices)} potential outliers.")
        except Exception as e:
            logging.error(f"Error in IQR: {e}")
            outlier_summary['iqr'] = {'count': 'Error', 'error': str(e)}
        try:
            methods_run.append("Modified Z-Score")
            median_vals = spectral_data.median(axis=0)
            mad = np.median(np.abs(spectral_data - median_vals), axis=0)
            mad[mad == 0] = 1e-9
            with np.errstate(divide='ignore', invalid='ignore'):
                modified_zscore = 0.6745 * (spectral_data - median_vals) / mad
            zscore_mask = (np.abs(modified_zscore) > MOD_ZSCORE_THRESHOLD).any(axis=1)
            zscore_indices = set(spectral_data.index[zscore_mask])
            outlier_summary['mod_zscore'] = {
                'count': len(zscore_indices),
                'indices': sorted(list(zscore_indices))
            }
            outlier_indices.update(zscore_indices)
            logging.info(f"Modified Z-Score found {len(zscore_indices)} potential outliers.")
        except Exception as e:
            logging.error(f"Error in Mod Z-score: {e}")
            outlier_summary['mod_zscore'] = {'count': 'Error', 'error': str(e)}
        if RUN_LOF_OUTLIERS:
            try:
                methods_run.append("LOF")
                n_neighbors_lof = min(max(5, int(0.05 * valid_samples)), 20)
                if valid_samples > n_neighbors_lof:
                    lof = LocalOutlierFactor(
                        n_neighbors=n_neighbors_lof,
                        contamination='auto',
                        novelty=False
                    )
                    logging.info(f"Running LOF with n_neighbors={n_neighbors_lof}...")
                    lof_preds = lof.fit_predict(spectral_data)
                    lof_mask = pd.Series(lof_preds == -1, index=spectral_data.index)
                    lof_indices = set(spectral_data.index[lof_mask])
                    outlier_summary['lof'] = {
                        'count': len(lof_indices),
                        'indices': sorted(list(lof_indices)),
                        'n_neighbors': n_neighbors_lof
                    }
                    outlier_indices.update(lof_indices)
                    logging.info(f"LOF found {len(lof_indices)} potential outliers.")
                else:
                    logging.warning(
                        f"Skipping LOF: Samples ({valid_samples}) "
                        f"<= n_neighbors ({n_neighbors_lof})."
                    )
                    outlier_summary['lof'] = {
                        'count': 'Skipped',
                        'reason': (f'Samples {valid_samples} '
                                   f'<= n_neighbors {n_neighbors_lof}')
                    }
            except Exception as e:
                logging.error(f"Error in LOF: {e}", exc_info=True)
                outlier_summary['lof'] = {'count': 'Error', 'error': str(e)}
        else:
            logging.info("Skipping LOF as per config.")
            outlier_summary['lof'] = {
                'count': 'Skipped',
                'reason': 'Disabled by config'
            }
        results['outliers'] = {
            'methods_run': methods_run,
            'details_per_method': outlier_summary,
            'combined_unique_outliers': {
                'count': len(outlier_indices),
                'indices': sorted(list(outlier_indices))
            }
        }
        if outlier_indices:
            outlier_df = pd.DataFrame({
                'outlier_sample_index': sorted(list(outlier_indices))
            })
            outlier_csv_path = os.path.join(
                output_dir_core, 'core_outlier_indices.csv'
            )
            outlier_df.to_csv(outlier_csv_path, index=False)
            results['files']['outlier_indices_csv'] = os.path.basename(outlier_csv_path)

    logging.info("--- 2b. Calculating Basic Quality Metrics ---")
    metrics = {}
    std_devs = None
    corr_matrix_sampled = None
    sampled_tick_labels = None
    sampled_tick_locs = None
    if spectral_data.empty:
        results['basic_metrics']['status'] = "Skipped - Empty data"
    else:
        try:
            std_devs = spectral_data.std()
            metrics['mean_std_dev'] = std_devs.mean()
            metrics['median_std_dev'] = std_devs.median()
            signal_approx = spectral_data.mean(axis=0)
            noise_approx = std_devs.copy()
            noise_approx[noise_approx == 0] = 1e-9
            snr_per_band = signal_approx / noise_approx
            metrics['mean_snr_approx'] = snr_per_band.mean()
            metrics['median_snr_approx'] = snr_per_band.median()
            metrics['mean_spectral_diff'] = np.abs(
                np.diff(spectral_data.values, axis=1)
            ).mean()
            num_bands = spectral_data.shape[1]
            if num_bands > 100:
                step = max(1, num_bands // 100)
                sample_indices = np.arange(0, num_bands, step)
                corr_matrix_sampled = spectral_data.iloc[:, sample_indices].corr()
                sampled_tick_labels = spectral_data.columns[sample_indices]
                sampled_tick_locs = np.arange(len(sample_indices))
                logging.info("Sampled ~100 bands for correlation plot.")
            else:
                corr_matrix_sampled = spectral_data.corr()
                sampled_tick_labels = spectral_data.columns
                sampled_tick_locs = np.arange(num_bands)
                logging.info("Using full bands for correlation plot.")
            if num_bands > 500:
                sample_cols_metric = np.random.choice(
                    num_bands, 500, replace=False
                )
                corr_matrix_metric = spectral_data.iloc[:, sample_cols_metric].corr()
                logging.info("Sampled 500 bands for mean correlation metric.")
            else:
                corr_matrix_metric = corr_matrix_sampled \
                    if num_bands <= 100 else spectral_data.corr()
            mean_corr = corr_matrix_metric.values[
                np.triu_indices_from(corr_matrix_metric.values, k=1)
            ].mean()
            metrics['mean_interband_correlation'] = mean_corr
            results['basic_metrics'] = metrics
            logging.info("Basic quality metrics calculated.")
        except Exception as e:
            logging.error(f"Error calculating basic metrics: {e}", exc_info=True)
            results['basic_metrics']['status'] = "Error"
            results['basic_metrics']['error'] = str(e)

    results['plotting_data'] = {
        'outlier_summary': results['outliers'].get('details_per_method', {}),
        'methods_run_outliers': results['outliers'].get('methods_run', []),
        'std_devs': std_devs,
        'wavelengths': wavelengths,
        'corr_matrix_sampled': corr_matrix_sampled,
        'corr_tick_labels': sampled_tick_labels,
        'corr_tick_locs': sampled_tick_locs
    }
    logging.info("--- Core Quality Assessment Finished ---")
    return results

#############################################################################
# --- STEP 3: NORMALITY ASSESSMENT ---
#############################################################################

def perform_normality_assessment(spectral_data, output_dir_base):
    """Tests normality per band, calculates summary, returns data for plotting."""
    if not RUN_NORMALITY_TESTS:
        logging.warning("--- Skipping Normality Assessment ---")
        return {"status": "Skipped - Disabled by config"}
    logging.info("--- 3. Performing Normality Assessment (Shapiro-Wilk) ---")
    output_dir_normality = os.path.join(output_dir_base, "Normality_Assessment")
    os.makedirs(output_dir_normality, exist_ok=True)
    results = {'summary': {}, 'details': {}, 'plotting_data': {}, 'files': {}}
    if spectral_data.empty:
        logging.warning("Skipping normality: Empty data.")
        results['status'] = "Skipped - Empty data"
        return results

    spectral_cols = spectral_data.columns
    shapiro_results_list = []
    p_values_list = []
    logging.info(f"Testing normality for {len(spectral_cols)} spectral bands...")
    iterator = get_iterator(spectral_cols, desc="Shapiro-Wilk Tests")
    for col in iterator:
        data_clean = spectral_data[col].dropna()
        stat, p_val, is_norm, status = np.nan, np.nan, None, "Skipped"
        if len(data_clean) >= 3:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    stat, p_val = stats.shapiro(data_clean)
                is_norm = p_val > ALPHA
                status = "Tested"
                p_values_list.append(p_val)
            except Exception as e:
                logging.debug(f"Shapiro test failed for column {col}: {e}")
                status = f"Error: {e}"
        else:
            status = f"Skipped - Insufficient data ({len(data_clean)})"
            logging.debug(status + f" for column {col}")
        shapiro_results_list.append({
            'wavelength_column': col,
            'statistic': stat,
            'p_value': p_val,
            'is_normal': is_norm,
            'status': status
        })

    results_df = pd.DataFrame(shapiro_results_list)
    detailed_results_filename = os.path.join(
        output_dir_normality, 'normality_details_shapiro.csv'
    )
    results_df.to_csv(
        detailed_results_filename, index=False, float_format='%.4g'
    )
    results['files']['details_csv'] = os.path.basename(detailed_results_filename)
    logging.info(f"Detailed normality results saved to {detailed_results_filename}")

    total_bands = len(spectral_cols)
    tested_bands = len(p_values_list)
    normal_count = results_df[results_df['is_normal'] == True]['is_normal'].count() \
        if tested_bands > 0 else 0
    percent_normal = (normal_count / tested_bands * 100) \
        if tested_bands > 0 else 0
    error_count = results_df['status'].str.startswith('Error').sum()
    skipped_count = total_bands - tested_bands - error_count
    results['summary'] = {
        'test_used': 'Shapiro-Wilk',
        'alpha_level': ALPHA,
        'total_bands': total_bands,
        'bands_tested': tested_bands,
        'bands_normal (p>alpha)': normal_count,
        'percent_normal_of_tested': percent_normal,
        'bands_skipped_insufficient_data': skipped_count,
        'bands_error': error_count
    }
    logging.info(
        f"Normality summary: {percent_normal:.1f}% of tested bands assessed as normal."
    )

    spectral_data_samples = None
    sample_wavelength_names = []
    num_bands = len(spectral_cols)
    if num_bands > 0 and not spectral_data.empty:
        step = max(1, num_bands // 5)
        sample_indices = np.arange(0, num_bands, step)[:5]
        sample_wavelength_names = spectral_cols[sample_indices].tolist()
        spectral_data_samples = spectral_data[sample_wavelength_names]

    results['plotting_data'] = {
        'p_values': p_values_list,
        'spectral_data_samples': spectral_data_samples,
        'sample_wavelength_names': sample_wavelength_names,
        'results_df': results_df
    }
    logging.info("--- Normality Assessment Finished ---")
    return results

#############################################################################
# --- STEP 4: BASELINE & STABILITY ASSESSMENT ---
#############################################################################

def perform_robust_baseline_assessment(spectral_data, output_dir_base):
    """Assesses baseline trends/stability, returns data for plotting, includes enhanced summary."""
    if not RUN_BASELINE_ASSESSMENT:
        logging.warning("--- Skipping Baseline Assessment ---")
        return {"status": "Skipped - Disabled by config"}
    logging.info("--- 4. Performing Robust Baseline Assessment ---")
    output_dir_baseline = os.path.join(output_dir_base, "Baseline_Assessment")
    os.makedirs(output_dir_baseline, exist_ok=True)
    results = {'summary': {}, 'details': {}, 'plotting_data': {}, 'files': {}}
    if spectral_data.empty:
        logging.warning("Skipping baseline: Empty data.")
        results['status'] = "Skipped - Empty data"
        return results

    x_indices = np.arange(spectral_data.shape[1])
    trends_list = []
    all_sample_derivatives = []
    logging.info(f"Analyzing baseline trends for {spectral_data.shape[0]} samples...")
    iterator = get_iterator(spectral_data.iterrows(), desc="Baseline Assessment")
    for index, sample_reflectance in iterator:
        reflectance_values = sample_reflectance.values.astype(float)
        valid_mask = ~np.isnan(reflectance_values)
        slope, p_val = np.nan, np.nan
        status_trend, status_mw = "Skipped", "Skipped"
        if valid_mask.sum() >= 5:
            current_reflectance = reflectance_values[valid_mask]
            current_x = x_indices[valid_mask]
            try:
                slope, intercept, low_slope, high_slope = theilslopes(
                    current_reflectance, current_x
                )
                status_trend = "Calculated"
            except Exception as e:
                logging.debug(f"Theil-Sen failed for {index}: {e}")
                status_trend = f"Error: {e}"
            n_half = len(current_reflectance) // 2
            if n_half >= 5:
                try:
                    first_half = current_reflectance[:n_half]
                    second_half = current_reflectance[-n_half:]
                    if not np.array_equal(first_half, second_half):
                        stat_mw, p_val = mannwhitneyu(
                            first_half, second_half, alternative='two-sided'
                        )
                        status_mw = "Tested"
                    else:
                        status_mw = "Skipped - Identical halves"
                except ValueError as ve:
                    logging.debug(f"MW test error for {index}: {ve}")
                    status_mw = "Skipped - Test Error"
                except Exception as e:
                    logging.debug(f"MW test failed for {index}: {e}")
                    status_mw = f"Error: {e}"
            else:
                status_mw = "Skipped - Insufficient halves"
            if len(current_reflectance) >= 2:
                derivatives = np.diff(current_reflectance)
                all_sample_derivatives.extend(derivatives)
        trends_list.append({
            'sample_index': index,
            'theil_sen_slope': slope,
            'mann_whitney_p_value': p_val,
            'theil_sen_status': status_trend,
            'mann_whitney_status': status_mw
        })

    trends_df = pd.DataFrame(trends_list)
    detailed_trends_filename = os.path.join(
        output_dir_baseline, 'baseline_trends_details.csv'
    )
    trends_df.to_csv(
        detailed_trends_filename, index=False, float_format='%.6g'
    )
    results['files']['details_csv'] = os.path.basename(detailed_trends_filename)
    logging.info(f"Detailed baseline trends saved to {detailed_trends_filename}")

    summary_metrics = {}
    valid_slopes = trends_df['theil_sen_slope'].dropna()
    valid_p_vals = trends_df['mann_whitney_p_value'].dropna()
    if not valid_slopes.empty:
        slope_iqr_val = iqr(valid_slopes)
        abs_slope_threshold = BASELINE_SLOPE_THRESHOLD_FACTOR * slope_iqr_val \
            if slope_iqr_val > 1e-9 else np.percentile(np.abs(valid_slopes), 95)
        strong_slope_count = (np.abs(valid_slopes) > abs_slope_threshold).sum()
        strong_slope_percentage = (strong_slope_count / len(valid_slopes)) * 100 \
            if len(valid_slopes) > 0 else 0
        summary_metrics['trend_analysis'] = {
            'median_theil_sen_slope': valid_slopes.median(),
            'slope_iqr': slope_iqr_val,
            'slope_5th_percentile': np.percentile(valid_slopes, 5),
            'slope_95th_percentile': np.percentile(valid_slopes, 95),
            'strong_slope_threshold_abs_value': abs_slope_threshold,
            'strong_slope_threshold_defn': (
                f">{BASELINE_SLOPE_THRESHOLD_FACTOR} * IQR "
                "(or 95th Perc if IQR=0)"
            ),
            'strong_slope_sample_count': strong_slope_count,
            'strong_slope_sample_percentage': strong_slope_percentage,
            'percentage_significant_trends_mw':
                (valid_p_vals < ALPHA).mean() * 100
                if not valid_p_vals.empty else 'N/A'
        }
        logging.info(
            f"Baseline Trend Summary: Median Slope={valid_slopes.median():.3g}, "
            f"IQR={slope_iqr_val:.3g}, {strong_slope_percentage:.1f}% "
            "samples have strong slopes."
        )
    else:
        summary_metrics['trend_analysis'] = {'status': 'No valid slope data'}
        logging.warning("Could not calculate valid baseline trend statistics.")
    if all_sample_derivatives:
        all_sample_derivatives = np.array(all_sample_derivatives)
        all_abs_derivatives = np.abs(all_sample_derivatives)
        summary_metrics['baseline_stability'] = {
            'median_derivative_diff': np.median(all_sample_derivatives),
            'derivative_iqr': iqr(all_sample_derivatives),
            'median_absolute_derivative': np.median(all_abs_derivatives),
            'percentile95_abs_derivative': np.percentile(all_abs_derivatives, 95)
        }
        logging.info(
            f"Baseline Stability Summary: Median Abs Diff="
            f"{np.median(all_abs_derivatives):.3g}, "
            f"Derivative IQR={iqr(all_sample_derivatives):.3g}"
        )
    else:
        summary_metrics['baseline_stability'] = {
            'status': 'No derivatives calculated'
        }
        logging.warning("Could not calculate baseline stability statistics.")
    results['summary']['metrics'] = summary_metrics

    correction_needed = False
    decision_factors = {}
    try:
        trend_metrics = summary_metrics.get('trend_analysis', {})
        stability_metrics = summary_metrics.get('baseline_stability', {})
        if isinstance(trend_metrics, dict):
            med_slope = trend_metrics.get('median_theil_sen_slope', 0)
            strong_perc = trend_metrics.get('strong_slope_sample_percentage', 0)
            abs_thresh = trend_metrics.get('strong_slope_threshold_abs_value', np.inf)
            factor_slope_mag = abs(med_slope) > abs_thresh \
                if not np.isinf(abs_thresh) else False
            decision_factors['strong_median_trend'] = {
                'value': bool(factor_slope_mag),
                'details': f'MedianSlope={med_slope:.3g}, Threshold={abs_thresh:.3g}'
            }
            if factor_slope_mag:
                correction_needed = True
            if (isinstance(strong_perc, (int, float)) and
                    strong_perc > BASELINE_SIG_TREND_PERC_THRESHOLD):
                decision_factors['high_perc_strong_slopes'] = {
                    'value': True, 'details': f'Percentage={strong_perc:.1f}%'
                }
                correction_needed = True
            else:
                decision_factors['high_perc_strong_slopes'] = {
                    'value': False, 'details': f'Percentage={strong_perc}%'
                }
        if isinstance(stability_metrics, dict):
            med_abs_deriv = stability_metrics.get('median_absolute_derivative', 0)
            deriv_iqr_val = stability_metrics.get('derivative_iqr', 0)
            factor_large_offset = med_abs_deriv > BASELINE_ABS_DERIV_THRESHOLD
            decision_factors['large_median_offset_abs_deriv'] = {
                'value': bool(factor_large_offset),
                'details': f'MedianAbsDeriv={med_abs_deriv:.3g}'
            }
            if factor_large_offset:
                correction_needed = True
            factor_high_variability = deriv_iqr_val > BASELINE_DERIV_IQR_THRESHOLD
            decision_factors['high_baseline_variability_deriv_iqr'] = {
                'value': bool(factor_high_variability),
                'details': f'DerivativeIQR={deriv_iqr_val:.3g}'
            }
            if factor_high_variability:
                correction_needed = True
    except Exception as metric_e:
        logging.error(f"Error evaluating baseline decision: {metric_e}")
        correction_needed = 'Error'
        decision_factors['error'] = str(metric_e)
    results['summary']['assessment'] = {
        'correction_potentially_needed':
            bool(correction_needed) if isinstance(correction_needed, bool)
            else correction_needed,
        'heuristic_decision_factors': decision_factors,
        'note': 'Review factors and plots. Thresholds are heuristic.'
    }
    logging.info(
        "Baseline Correction Assessment: Potentially Needed = "
        f"{results['summary']['assessment']['correction_potentially_needed']}"
    )

    results['plotting_data'] = {
        'valid_slopes': valid_slopes,
        'all_sample_derivatives': all_sample_derivatives,
        'summary_metrics': summary_metrics
    }
    logging.info("--- Robust Baseline Assessment Finished ---")
    return results

#############################################################################
# --- STEP 5: DERIVATIVE ANALYSIS (SAVITZKY-GOLAY) ---
#############################################################################

def calculate_savitzky_golay_derivatives(spectral_data, wavelengths,
                                         output_dir_base):
    """Performs Sav-Gol derivative analysis, returns data for plotting."""
    if not RUN_DERIVATIVE_ANALYSIS:
        logging.warning("--- Skipping Derivative Analysis ---")
        return {"status": "Skipped - Disabled by config"}
    logging.info("--- 5. Performing Derivative Analysis (Savitzky-Golay) ---")
    output_dir_deriv = os.path.join(
        output_dir_base, "Derivative_Analysis_SG"
    )
    os.makedirs(output_dir_deriv, exist_ok=True)
    results = {'summary': {}, 'details': {}, 'plotting_data': {}, 'files': {}}
    if spectral_data.empty:
        logging.warning("Skipping SG derivatives: Empty data.")
        results['status'] = "Skipped - Empty data"
        return results

    window = DERIVATIVE_WINDOW_LENGTH
    poly = DERIVATIVE_POLYORDER
    if (not isinstance(window, int) or window % 2 == 0 or window < 3 or
            spectral_data.shape[1] < window or not isinstance(poly, int) or
            poly < 1):
        logging.error(
            f"Invalid SG params/data shape. Win={window}, Poly={poly}, "
            f"Bands={spectral_data.shape[1]}. Skipping."
        )
        results['status'] = "Skipped - Invalid params/data"
        return results

    all_first_derivs_sg = []
    all_second_derivs_sg = []
    metrics_list_sg = []
    logging.info(
        f"Calculating Sav-Gol derivatives (window={window}, polyorder={poly})..."
    )
    iterator = get_iterator(spectral_data.iterrows(), desc="Sav-Gol Derivatives")
    for index, sample_reflectance in iterator:
        reflectance_values = sample_reflectance.values.astype(float)
        if np.isnan(reflectance_values).any():
            logging.warning(f"NaNs in sample {index}, skipping SG.")
            continue
        try:
            first_deriv_sg = signal.savgol_filter(
                reflectance_values, window, poly, deriv=1, mode='interp'
            )
            second_deriv_sg = signal.savgol_filter(
                reflectance_values, window, poly, deriv=2, mode='interp'
            )
            all_first_derivs_sg.append(first_deriv_sg)
            all_second_derivs_sg.append(second_deriv_sg)
            metrics_list_sg.append({
                'sample_index': index,
                'first_deriv_sg_median': np.median(first_deriv_sg),
                'first_deriv_sg_iqr': iqr(first_deriv_sg),
                'second_deriv_sg_median': np.median(second_deriv_sg),
                'second_deriv_sg_iqr': iqr(second_deriv_sg),
                'first_deriv_sg_max_abs': np.max(np.abs(first_deriv_sg)),
                'second_deriv_sg_max_abs': np.max(np.abs(second_deriv_sg))
            })
        except Exception as e:
            logging.error(f"Error calculating SG derivative for {index}: {e}")
            continue

    if not all_first_derivs_sg:
        logging.error("No valid SG derivatives calculated.")
        results['status'] = "Error - No valid derivatives"
        return results

    all_first_derivs_sg = np.array(all_first_derivs_sg)
    all_second_derivs_sg = np.array(all_second_derivs_sg)
    metrics_df_sg = pd.DataFrame(metrics_list_sg)
    detailed_metrics_filename = os.path.join(
        output_dir_deriv, 'derivative_sg_details_per_sample.csv'
    )
    metrics_df_sg.to_csv(
        detailed_metrics_filename, index=False, float_format='%.6g'
    )
    results['files']['details_csv'] = os.path.basename(detailed_metrics_filename)

    median_first_deriv_sg = np.median(all_first_derivs_sg, axis=0)
    median_second_deriv_sg = np.median(all_second_derivs_sg, axis=0)
    first_deriv_sg_25th = np.percentile(all_first_derivs_sg, 25, axis=0)
    first_deriv_sg_75th = np.percentile(all_first_derivs_sg, 75, axis=0)
    second_deriv_sg_25th = np.percentile(all_second_derivs_sg, 25, axis=0)
    second_deriv_sg_75th = np.percentile(all_second_derivs_sg, 75, axis=0)
    results['summary'] = {
        'parameters': {'window_length': window, 'polyorder': poly},
        'first_derivative_stats': {
            'overall_median_of_medians':
                metrics_df_sg['first_deriv_sg_median'].median(),
            'overall_iqr_of_medians':
                iqr(metrics_df_sg['first_deriv_sg_median'].dropna()),
            'median_of_max_absolute':
                metrics_df_sg['first_deriv_sg_max_abs'].median()
        },
        'second_derivative_stats': {
            'overall_median_of_medians':
                metrics_df_sg['second_deriv_sg_median'].median(),
            'overall_iqr_of_medians':
                iqr(metrics_df_sg['second_deriv_sg_median'].dropna()),
            'median_of_max_absolute':
                metrics_df_sg['second_deriv_sg_max_abs'].median()
        }
    }

    derivative_spectra_sg = pd.DataFrame({
        'wavelength': wavelengths,
        'median_first_derivative_sg': median_first_deriv_sg,
        'first_deriv_sg_25th': first_deriv_sg_25th,
        'first_deriv_sg_75th': first_deriv_sg_75th,
        'median_second_derivative_sg': median_second_deriv_sg,
        'second_deriv_sg_25th': second_deriv_sg_25th,
        'second_deriv_sg_75th': second_deriv_sg_75th
    })
    spectra_filename = os.path.join(
        output_dir_deriv, 'derivative_sg_median_spectra.csv'
    )
    derivative_spectra_sg.to_csv(
        spectra_filename, index=False, float_format='%.6g'
    )
    results['files']['median_spectra_csv'] = os.path.basename(spectra_filename)

    results['plotting_data'] = {
        'wavelengths': wavelengths,
        'median_first_deriv_sg': median_first_deriv_sg,
        'first_deriv_sg_25th': first_deriv_sg_25th,
        'first_deriv_sg_75th': first_deriv_sg_75th,
        'median_second_deriv_sg': median_second_deriv_sg,
        'second_deriv_sg_25th': second_deriv_sg_25th,
        'second_deriv_sg_75th': second_deriv_sg_75th
    }
    logging.info("--- Derivative Analysis (Savitzky-Golay) Finished ---")
    return results

#############################################################################
# --- STEP 6: GENERATE COMPOSITE FIGURE ---
#############################################################################
def generate_composite_figure(core_plotting_data, normality_plotting_data,
                              baseline_plotting_data, deriv_plotting_data,
                              output_dir_base):
    """Generates a single 3x3 composite figure summarizing quality checks."""
    logging.info("--- 6. Generating Composite Summary Figure ---")
    set_plot_styles()
    output_dir_plots = os.path.join(output_dir_base, "Summary_Plots")
    os.makedirs(output_dir_plots, exist_ok=True)

    try:
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))

        panel_titles = [
            'A) Outlier Samples', 'B) Spectral Standard Deviation',
            'C) Correlation Heatmap', 'D) Shapiro P-value Distribution',
            'E) Example Band KDEs', 'F) Distribution of Baseline Slopes',
            'G) Boxplot of Baseline Slopes',
            'H) Median 1st Derivative (Sav-Gol)',
            'I) Median 2nd Derivative (Sav-Gol)'
        ]

        # Row 1: Core Quality
        ax = axes[0, 0]
        outlier_summary = core_plotting_data.get('outlier_summary', {})
        methods_run = core_plotting_data.get('methods_run_outliers', [])
        outlier_plot_data = {
            'IQR': outlier_summary.get('iqr', {}).get('count', 0),
            'Mod Z': outlier_summary.get('mod_zscore', {}).get('count', 0),
            'LOF': outlier_summary.get('lof', {}).get('count', 0)
        }
        plot_counts = {
            k: v for k, v in outlier_plot_data.items()
            if isinstance(v, int) and k in [
                ('IQR' if 'IQR' in methods_run else None),
                ('Mod Z' if 'Modified Z-Score' in methods_run else None),
                ('LOF' if 'LOF' in methods_run else None)
            ] if v is not None
        }

        if plot_counts:
            colors = plt.get_cmap(PLOT_CMAP)(
                np.linspace(0.3, 0.7, len(plot_counts))
            )
            bars = ax.bar(plot_counts.keys(), plot_counts.values(),
                          color=colors, alpha=0.8)
            ax.bar_label(bars, fmt='%d', fontsize=PLOT_LEGEND_FONTSIZE)
            ax.set_ylabel('Number of Samples Flagged', fontsize=PLOT_AXIS_FONTSIZE)
        else:
            ax.text(0.5, 0.5, 'No outlier counts to plot\n(Check logs)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=PLOT_AXIS_FONTSIZE)
        ax.set_title(panel_titles[0], weight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=0, labelsize=PLOT_AXIS_FONTSIZE-1)

        ax = axes[0, 1]
        std_devs = core_plotting_data.get('std_devs')
        if std_devs is not None:
            wavelengths_core = core_plotting_data.get('wavelengths')
            ax.plot(wavelengths_core, std_devs.values,
                    color=plt.get_cmap(PLOT_CMAP)(0.6))
            ax.set_xlabel('Wavelength / Index', fontsize=PLOT_AXIS_FONTSIZE)
            ax.set_ylabel('Standard Deviation', fontsize=PLOT_AXIS_FONTSIZE)
            ax.grid(axis='both', linestyle='--', alpha=0.6)
        else:
            ax.text(0.5, 0.5, 'STD data unavailable', ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_AXIS_FONTSIZE)
        ax.set_title(panel_titles[1], weight='bold')

        ax = axes[0, 2]
        corr_matrix = core_plotting_data.get('corr_matrix_sampled')
        title_suffix = ''
        if corr_matrix is not None:
            tick_labels = core_plotting_data.get('corr_tick_labels')
            tick_locs = core_plotting_data.get('corr_tick_locs')
            sns.heatmap(corr_matrix, cmap=PLOT_CMAP, vmin=0, vmax=1, ax=ax,
                        cbar=True, cbar_kws={'shrink': .6})
            title_suffix = ' (~100 Bands)' if len(tick_locs) < corr_matrix.shape[0] else ''
            tick_step = max(1, len(tick_locs) // 6)
            ax.set_xticks(tick_locs[::tick_step])
            ax.set_xticklabels(
                [str(lbl).split('_')[-1] for lbl in tick_labels[::tick_step]],
                rotation=90, fontsize=PLOT_AXIS_FONTSIZE - 2
            )
            ax.set_yticks(tick_locs[::tick_step])
            ax.set_yticklabels(
                [str(lbl).split('_')[-1] for lbl in tick_labels[::tick_step]],
                rotation=0, fontsize=PLOT_AXIS_FONTSIZE - 2
            )
        else:
            ax.text(0.5, 0.5, 'Corr matrix unavailable', ha='center',
                    va='center', transform=ax.transAxes,
                    fontsize=PLOT_AXIS_FONTSIZE)
        ax.set_title(panel_titles[2] + title_suffix, weight='bold')

        # Row 2: Normality & Baseline
        ax = axes[1, 0]
        p_values = normality_plotting_data.get('p_values')
        if p_values:
            ax.hist(p_values, bins=50, edgecolor='black',
                    color=plt.get_cmap(PLOT_CMAP)(0.6), alpha=0.8)
            ax.axvline(x=ALPHA, color='r', linestyle='--', label=f'α = {ALPHA}')
            ax.set_xlabel('P-value', fontsize=PLOT_AXIS_FONTSIZE)
            ax.set_ylabel('Frequency', fontsize=PLOT_AXIS_FONTSIZE)
            ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        else:
            ax.text(0.5, 0.5, 'No p-values', ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_AXIS_FONTSIZE)
        ax.set_title(panel_titles[3], weight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)

        ax = axes[1, 1]
        data_samples = normality_plotting_data.get('spectral_data_samples')
        if data_samples is not None:
            sample_names = normality_plotting_data.get('sample_wavelength_names')
            results_df_norm = normality_plotting_data.get('results_df')
            colors = plt.get_cmap(PLOT_CMAP)(
                np.linspace(0.2, 0.8, len(sample_names))
            )
            for i, wave in enumerate(sample_names):
                p_val_info = results_df_norm[
                    results_df_norm['wavelength_column'] == wave
                ]['p_value'].iloc[0]
                label = (f"{str(wave).split('_')[-1]} (p={p_val_info:.1e})"
                         if not pd.isna(p_val_info)
                         else f"{str(wave).split('_')[-1]} (N/A)")
                sns.kdeplot(data=data_samples[wave].dropna(), ax=ax, label=label,
                            color=colors[i], fill=True, alpha=0.4,
                            warn_singular=False)
            ax.set_xlabel('Reflectance / Value', fontsize=PLOT_AXIS_FONTSIZE)
            ax.set_ylabel('Density', fontsize=PLOT_AXIS_FONTSIZE)
            ax.legend(title='Band (Shapiro p)', fontsize=PLOT_LEGEND_FONTSIZE)
        else:
            ax.text(0.5, 0.5, 'No sample data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_AXIS_FONTSIZE)
        ax.set_title(panel_titles[4], weight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)

        ax = axes[1, 2]
        valid_slopes_base = baseline_plotting_data.get('valid_slopes')
        if valid_slopes_base is not None and not valid_slopes_base.empty:
            sns.histplot(valid_slopes_base, bins=30, kde=True, ax=ax,
                         color=plt.get_cmap(PLOT_CMAP)(0.7), alpha=0.8)
            med_slope_plot = baseline_plotting_data.get('summary_metrics', {})\
                .get('trend_analysis', {}).get('median_theil_sen_slope', np.nan)
            ax.axvline(med_slope_plot, color='r', linestyle='--',
                        label=f'Median = {med_slope_plot:.2e}')
            ax.set_xlabel('Theil-Sen Slope per Sample', fontsize=PLOT_AXIS_FONTSIZE)
            ax.set_ylabel('Count', fontsize=PLOT_AXIS_FONTSIZE)
            ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        else:
            ax.text(0.5, 0.5, 'No slope data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_AXIS_FONTSIZE)
        ax.set_title(panel_titles[5], weight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)

        # Row 3: Baseline & Derivatives
        ax = axes[2, 0]
        if valid_slopes_base is not None and not valid_slopes_base.empty:
            sns.boxplot(y=valid_slopes_base, ax=ax,
                        color=plt.get_cmap(PLOT_CMAP)(0.5), width=0.4,
                        showfliers=True,
                        flierprops={"marker": ".", "markersize": 3,
                                    "markerfacecolor": "gray", "alpha": 0.5})
            ax.set_ylabel('Slope per Sample', fontsize=PLOT_AXIS_FONTSIZE)
            ax.set_xlabel('', fontsize=PLOT_AXIS_FONTSIZE)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
            ax.text(0.5, 0.5, 'No slope data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_AXIS_FONTSIZE)
        ax.set_title(panel_titles[6], weight='bold')

        ax = axes[2, 1]
        median_deriv1 = deriv_plotting_data.get('median_first_deriv_sg')
        if median_deriv1 is not None:
            wavelengths_deriv = deriv_plotting_data.get('wavelengths')
            d1_25 = deriv_plotting_data.get('first_deriv_sg_25th')
            d1_75 = deriv_plotting_data.get('first_deriv_sg_75th')
            ax.plot(wavelengths_deriv, median_deriv1,
                    color=plt.get_cmap(PLOT_CMAP)(0.7), label='Median')
            ax.fill_between(wavelengths_deriv, d1_25, d1_75,
                            color=plt.get_cmap(PLOT_CMAP)(0.7), alpha=0.3,
                            label='IQR')
            ax.set_xlabel('Wavelength / Index', fontsize=PLOT_AXIS_FONTSIZE)
            ax.set_ylabel('1st Derivative', fontsize=PLOT_AXIS_FONTSIZE)
            ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        else:
            ax.text(0.5, 0.5, 'No SG deriv data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_AXIS_FONTSIZE)
        ax.set_title(panel_titles[7], weight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)

        ax = axes[2, 2]
        median_deriv2 = deriv_plotting_data.get('median_second_deriv_sg')
        if median_deriv2 is not None:
            wavelengths_deriv = deriv_plotting_data.get('wavelengths')
            d2_25 = deriv_plotting_data.get('second_deriv_sg_25th')
            d2_75 = deriv_plotting_data.get('second_deriv_sg_75th')
            ax.plot(wavelengths_deriv, median_deriv2,
                    color=plt.get_cmap(PLOT_CMAP)(0.5), label='Median')
            ax.fill_between(wavelengths_deriv, d2_25, d2_75,
                            color=plt.get_cmap(PLOT_CMAP)(0.5), alpha=0.3,
                            label='IQR')
            ax.set_xlabel('Wavelength / Index', fontsize=PLOT_AXIS_FONTSIZE)
            ax.set_ylabel('2nd Derivative', fontsize=PLOT_AXIS_FONTSIZE)
            ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        else:
            ax.text(0.5, 0.5, 'No SG deriv data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=PLOT_AXIS_FONTSIZE)
        ax.set_title(panel_titles[8], weight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)

        fig.suptitle(
            'Comprehensive Data Quality Assessment Summary',
            fontsize=PLOT_TITLE_FONTSIZE + 4, y=1.0, weight='bold'
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        composite_plot_filename = os.path.join(
            output_dir_plots, 'composite_quality_summary.png'
        )
        plt.savefig(composite_plot_filename, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Composite summary plot saved to {composite_plot_filename}")
        return os.path.basename(composite_plot_filename)

    except Exception as e:
        logging.error(f"Failed to create composite summary plot: {e}", exc_info=True)
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

#############################################################################
# --- MAIN PIPELINE FUNCTION ---
#############################################################################

def run_data_quality_pipeline(input_file, spec_start_col, output_dir):
    """Executes the complete pre-analysis data quality pipeline."""
    logger = logging.getLogger()
    logger.info("=== Starting Data Quality Pipeline ===")
    logger.info(f"Input File: {input_file}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(
        f"Config Flags: LOF={RUN_LOF_OUTLIERS}, Normality={RUN_NORMALITY_TESTS}, "
        f"Baseline={RUN_BASELINE_ASSESSMENT}, Derivative={RUN_DERIVATIVE_ANALYSIS}"
    )

    pipeline_results = {
        "pipeline_start_time": datetime.now().isoformat(),
        "input_file": input_file,
        "output_directory": output_dir,
        "steps_run": [],
        "data_preparation": {},
        "core_quality": {},
        "normality": {},
        "baseline": {},
        "derivatives_sg": {},
        "summary_plots": {}
    }
    composite_plot_data = {}

    try:
        metadata, spectral_data, wavelengths, prep_info = \
            load_and_prepare_data(input_file, spec_start_col)
        pipeline_results["data_preparation"] = prep_info
        pipeline_results["steps_run"].append("1_Load_Prepare")
        composite_plot_data['spectral_data'] = spectral_data

        core_results = perform_core_quality_assessment(
            spectral_data, wavelengths, output_dir
        )
        pipeline_results["core_quality"] = core_results
        pipeline_results["steps_run"].append("2_Core_Quality")
        composite_plot_data['core'] = core_results.get('plotting_data', {})

        normality_results = perform_normality_assessment(spectral_data, output_dir)
        pipeline_results["normality"] = normality_results
        pipeline_results["steps_run"].append("3_Normality")
        composite_plot_data['normality'] = normality_results.get('plotting_data', {})

        baseline_results = perform_robust_baseline_assessment(
            spectral_data, output_dir
        )
        pipeline_results["baseline"] = baseline_results
        pipeline_results["steps_run"].append("4_Baseline")
        composite_plot_data['baseline'] = baseline_results.get('plotting_data', {})

        derivative_results = calculate_savitzky_golay_derivatives(
            spectral_data, wavelengths, output_dir
        )
        pipeline_results["derivatives_sg"] = derivative_results
        pipeline_results["steps_run"].append("5_Derivatives_SG")
        composite_plot_data['derivatives_sg'] = derivative_results.get(
            'plotting_data', {}
        )

        composite_filename = generate_composite_figure(
            core_plotting_data=composite_plot_data.get('core', {}),
            normality_plotting_data=composite_plot_data.get('normality', {}),
            baseline_plotting_data=composite_plot_data.get('baseline', {}),
            deriv_plotting_data=composite_plot_data.get('derivatives_sg', {}),
            output_dir_base=output_dir
        )
        if composite_filename:
            pipeline_results["summary_plots"]["composite_figure"] = composite_filename
        pipeline_results["steps_run"].append("6_Composite_Plot")

        logger.info("=== Data Quality Pipeline Finished Successfully ===")

    except Exception as e:
        logger.exception("FATAL ERROR during pipeline execution.")
        pipeline_results["pipeline_error"] = str(e)

    finally:
        pipeline_results["pipeline_end_time"] = datetime.now().isoformat()
        report_filename = os.path.join(
            output_dir, "pipeline_summary_report.json"
        )
        save_report_to_json(pipeline_results, report_filename)
        logger.info(f"Pipeline summary report saved to {report_filename}")

        print("\n" + "="*60)
        print("Pipeline Execution Summary:")
        print(
            f"  Status: {'Completed' if 'pipeline_error' not in pipeline_results else 'Failed'}"
        )
        if 'pipeline_error' in pipeline_results:
            print(f"  Error: {pipeline_results['pipeline_error']}")
        print(f"  Steps Run: {pipeline_results['steps_run']}")
        core_summary = pipeline_results.get('core_quality', {})
        outlier_count = core_summary.get('outliers', {})\
            .get('combined_unique_outliers', {}).get('count', 'N/A')
        median_std = core_summary.get('basic_metrics', {})\
            .get('median_std_dev', 'N/A')
        print("\n  Core Quality:")
        print(f"    - Potential Outliers Found (Combined): {outlier_count}")
        print(
            f"    - Median Spectral Std Dev: {median_std:.4g}"
            if isinstance(median_std, (int, float))
            else f"Median Spectral Std Dev: {median_std}"
        )
        norm_summary = pipeline_results.get('normality', {}).get('summary', {})
        perc_normal = norm_summary.get('percent_normal_of_tested', 'N/A')
        print("\n  Normality (Shapiro-Wilk):")
        print(
            f"    - % Normal Bands (of Tested): {perc_normal:.1f}%"
            if isinstance(perc_normal, (int, float))
            else f"% Normal Bands (of Tested): {perc_normal}"
        )
        base_summary = pipeline_results.get('baseline', {}).get('summary', {})
        base_needed = base_summary.get('assessment', {})\
            .get('correction_potentially_needed', 'N/A')
        print("\n  Baseline Assessment:")
        print(f"    - Baseline Correction Potentially Needed: {base_needed}")
        if pipeline_results.get("summary_plots", {})\
                .get("composite_figure"):
            print(
                "\n  Composite Summary Plot Generated: "
                f"{pipeline_results['summary_plots']['composite_figure']}"
            )
        print("="*60)


######################################################
############ MAIN EXECUTION BLOCK
######################################################

if __name__ == "__main__":
    logger = setup_logging(OUTPUT_DIR)
    print("\nStarting hyperspectral pre-analysis pipeline...")
    print(f"Output will be saved to: {OUTPUT_DIR}\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_data_quality_pipeline(
        input_file=INPUT_FILE_HYPER,
        spec_start_col=SPECTRAL_DATA_START_COL,
        output_dir=OUTPUT_DIR
    )
    print("\n==============================================")
    print("Pipeline execution finished.")
    print(f"Check the output directory for results:\n{OUTPUT_DIR}")
    print("Check the main log file and JSON report for details.")
    print("==============================================")