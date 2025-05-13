# ==============================================================================
# Missing Value Imputation for Metabolomics Data
# ==============================================================================
#
# This script implements three different approaches for missing value imputation
# in metabolomics data:
#
# 1. Random Forest (RF) Imputation:
#    - Uses missForest package
#    - Performs well with mixed-type data (numerical and categorical)
#    - Handles non-linear relationships
#    - Computationally intensive but accurate
#
# 2. Predictive Mean Matching (PMM) with Chunking:
#    - Uses MICE package with PMM method
#    - Processes data in smaller chunks to handle large datasets
#    - Preserves data distribution
#    - Memory efficient for high-dimensional data
#
# 3. PMM with Different Chunk Size:
#    - Similar to approach 2 but with larger chunk size
#    - Suitable for datasets with different dimensionality
#
# Required packages: dplyr, missForest, mice, doParallel
# Input: CSV files containing metabolomics data with cluster variables
# Output: Imputed datasets saved as CSV files
# ==============================================================================

# Load required libraries
suppressPackageStartupMessages({
  library(dplyr)
  library(missForest)
  library(mice)
  library(doParallel)
})

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

#' Prepare Data for Imputation
#' @param file_path Path to input CSV file
#' @param cluster_prefix Prefix for cluster columns (e.g., "P_Cluster" or "N_Cluster")
#' @return List containing prepared data and subset for imputation
prepare_data <- function(file_path, cluster_prefix) {
  # Read data
  data <- read.csv(file_path)
  
  # Convert categorical variables to factors
  categorical_vars <- c("Day", "Batch", "Genotype", "Treatment")
  data[categorical_vars] <- lapply(data[categorical_vars], as.factor)
  
  # Select relevant columns
  data_for_imputation <- select(data, 
                               contains(cluster_prefix), 
                               Day, Batch, Genotype, Treatment)
  
  return(list(full_data = data, 
             imputation_data = data_for_imputation))
}

#' Save Imputed Data
#' @param data Data frame to save
#' @param file_path Output file path
#' @param chunk_num Optional chunk number for chunked processing
save_imputed_data <- function(data, file_path, chunk_num = NULL) {
  if (!is.null(chunk_num)) {
    file_path <- gsub("\\.csv$", sprintf("_%d.csv", chunk_num), file_path)
  }
  write.csv(data, file_path, row.names = FALSE)
}

# -----------------------------------------------------------------------------
# Random Forest Imputation
# -----------------------------------------------------------------------------

perform_rf_imputation <- function(input_path, output_path) {
  # Prepare data
  data_list <- prepare_data(input_path, "P_Cluster")
  
  # Setup parallel processing
  registerDoParallel(cores = detectCores())
  
  # Perform RF imputation
  imputed_data_rf <- missForest(data_list$imputation_data, 
                               maxiter = 5, 
                               ntree = 50, 
                               parallelize = 'variables')
  
  # Stop parallel processing
  stopImplicitCluster()
  
  # Update original data with imputed values
  data_list$full_data[, names(data_list$imputation_data)] <- imputed_data_rf$ximp
  
  # Save results
  save_imputed_data(data_list$full_data, output_path)
  print("RF imputation completed and data saved.")
}

# -----------------------------------------------------------------------------
# Chunked PMM Imputation
# -----------------------------------------------------------------------------

perform_chunked_pmm_imputation <- function(input_path, output_path, chunk_size = 100) {
  # Prepare data
  data_list <- prepare_data(input_path, "P_Cluster")
  
  # Calculate number of chunks
  n_chunks <- ceiling(ncol(data_list$imputation_data) / chunk_size)
  
  # Process each chunk
  for (i in 1:n_chunks) {
    # Select columns for current chunk
    cols <- ((i-1) * chunk_size + 1):min(i * chunk_size, ncol(data_list$imputation_data))
    chunk_data <- data_list$imputation_data[, cols, drop = FALSE]
    
    # Perform MICE imputation
    imputed_data_pmm <- mice(chunk_data, 
                            method = 'pmm', 
                            m = 5, 
                            maxit = 5, 
                            seed = 500)
    completed_data_pmm <- complete(imputed_data_pmm, action = 1)
    
    # Update data
    data_list$full_data[, names(chunk_data)] <- completed_data_pmm
    
    # Save chunk results
    save_imputed_data(data_list$full_data[, names(chunk_data)], 
                     output_path, 
                     chunk_num = i)
    
    cat(sprintf("Chunk %d/%d imputation completed and saved.\n", i, n_chunks))
  }
  
  print("All chunks imputation completed and data saved.")
}

# -----------------------------------------------------------------------------
# Large Chunk PMM Imputation
# -----------------------------------------------------------------------------

perform_large_chunk_pmm_imputation <- function(input_path, output_path, chunk_size = 200) {
  # Prepare data
  data_list <- prepare_data(input_path, "N_Cluster")
  
  # Calculate number of chunks
  n_chunks <- ceiling(ncol(data_list$imputation_data) / chunk_size)
  
  # Process each chunk
  for (i in 1:n_chunks) {
    cols <- ((i-1) * chunk_size + 1):min(i * chunk_size, ncol(data_list$imputation_data))
    chunk_data <- data_list$imputation_data[, cols, drop = FALSE]
    
    # Perform MICE imputation with verbose output
    imputed_data_pmm <- mice(chunk_data, 
                            method = 'pmm', 
                            m = 5, 
                            maxit = 5, 
                            seed = 500, 
                            print = TRUE)
    completed_data_pmm <- complete(imputed_data_pmm, action = 1)
    
    # Update data
    data_list$full_data[, names(chunk_data)] <- completed_data_pmm
  }
  
  # Save final results
  save_imputed_data(data_list$full_data, output_path)
  print("Data imputation complete and file saved.")
}

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

# Random Forest Imputation
rf_input_path <- "C:/Users/ms/Desktop/data_chem/data/outlier/p_l_rf_outliers_removed.csv"
rf_output_path <- "C:/Users/ms/Desktop/data_chem/data/outlier/p_l_if.csv"
perform_rf_imputation(rf_input_path, rf_output_path)

# Chunked PMM Imputation
pmm_input_path <- "C:/Users/ms/Desktop/data_chem/data/old_2/p_column_data_r_deducted.csv"
pmm_output_path <- "C:/Users/ms/Desktop/data_chem/imputated/p_column_data_r_deducted_pmm"
perform_chunked_pmm_imputation(pmm_input_path, pmm_output_path)

# Large Chunk PMM Imputation
large_pmm_input_path <- "C:/Users/ms/Desktop/data_chem/imputated/n_column_data_r_deducted_imputed_pmm3b.csv"
large_pmm_output_path <- "C:/Users/ms/Desktop/data_chem/imputated/n_column_data_r_deducted_imputed_pmm3c.csv"
perform_large_chunk_pmm_imputation(large_pmm_input_path, large_pmm_output_path)