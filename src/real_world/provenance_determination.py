#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module of Provenance Determination
--------------------------------------
Subsecuent step of the prediction process that analyzes
the uncertainty of the predictions and determines the provenance by consensus.
"""

import pandas as pd
import numpy as np
import logging
import os
import datetime

# Configuración de logging
logger = logging.getLogger("Provenance")

def determine_provenance(uncertainty_df, confidence_threshold=0.7):
    """
    Performs a provenance determination analysis based on consensus.
    
    Args:
        uncertainty_df (pd.DataFrame): DataFrame with uncertainty analysis results.
        confidence_threshold (float): Confidence threshold to consider reliable predictions.
    
    Returns:
        pd.DataFrame: DataFrame with provenance results by site.
    """
    logger.info("Starting provenance determination analysis...")
    
    try:
        # Copy DataFrame to avoid modifying the original
        df = uncertainty_df.copy()
        
        # Verify that the DataFrame has the necessary columns
        required_cols = ['Site', 'Original_predictions', 'Entropy']
        score_cols = [col for col in df.columns if col.startswith('prediction_score_')]
        
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Required column {col} not found in the DataFrame")
                return None
        
        if not score_cols:
            logger.error("No score columns found in the DataFrame")
            return None
        
        # Calculate maximum probability for each sample
        df['max_prob'] = df[score_cols].max(axis=1)
        
        # Verify threshold prediction column
        if 'Uncertainty_threshold_predictions' not in df.columns:
            logger.warning("No threshold prediction column found, it will be generated")
            uncertain_mask = df['max_prob'] < confidence_threshold
            df['Uncertainty_threshold_predictions'] = df['Original_predictions'].copy()
            df.loc[uncertain_mask, 'Uncertainty_threshold_predictions'] = 'uncertain'
        
        # List to store results by site
        results = []
        
        # Analyze each site separately
        for site in df['Site'].unique():
            site_data = df[df['Site'] == site]
            avg_uncertain = round(sum(site_data['max_prob'] < confidence_threshold) / len(site_data) * 100, 2)
            high_conf = site_data[site_data['max_prob'] >= confidence_threshold]
            median_entropy = site_data['Entropy'].median() if 'Entropy' in site_data.columns else np.nan
            
            # Determine consensus based on high confidence samples
            if len(high_conf) > 0:
                # Use the most frequent value as consensus
                consensus_counts = high_conf['Uncertainty_threshold_predictions'].value_counts()
                consensus = consensus_counts.index[0]
                consistency = consensus_counts.iloc[0] / len(high_conf)
                n_consensus_pred = len(high_conf)
            else:
                consensus = 'No consensus'
                consistency = 0
                n_consensus_pred = 0
            
            # Count predictions by class
            ct_count = len(site_data[site_data['Original_predictions'] == 'CT'])
            pcm_count = len(site_data[site_data['Original_predictions'] == 'PCM'])
            pdlc_count = len(site_data[site_data['Original_predictions'] == 'PDLC'])
            
            # Add results for this site
            results.append({
                'Site': site,
                'Samples_analyzed': len(site_data),
                'Gavá': ct_count,
                'Encinasola': pcm_count,
                'Aliste': pdlc_count,
                'Uncertain(%)': round(avg_uncertain,2),
                'Samples_for_provenance': n_consensus_pred,
                'Median_entropy': round(median_entropy, 2) if not np.isnan(median_entropy) else np.nan,
                'Consensus': consensus,
                'Homogeneity': round(consistency, 2)
            })
        
        # Create DataFrame with results
        result_df = pd.DataFrame(results)
        logger.info(f"Provenance determination analysis completed for {len(result_df)} sites")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error in provenance determination analysis: {str(e)}")
        return None

def save_provenance_results(provenance_df, output_path=None):
    """
    Saves the provenance analysis results in Excel and CSV files.
    
    Args:
        provenance_df (pd.DataFrame): DataFrame con resultados de procedencia.
        output_path (str, optional): Base path for the results files.
            If None, it will be generated automatically.
    
    Returns:
        dict: Dictionary with paths to the generated files or None if failed.
    """
    if provenance_df is None:
        logger.error("No provenance results to save")
        return None
    
    try:
        # Generate base path if not provided
        if output_path is None:
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_path = f"provenance_analysis_{current_date}"
        else:
            output_path = os.path.splitext(output_path)[0]
        
        # Output paths
        excel_path = f"{output_path}.xlsx"
        #csv_path = f"{output_path}.csv"
        
        # Save results
        provenance_df.to_excel(excel_path, index=False)
        provenance_df.to_csv(csv_path, index=False)
        
        logger.info(f"Provenance results saved in Excel: {excel_path}")
        logger.info(f"Provenance results saved in CSV: {csv_path}")
        
        return {
            'excel': excel_path,
            'csv': csv_path
        }
    
    except Exception as e:
        logger.error(f"Error saving provenance results: {str(e)}")
        return None

def process_provenance_determination(uncertainty_df=None, uncertainty_path=None, 
                                    output_path=None, confidence_threshold=0.7):
    """
    Processes the uncertainty results and performs provenance analysis.
    
    Args:
        uncertainty_df (pd.DataFrame, optional): DataFrame with uncertainty results.
        uncertainty_path (str, optional): Path to the file with uncertainty results.
        output_path (str, optional): Path to save results.
        confidence_threshold (float): Confidence threshold.
    
    Returns:
        pd.DataFrame: DataFrame with provenance results.
    """
    # Load uncertainty results if path is provided
    if uncertainty_df is None and uncertainty_path:
        try:
            file_ext = os.path.splitext(uncertainty_path)[1].lower()
            if file_ext == '.csv':
                uncertainty_df = pd.read_csv(uncertainty_path)
            elif file_ext in ['.xlsx', '.xls']:
                uncertainty_df = pd.read_excel(uncertainty_path)
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return None
            
            logger.info(f"Loaded uncertainty results from {uncertainty_path}")
        except Exception as e:
            logger.error(f"Error loading uncertainty results: {str(e)}")
            return None
    
    # Verificar que tenemos un DataFrame para analizar
    if uncertainty_df is None:
        logger.error("No provided a DataFrame or a valid path")
        return None
    
    # Perform provenance analysis
    provenance_df = determine_provenance(uncertainty_df, confidence_threshold)
    
    # Save results if path is provided
    if provenance_df is not None and output_path:
        save_provenance_results(provenance_df, output_path)
    
    return provenance_df

# For direct module testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import sys
    if len(sys.argv) > 1:
        uncertainty_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        
        results = process_provenance_determination(
            uncertainty_path=uncertainty_path,
            output_path=output_path,
            confidence_threshold=confidence_threshold
        )
        
        if results is not None:
            print(f"Provenance analysis completed. Analyzed {len(results)} sites.")
        else:
            print("Error in provenance analysis.")
    else:
        print("Usage: python provenance_determination.py <uncertainty_file> [output_path] [confidence_threshold]")