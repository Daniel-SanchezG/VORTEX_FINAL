#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uncertainty Analysis Module
-----------------------------------
Complement to the archaeological prediction system that adds
uncertainty analysis to the generated predictions.
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
import logging
import os
import datetime

# Logging configuration
logger = logging.getLogger("Uncertainty")

def analyze_uncertainty(prediction_df, confidence_threshold=0.7):
    """
    Performs uncertainty analysis on a DataFrame of predictions.

    Args:
        prediction_df (pd.DataFrame): DataFrame with predictions and scores.
        confidence_threshold (float): Confidence threshold to mark as uncertain.

    Returns:
        pd.DataFrame: DataFrame with analysis results.
    """
    logger.info("Starting uncertainty analysis...")
    
    try:
        # Copy DataFrame to avoid modifying the original
        df = prediction_df.copy()
        
        # Identify score columns
        score_cols = [col for col in df.columns if col.startswith('prediction_score_')]
        
        if not score_cols:
            logger.warning("No score columns found.")
            return None
        
        logger.info(f"Using score columns: {score_cols}")
        
        # Convert score_cols to numbers if they are strings
        for col in score_cols:
            if df[col].dtype == object:  # If it's a string
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Get probabilities
        probas = df[score_cols].values
        
        # Get prediction labels and confidence
        prediction_col = 'Prediction' if 'Prediction' in df.columns else 'predictions'
        predictions = df[prediction_col].values
        confidences = np.max(probas, axis=1)
        
        # Mark predictions below the threshold as uncertain
        uncertain_mask = confidences < confidence_threshold
        predictions_with_uncertainty = predictions.copy()
        predictions_with_uncertainty[uncertain_mask] = 'uncertain'
        
        # Calculate entropy
        entropies = np.array([entropy(probs, base=2) for probs in probas])
        
        # Create DataFrame with results
        results_df = df.copy()
        
        # Add new analysis columns
        results_df['Original_predictions'] = predictions
        results_df['Confidence'] = confidences
        results_df['Uncertainty_threshold_predictions'] = predictions_with_uncertainty
        results_df['Entropy'] = entropies
        
        # Calculate global metrics
        n_uncertain = np.sum(uncertain_mask)
        uncertain_percent = (n_uncertain / len(df) * 100)
        mean_entropy = entropies.mean()
        
        logger.info(f"Uncertain predictions: {n_uncertain}/{len(df)} ({uncertain_percent:.1f}%)")
        logger.info(f"Mean entropy: {mean_entropy:.3f}")
        
        # Calculate median entropy by site if the Site column exists
        if 'Site' in results_df.columns:
            entropy_median_by_site = results_df.groupby('Site')['Entropy'].median()
            for site, median in entropy_median_by_site.items():
                logger.info(f"Median entropy for {site}: {median:.3f}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error in uncertainty analysis: {str(e)}")
        return None

def save_uncertainty_results(results_df, output_path=None):
    """
    Saves the analysis results in Excel and CSV files.

    Args:
        results_df (pd.DataFrame): DataFrame with analysis results.
        output_path (str, optional): Base path for result files.
            If None, it is generated automatically.

    Returns:
        dict: Dictionary with paths to the generated files or None if failed.
    """
    if results_df is None:
        logger.error("No results to save.")
        return None
    
    try:
        # Generate base path if not provided
        if output_path is None:
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_path = f"uncertainty_analysis_{current_date}"
        else:
            output_path = os.path.splitext(output_path)[0]

        # Output paths
        excel_path = f"{output_path}.xlsx"
        csv_path = f"{output_path}.csv"
        
        # Save results
        results_df.to_excel(excel_path, index=False)
        results_df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved in Excel: {excel_path}")
        logger.info(f"Results saved in CSV: {csv_path}")
        
        return {
            'excel': excel_path,
            'csv': csv_path
        }
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return None

def process_predictions_with_uncertainty(prediction_df=None, prediction_path=None, 
                                        output_path=None, confidence_threshold=0.7):
    """
    Processes predictions and performs uncertainty analysis.

    Args:
        prediction_df (pd.DataFrame, optional): DataFrame with predictions.
        prediction_path (str, optional): Path to the file with predictions.
        output_path (str, optional): Path to save results.
        confidence_threshold (float): Confidence threshold.

    Returns:
        pd.DataFrame: DataFrame with uncertainty results.
    """
    # Load predictions if path is provided
    if prediction_df is None and prediction_path:
        try:
            file_ext = os.path.splitext(prediction_path)[1].lower()
            if file_ext == '.csv':
                prediction_df = pd.read_csv(prediction_path)
            elif file_ext in ['.xlsx', '.xls']:
                prediction_df = pd.read_excel(prediction_path)
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return None
                
            logger.info(f"Loaded {len(prediction_df)} predictions from {prediction_path}")
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
            return None
    
    # Verify that we have a DataFrame to analyze
    if prediction_df is None:
        logger.error("No DataFrame or valid path provided")
        return None
    
    # Perform uncertainty analysis
    results_df = analyze_uncertainty(prediction_df, confidence_threshold)
    
    # Save results if path is provided
    if results_df is not None and output_path:
        save_uncertainty_results(results_df, output_path)
    
    return results_df

# For direct module testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import sys
    if len(sys.argv) > 1:
        prediction_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        
        results = process_predictions_with_uncertainty(
            prediction_path=prediction_path,
            output_path=output_path,
            confidence_threshold=confidence_threshold
        )
        
        if results is not None:
            print(f"Analysis completed. {len(results)} rows registered.")
        else:
            print("Error in uncertainty analysis.")
    else:
        print("Usage: python uncertainty_analysis.py <predictions_file> [output_file] [confidence_threshold]")