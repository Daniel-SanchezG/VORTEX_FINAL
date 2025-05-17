#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Control Test Analysis Module for VORTEX
---------------------------------------
Module to test the VORTEX system with new control datasets using only the full_model.
This script simplifies testing by using only the full_model on new datasets.

Example of use:
    python control_test.py --data "/path/to/CAA_control_test.xlsx" --models "models" --output "control_test_results"
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import datetime
import logging
from pycaret.classification import load_model, predict_model
from scipy.stats import entropy

# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import original modules
try:
    from src.real_world.uncertainty_analysis import process_predictions_with_uncertainty
    from src.real_world.provenance_determination import process_provenance_determination
    from src.real_world.visualization import generate_visualization
    ORIGINAL_MODULES_AVAILABLE = True
except ImportError:
    ORIGINAL_MODULES_AVAILABLE = False

def setup_logging(output_dir):
    """Set up logging for the control test."""
    log_file = os.path.join(output_dir, f"control_test_{datetime.datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ControlTest")

def load_data_from_excel(data_path, logger):
    """
    Load test data from an Excel file.
    
    Args:
        data_path (str): Path to the Excel file.
        logger (logging.Logger): Logger object.
        
    Returns:
        tuple: (DataFrame with data, metadata dictionary, sheet name)
    """
    logger.info(f"Loading data from: {data_path}")
    
    try:
        # Get the first sheet of the Excel file
        xl = pd.ExcelFile(data_path)
        sheet_name = xl.sheet_names[0]
        logger.info(f"Using sheet: {sheet_name}")
        
        # Load the data as strings to avoid categorical issues
        df = pd.read_excel(data_path, sheet_name=sheet_name, dtype=str)
        logger.info(f"Loaded {len(df)} rows of data")
        
        # Extract ID and site information
        metadata = {}
        
        # Handle ID column
        id_col = None
        for col in ['id', 'id_inv', 'ID', 'Id']:
            if col in df.columns:
                id_col = col
                break
        
        if id_col:
            metadata['id'] = df[id_col].tolist()
            df = df.drop(columns=[id_col])
        else:
            # Create sequential IDs
            metadata['id'] = list(range(1, len(df) + 1))
        
        # Handle site column
        site_col = None
        for col in ['site', 'Site', 'Yac', 'yac']:
            if col in df.columns:
                site_col = col
                break
        
        if site_col:
            metadata['Yac'] = df[site_col].tolist()
            df = df.drop(columns=[site_col])
            site_name = metadata['Yac'][0]  # Use the first site name as a reference
        else:
            # Use sheet name as site
            site_name = sheet_name
            metadata['Yac'] = [site_name] * len(df)
        
        # Convert numeric columns back to float
        numeric_cols = []
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
                numeric_cols.append(col)
            except:
                pass
        
        logger.info(f"Converted {len(numeric_cols)} columns to numeric type")
        
        return df, metadata, site_name
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None

def make_predictions(model, data_df, metadata, site_name, logger):
    """
    Make predictions using the full_model.
    
    Args:
        model: Loaded model object.
        data_df (pd.DataFrame): DataFrame with data.
        metadata (dict): Dictionary with metadata.
        site_name (str): Name of the site.
        logger (logging.Logger): Logger object.
        
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    logger.info(f"Making predictions for site: {site_name}")
    
    try:
        # Make prediction with scores
        predictions = predict_model(model, data=data_df, raw_score=True)
        
        # Create a new DataFrame for results
        result_df = pd.DataFrame()
        
        # Add metadata
        result_df['id'] = metadata['id']
        result_df['Site'] = site_name
        result_df['Yac'] = metadata['Yac']
        
        # Handle different conventions of prediction column names
        prediction_col = None
        if 'Label' in predictions.columns:
            prediction_col = 'Label'
        elif 'prediction_label' in predictions.columns:
            prediction_col = 'prediction_label'
        
        if prediction_col:
            result_df['Prediction'] = predictions[prediction_col].astype(str)
        else:
            result_df['Prediction'] = "Unknown"
        
        # Extract and rename score columns if they exist
        score_patterns = {
            'CT': ['Score_CT', 'prediction_score_CT', 'score_CT'],
            'PCM': ['Score_PCM', 'prediction_score_PCM', 'score_PCM'],
            'PDLC': ['Score_PDLC', 'prediction_score_PDLC', 'score_PDLC']
        }
        
        for class_name, possible_cols in score_patterns.items():
            target_col = f'prediction_score_{class_name}'
            found = False
            
            for col in possible_cols:
                if col in predictions.columns:
                    result_df[target_col] = predictions[col]
                    found = True
                    break
            
            if not found:
                result_df[target_col] = np.nan
        
        logger.info(f"Generated predictions for {len(result_df)} samples")
        return result_df
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return None

def analyze_uncertainty(prediction_df, confidence_threshold=0.7, logger=None):
    """
    Performs uncertainty analysis on predictions.
    """
    if logger:
        logger.info("Starting uncertainty analysis...")
    
    try:
        # Create a fresh DataFrame to avoid categorical issues
        results_df = pd.DataFrame()
        
        # Copy basic columns
        for col in ['id', 'Site', 'Yac', 'Prediction']:
            if col in prediction_df.columns:
                results_df[col] = prediction_df[col].astype(str)
        
        # Copy score columns
        score_cols = [col for col in prediction_df.columns if col.startswith('prediction_score_')]
        
        if len(score_cols) == 0:
            if logger:
                logger.error("No score columns found in the input data")
            return None
        
        # Copy score values
        for col in score_cols:
            results_df[col] = pd.to_numeric(prediction_df[col], errors='coerce')
        
        # Calculate confidence and entropy
        probas = results_df[score_cols].values
        confidences = np.max(probas, axis=1)
        
        # Store original predictions
        results_df['Original_predictions'] = results_df['Prediction'].values
        results_df['Confidence'] = confidences
        
        # Create uncertainty threshold predictions as list
        predictions = results_df['Prediction'].values
        uncertain_preds = [p if c >= confidence_threshold else 'uncertain' 
                          for p, c in zip(predictions, confidences)]
        
        # Add back to DataFrame
        results_df['Uncertainty_threshold_predictions'] = uncertain_preds
        
        # Calculate entropy
        entropies = []
        for probs in probas:
            try:
                ent = entropy(probs, base=2)
            except:
                ent = np.nan
            entropies.append(ent)
        
        results_df['Entropy'] = entropies
        
        # Log statistics
        if logger:
            n_uncertain = sum(c < confidence_threshold for c in confidences)
            uncertain_percent = (n_uncertain / len(results_df) * 100)
            logger.info(f"Uncertain predictions: {n_uncertain}/{len(results_df)} ({uncertain_percent:.1f}%)")
            logger.info(f"Mean entropy: {np.nanmean(entropies):.3f}")
        
        return results_df
        
    except Exception as e:
        if logger:
            logger.error(f"Error in uncertainty analysis: {str(e)}")
        return None

def determine_provenance(uncertainty_df, confidence_threshold=0.7, logger=None):
    """
    Performs provenance determination based on uncertainty analysis.
    """
    if logger:
        logger.info("Starting provenance determination analysis...")
    
    try:
        # Copy DataFrame to avoid modifying the original
        df = uncertainty_df.copy()
        
        # Verify that the DataFrame has the necessary columns
        required_cols = ['Site', 'Original_predictions', 'Entropy']
        score_cols = [col for col in df.columns if col.startswith('prediction_score_')]
        
        for col in required_cols:
            if col not in df.columns:
                if logger:
                    logger.error(f"Required column {col} not found in the DataFrame")
                return None
        
        # Calculate maximum probability for each sample
        df['max_prob'] = df[score_cols].max(axis=1)
        
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
        if logger:
            logger.info(f"Provenance determination analysis completed for {len(result_df)} sites")
        
        return result_df
    
    except Exception as e:
        if logger:
            logger.error(f"Error in provenance determination analysis: {str(e)}")
        return None

def plot_site_entropy_distribution(uncertainty_df, output_file, logger=None):
    """
    Create entropy distribution visualization.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if logger:
            logger.info("Generating entropy distribution visualization...")
        
        # Copy DataFrame to avoid modifying the original
        df = uncertainty_df.copy()
        
        # Identify score columns
        score_cols = [col for col in df.columns if col.startswith('prediction_score_')]
        
        # Calculate medians by site
        site_medians = df.groupby('Site')[score_cols + ['Entropy']].median()
        
        # Configure plot style
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create stacked bar plot
        bottom = np.zeros(len(site_medians))
        colors = ['#4B0082', '#228B22', '#B8860B']  # Colors for CT, PCM, PDLC
        
        # Use site names as x positions
        x = np.arange(len(site_medians.index))

        class_names = {
            'CT': 'Gavà',
            'PCM': 'Terena',
            'PDLC': 'Aliste'
        }
        
        for i, col in enumerate(score_cols):
            class_code = col.replace('prediction_score_', '')
            class_label = class_names.get(class_code, class_code)
            
            ax.bar(x, site_medians[col], bottom=bottom, 
                  label=class_label,
                  color=colors[i], alpha=0.7)
            bottom += site_medians[col]
        
        # Add entropy line
        ax2 = ax.twinx()
        ax2.plot(x, site_medians['Entropy'], 
                color='red', linewidth=2, label='Entropy', 
                marker='o')
        
        # Configure axes and labels
        ax.set_xlabel('Sites', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Distribution (Median)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold', color='red')
        
        # Rotate x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(site_medians.index, rotation=45, ha='right')
        
        # Adjust legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, 
                 loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Adjust layout to avoid cutting labels
        plt.subplots_adjust(bottom=0.2)
        
        # Save plot
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.info(f"Visualization saved in: {output_file}")
        
        # Create site statistics
        stats = pd.DataFrame({
            'median_CT': site_medians['prediction_score_CT'] if 'prediction_score_CT' in site_medians else np.nan,
            'median_PCM': site_medians['prediction_score_PCM'] if 'prediction_score_PCM' in site_medians else np.nan,
            'median_PDLC': site_medians['prediction_score_PDLC'] if 'prediction_score_PDLC' in site_medians else np.nan,
            'median_entropy': site_medians['Entropy'],
            'n_samples': df.groupby('Site').size(), 
            'mean_entropy': df.groupby('Site')['Entropy'].mean(),
            'std_entropy': df.groupby('Site')['Entropy'].std()
        }).round(3)
        
        if logger:
            logger.info("Site statistics generated")
        
        return stats
        
    except Exception as e:
        if logger:
            logger.error(f"Error generating visualization: {str(e)}")
        return None

def main():
    """Main function to execute the control test."""
    parser = argparse.ArgumentParser(description='VORTEX Control Test Analysis')
    parser.add_argument('--data', required=True, help='Path to the Excel file with control/test data')
    parser.add_argument('--models', required=True, help='Directory containing trained models')
    parser.add_argument('--output', required=True, help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold (default: 0.7)')
    parser.add_argument('--no-uncertainty', action='store_true', help='Skip uncertainty analysis')
    parser.add_argument('--no-provenance', action='store_true', help='Skip provenance determination')
    parser.add_argument('--no-visualization', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.output)
    logger.info("Starting control test analysis")
    
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    prediction_file = os.path.join(args.output, f"control_predictions_{current_date}.xlsx")
    uncertainty_file = os.path.join(args.output, f"uncertainty_analysis_{current_date}.xlsx")
    provenance_file = os.path.join(args.output, f"provenance_analysis_{current_date}.xlsx")
    vis_dir = os.path.join(args.output, 'visualizations')
    
    try:
        # Step 1: Load the model
        logger.info(f"Loading full_model from {args.models}")
        model_path = os.path.join(args.models, 'final_model')
        
        try:
            model = load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            print(f"❌ Error loading model: {str(e)}")
            return False
        
        # Step 2: Load the data
        data_df, metadata, site_name = load_data_from_excel(args.data, logger)
        
        if data_df is None:
            print(f"❌ Error loading data")
            return False
        
        # Step 3: Make predictions
        logger.info("Making predictions")
        results = make_predictions(model, data_df, metadata, site_name, logger)
        
        if results is None:
            print(f"❌ Error making predictions")
            return False
        
        # Save predictions
        results.to_excel(prediction_file, index=False)
        logger.info(f"Predictions saved to {prediction_file}")
        print(f"✅ Predictions generated and saved to {prediction_file}")
        
        # Step 4: Uncertainty analysis (optional)
        if args.no_uncertainty:
            logger.info("Uncertainty analysis skipped per configuration")
            print("\n🔍 Uncertainty analysis skipped per configuration")
            
            if not args.no_provenance or not args.no_visualization:
                logger.error("Cannot perform subsequent analyses without uncertainty analysis")
                print("\n❌ Cannot perform subsequent analyses without uncertainty analysis")
                return True
            
            return True
        
        # Always use our own implementation
        logger.info("Using internal uncertainty analysis implementation")
        uncertainty_df = analyze_uncertainty(results, args.threshold, logger)
        
        if uncertainty_df is None:
            logger.error("Uncertainty analysis failed")
            print("\n⚠️ Uncertainty analysis failed - check logs for details")
            return True
        
        # Save uncertainty analysis
        uncertainty_df.to_excel(uncertainty_file, index=False)
        logger.info(f"Uncertainty analysis saved to {uncertainty_file}")
        print(f"✅ Uncertainty analysis completed and saved to {uncertainty_file}")
        
        # Step 5: Provenance determination (optional)
        if args.no_provenance:
            logger.info("Provenance determination skipped per configuration")
            print("\n🔎 Provenance determination skipped per configuration")
        else:
            # Always use our own implementation
            logger.info("Using internal provenance determination implementation")
            provenance_df = determine_provenance(uncertainty_df, args.threshold, logger)
            
            if provenance_df is None:
                logger.error("Provenance determination failed")
                print("\n⚠️ Provenance determination failed")
            else:
                provenance_df.to_excel(provenance_file, index=False)
                logger.info(f"Provenance determination saved to {provenance_file}")
                print(f"✅ Provenance determination completed and saved to {provenance_file}")
        
        # Step 6: Visualization (optional)
        if args.no_visualization:
            logger.info("Visualization generation skipped per configuration")
            print("\n📈 Visualization generation skipped per configuration")
        else:
            # Always use our own implementation
            logger.info("Using internal visualization implementation")
            os.makedirs(vis_dir, exist_ok=True)
            
            vis_out_file = os.path.join(vis_dir, f"entropy_distribution_{current_date}.pdf")
            stats = plot_site_entropy_distribution(uncertainty_df, vis_out_file, logger)
            
            if stats is not None:
                stats_out_file = os.path.join(vis_dir, f"site_statistics_{current_date}.xlsx")
                stats.to_excel(stats_out_file)
                
                logger.info("Visualizations generated successfully")
                print(f"✅ Visualizations generated successfully")
                print(f"   Visualization: {vis_out_file}")
                print(f"   Statistics: {stats_out_file}")
            else:
                logger.error("Visualization generation failed")
                print("\n⚠️ Visualization generation failed")
        
        print("\n🏆 CONTROL TEST COMPLETED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)