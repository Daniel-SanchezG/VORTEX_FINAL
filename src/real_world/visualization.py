#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization for Archaeological Analysis
-------------------------------------------------
Complement to create visualizations of entropy and probabilities
based on the results of uncertainty analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# Logging configuration
logger = logging.getLogger("Visualization")

def plot_site_entropy_distribution(uncertainty_df, output_file=None, 
                                  entropy_col='Entropy', score_cols=None):
    """
    Create a plot that shows the probability distribution and entropy by site.
    
    Args:
        uncertainty_df (pd.DataFrame): DataFrame with uncertainty analysis results.
        output_file (str, optional): Path to save the plot.
        entropy_col (str): Name of the entropy column.
        score_cols (list): List of columns with probability scores.
    
    Returns:
        pd.DataFrame: DataFrame with site statistics.
    """
    logger.info("Generating visualization of entropy distribution by site...")
    
    try:
        # Copiar DataFrame para no modificar el original
        df = uncertainty_df.copy()
        
        # Identify score columns if not specified
        if score_cols is None:
            score_cols = [col for col in df.columns if col.startswith('prediction_score_')]
            
            if not score_cols:
                logger.error("No score columns found")
                return None
        
        logger.info(f"Using score columns: {score_cols}")
        
        # Ensure columns are numeric
        for col in score_cols:
            if df[col].dtype == object:  # If it's a string
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                
        # Ensure the entropy column exists
        if entropy_col not in df.columns:
            logger.error(f"Entropy column '{entropy_col}' not found")
            return None
        
        # Calculate medians by site
        site_medians = df.groupby('Site')[score_cols + [entropy_col]].median()
        
        # Configure plot style
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create stacked bar plot
        bottom = np.zeros(len(site_medians))
        colors = ['#4B0082', '#228B22', '#B8860B']  # Colors for CT, PCM, PDLC
        
        # Use site names as x positions
        x = np.arange(len(site_medians.index))

        class_names = {
        'CT': 'Can Tintorer',
        'PCM': 'Terena',
        'PDLC': 'Aliste'
        }
        
        for i, col in enumerate(score_cols):
            class_code = col.replace('prediction_score_', '')
            class_label = class_names.get(class_code, class_code)  # Use the full name or the code if not in the mapping
            
            ax.bar(x, site_medians[col], bottom=bottom, 
                label=class_label,
                color=colors[i], alpha=0.7)
            bottom += site_medians[col]
        
        # Add entropy line
        ax2 = ax.twinx()
        ax2.plot(x, site_medians[entropy_col], 
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
        
        # Generate path to save if not provided
        if output_file is None:
            import datetime
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_file = f"site_entropy_distribution_{current_date}.png"
        
        # Save plot
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved in: {output_file}")
        
        # Create and show complete site statistics
        stats = pd.DataFrame({
            'median_CT': site_medians['prediction_score_CT'] if 'prediction_score_CT' in site_medians else np.nan,
            'median_PCM': site_medians['prediction_score_PCM'] if 'prediction_score_PCM' in site_medians else np.nan,
            'median_PDLC': site_medians['prediction_score_PDLC'] if 'prediction_score_PDLC' in site_medians else np.nan,
            'median_entropy': site_medians[entropy_col],
            'n_samples': df.groupby('Site').size(), 
            'mean_entropy': df.groupby('Site')[entropy_col].mean(),
            'std_entropy': df.groupby('Site')[entropy_col].std()
        }).round(3)
        
        logger.info("Site statistics generated")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return None

def save_statistics(stats_df, output_path=None):
    """
    Save statistics in Excel and CSV files.
    
    Args:
        stats_df (pd.DataFrame): DataFrame with site statistics.
        output_path (str, optional): Base path for result files.
    
    Returns:
        dict: Dictionary with paths to the generated files or None if failed.
    """
    if stats_df is None:
        logger.error("No statistics to save")
        return None
    
    try:
        # Generate base path if not provided
        if output_path is None:
            import datetime
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_path = f"site_statistics_{current_date}"
        else:
            output_path = os.path.splitext(output_path)[0]
        
        # Output paths
        excel_path = f"{output_path}.xlsx"
        csv_path = f"{output_path}.csv"
        
        # Save results
        stats_df.to_excel(excel_path)
        stats_df.to_csv(csv_path)
        
        logger.info(f"Statistics saved in Excel: {excel_path}")
        logger.info(f"Statistics saved in CSV: {csv_path}")
        
        return {
            'excel': excel_path,
            'csv': csv_path
        }
        
    except Exception as e:
        logger.error(f"Error saving statistics: {str(e)}")
        return None

def generate_visualization(uncertainty_df=None, uncertainty_path=None, 
                          output_dir=None, entropy_col='Entropy'):
    """
    Generate visualization and statistics based on uncertainty results.
    
    Args:
        uncertainty_df (pd.DataFrame, optional): DataFrame with uncertainty results.
        uncertainty_path (str, optional): Path to the file with uncertainty results.
        output_dir (str, optional): Directory to save results.
        entropy_col (str): Name of the entropy column.
    
    Returns:
        dict: Dictionary with paths to the generated files.
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
    
    # Verify that we have a DataFrame to analyze
    if uncertainty_df is None:
        logger.error("No DataFrame or valid path provided")
        return None
    
    results = {}
    
    # Generate output paths
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        import datetime
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        vis_file = os.path.join(output_dir, f"site_entropy_distribution_{current_date}.png")
        stats_file = os.path.join(output_dir, f"site_statistics_{current_date}")
    else:
        vis_file = None
        stats_file = None
    
    # Generate visualization
    stats_df = plot_site_entropy_distribution(
        uncertainty_df, 
        output_file=vis_file,
        entropy_col=entropy_col
    )
    
    if vis_file:
        results['visualization'] = vis_file
    
    # Save statistics
    if stats_df is not None:
        stats_files = save_statistics(stats_df, output_path=stats_file)
        if stats_files:
            results['statistics'] = stats_files
    
    return results if results else None

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
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        results = generate_visualization(
            uncertainty_path=uncertainty_path,
            output_dir=output_dir
        )
        
        if results:
            print("Visualization and statistics generated correctly.")
            for key, value in results.items():
                print(f"{key}: {value}")
        else:
            print("Error generating visualization and statistics.")
    else:
        print("Usage: python visualization.py <uncertainty_file> [output_dir]")