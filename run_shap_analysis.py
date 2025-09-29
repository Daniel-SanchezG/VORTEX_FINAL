#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run SHAP analysis on Excel files.
This script can be run independently from the main flow
to generate SHAP visualizations and feature importance analysis.
"""

import argparse
import os
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import shap
import sys

# Import the ShapAnalyzer class
from src.analysis.shap_analyzer import ShapAnalyzer

def setup_logging(output_dir):
    """Configure the logging system for SHAP analysis."""
    # Create directory if it doesn't exist
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure the logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'shap_analysis_{timestamp}.log'
    
    # Log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def parse_arguments():
    """Analyze command line arguments for SHAP analysis."""
    parser = argparse.ArgumentParser(
        description='Independent SHAP analysis for Excel files'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the Excel file with the data to analyze'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./shap_results',
        help='Directory where the analysis results will be saved'
    )
    parser.add_argument(
        '--sheets',
        type=str,
        nargs='+',
        default=None,
        help='Sheet names to analyze (default: all)'
    )
    parser.add_argument(
        '--top-features',
        type=int,
        default=20,
        help='Number of top features to show in plots'
    )
    return parser.parse_args()

def main():
    """Main function for independent SHAP analysis."""
    # Analyze arguments
    args = parse_arguments()
    
    # Configure directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = setup_logging(output_dir)
    
    # Register start
    logger.info("Starting independent SHAP analysis")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Verify that the input file exists
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"The input file {input_path} does not exist")
            sys.exit(1)
        
        # Create the SHAP analyzer
        analyzer = ShapAnalyzer(output_dir=output_dir)
        
        # Get list of sheets if not specified
        if args.sheets is None:
            try:
                excel = pd.ExcelFile(input_path)
                sheets = excel.sheet_names
                logger.info(f"Sheets found in the file: {', '.join(sheets)}")
            except Exception as e:
                logger.error(f"Error reading Excel file sheets: {str(e)}")
                sys.exit(1)
        else:
            sheets = args.sheets
            logger.info(f"Analyzing specified sheets: {', '.join(sheets)}")
        
        # Execute the analysis
        results = analyzer.analyze_multiple_sheets(
            excel_path=input_path,
            sheet_names=sheets
        )
        
        # Show summary of results
        if 'combined_importance' in results:
            combined_df = results['combined_importance']
            top_n = min(args.top_features, len(combined_df))
            
            logger.info(f"\nTop {top_n} most important features (average):")
            for i, (_, row) in enumerate(combined_df.head(top_n).iterrows()):
                if 'avg_importance' in row:
                    logger.info(f"{i+1}. {row['feature']}: {row['avg_importance']:.4f}")
        
        logger.info(f"\nSHAP analysis completed successfully.")
        logger.info(f"Results saved in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during SHAP analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()