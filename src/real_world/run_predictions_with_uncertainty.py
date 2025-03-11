#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run archaeological predictions with uncertainty analysis
-------------------------------------------------------------------
Script that integrates the prediction system and uncertainty analysis.

Example of use:
    python run_predictions_with_uncertainty.py
"""

import os
import sys
import datetime
import logging

# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the necessary modules
from archaeological_predictor import ArchaeologicalPredictor
from uncertainty_analysis import process_predictions_with_uncertainty

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"archaeological_pipeline_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ArchaeologicalPipeline")

# Configure paths - adjust according to production structure
BASE_DIR = '/home/dsg/VORTEX_FINAL/PRODUCTION'
DATA_PATH = os.path.join(BASE_DIR, 'DATA/real_world/real_world_data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

# Create results directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate output file names
current_date = datetime.datetime.now().strftime("%Y%m%d")
prediction_file = os.path.join(OUTPUT_DIR, f"archaeological_predictions_{current_date}.xlsx")
uncertainty_file = os.path.join(OUTPUT_DIR, f"uncertainty_analysis_{current_date}.xlsx")

def run_predictions_with_uncertainty(skip_uncertainty=False, confidence_threshold=0.7):
    """Run the complete prediction process with optional uncertainty analysis"""
    print(f"Starting archaeological prediction process...")
    print(f"Data: {DATA_PATH}")
    print(f"Models: {MODELS_DIR}")
    print(f"Output: {prediction_file}")
    
    try:
        # Step 1: Generate predictions
        predictor = ArchaeologicalPredictor(DATA_PATH, MODELS_DIR)
        result_path = predictor.run_prediction_pipeline(prediction_file)
        
        if not result_path:
            print(f"\n❌ The prediction process failed.")
            print("   Check the logs for more details.")
            return False
            
        print(f"\n✅ The prediction process completed successfully.")
        print(f"📊 Predictions saved in: {result_path}")
        
        # Paso 2: Análisis de incertidumbre (opcional)
        if skip_uncertainty:
            print("\n🔍 Uncertainty analysis skipped according to configuration.")
            return True
        
        print("\n🔍 Starting uncertainty analysis...")
        
        # Run uncertainty analysis
        uncertainty_results = process_predictions_with_uncertainty(
            prediction_path=result_path,
            output_path=uncertainty_file,
            confidence_threshold=confidence_threshold
        )
        
        if uncertainty_results is None:
            print(f"\n⚠️ The uncertainty analysis failed, but the predictions are available.")
            return True
        
        print(f"\n✅ Uncertainty analysis completed successfully.")
        print(f"📊 Results saved in: {uncertainty_file}")
        
        return True
            
    except Exception as e:
        print(f"\n❌ Error running the process: {str(e)}")
        logger.error(f"Error in the pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Archaeological Prediction System with Uncertainty Analysis')
    parser.add_argument('--data', help='Path to the Excel file with archaeological data')
    parser.add_argument('--models', help='Directory containing trained models')
    parser.add_argument('--output', help='Directory to save results')
    parser.add_argument('--no-uncertainty', action='store_true', help='Skip uncertainty analysis')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold (default: 0.7)')
    
    args = parser.parse_args()
    
    # Update configuration if arguments are provided
    if args.data:
        DATA_PATH = args.data
    if args.models:
        MODELS_DIR = args.models
    if args.output:
        OUTPUT_DIR = args.output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        prediction_file = os.path.join(OUTPUT_DIR, f"archaeological_predictions_{current_date}.xlsx")
        uncertainty_file = os.path.join(OUTPUT_DIR, f"uncertainty_analysis_{current_date}.xlsx")
    
    # Run pipeline
    success = run_predictions_with_uncertainty(
        skip_uncertainty=args.no_uncertainty,
        confidence_threshold=args.threshold
    )
    
    # Exit with appropriate code
    exit(0 if success else 1)