#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run archaeological predictions
-----------------------------------------------
Simple script to run the prediction system in production.

Example of use:
    python run_predictions.py
"""

import os
import sys

# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the class from the module
from archaeological_predictor import ArchaeologicalPredictor

# Configure paths - adjust according to production structure
BASE_DIR = '/home/dsg/VORTEX_FINAL/PRODUCTION'
DATA_PATH = os.path.join(BASE_DIR, 'DATA/real_world/real_world_data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

# Create results directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate output path
import datetime
output_file = os.path.join(
    OUTPUT_DIR, 
    f"archaeological_predictions_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
)

def run_predictions():
    """Run the complete prediction process"""
    print(f"Starting archaeological prediction process...")
    print(f"Data: {DATA_PATH}")
    print(f"Models: {MODELS_DIR}")
    print(f"Output: {output_file}")
    
    try:
        # Initialize and run the predictor
        predictor = ArchaeologicalPredictor(DATA_PATH, MODELS_DIR)
        result_path = predictor.run_prediction_pipeline(output_file)
        
        if result_path:
            print(f"\n✅ Process completed successfully.")
            print(f"📊 Results saved in: {result_path}")
            return True
        else:
            print(f"\n❌ The prediction process failed.")
            print("   Check the logs for more details.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running the process: {str(e)}")
        return False

if __name__ == "__main__":
    run_predictions()