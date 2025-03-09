#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script TO RUN THE COMPLETE REAL WORLD ANALYSIS PIPELINE
-----------------------------------------------------------------
Integrates prediction, uncertainty analysis, provenance determination
and visualization of results.

Example of use:
    python run_predictions_with_uncertainty_and_provenance.py
"""

import os
import sys
import datetime
import logging

# Ensure that the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.real_world.archaeological_predictor import ArchaeologicalPredictor
from src.real_world.uncertainty_analysis import process_predictions_with_uncertainty
from src.real_world.provenance_determination import process_provenance_determination
from src.real_world.visualization import generate_visualization

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"archaeological_pipeline_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ArchaeologicalPipeline")

# Configuration of paths - adjust according to the production structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
#BASE_DIR = '/home/dsg/VORTEX_FINAL/VORTEX'
DATA_PATH = os.path.join(BASE_DIR, 'DATA/real_world/real_world_data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'real_world_results')

# Create results directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate output file names
current_date = datetime.datetime.now().strftime("%Y%m%d")
prediction_file = os.path.join(OUTPUT_DIR, f"archaeological_predictions_{current_date}.xlsx")
uncertainty_file = os.path.join(OUTPUT_DIR, f"uncertainty_analysis_{current_date}.xlsx")
provenance_file = os.path.join(OUTPUT_DIR, f"provenance_analysis_{current_date}.xlsx")
visualization_file = os.path.join(OUTPUT_DIR, f"site_entropy_distribution_{current_date}.png")
statistics_file = os.path.join(OUTPUT_DIR, f"site_statistics_{current_date}")

def run_complete_pipeline(skip_uncertainty=False, skip_provenance=False, 
                         skip_visualization=False, confidence_threshold=0.7):
    """
    Runs the complete pipeline: predictions, uncertainty, provenance and visualization.
    
    Args:
        skip_uncertainty (bool): If True, skips the uncertainty analysis.
        skip_provenance (bool): If True, skips the provenance determination.
        skip_visualization (bool): If True, skips the visualization generation.
        confidence_threshold (float): Confidence threshold for analysis.
    
    Returns:
        bool: True if the process completed successfully, False otherwise.
    """
    print(f"Starting the complete archaeological analysis pipeline...")
    print(f"Data: {DATA_PATH}")
    print(f"Models: {MODELS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Variables to store intermediate DataFrames
    prediction_df = None
    uncertainty_df = None
    
    try:
        #-----------------------------------------------------------------
        # STEP 1: Generate predictions
        #-----------------------------------------------------------------
        logger.info("=== STARTING PREDICTION PIPELINE ===")
        print(f"\n📊 STEP 1: Generating real world  predictions...")

        predictor = ArchaeologicalPredictor(DATA_PATH, MODELS_DIR)
        prediction_path = predictor.run_prediction_pipeline(prediction_file)
        
        if not prediction_path:
            logger.error("Failed the prediction process. Aborting pipeline.")
            print("\n❌ The prediction process failed. Aborting pipeline.")
            return False
        
        logger.info(f"Predictions saved in: {prediction_path}")
        print(f"✅ Predictions generated correctly.")
        print(f"   File: {prediction_path}")
        
        #-----------------------------------------------------------------
        # STEP 2: Uncertainty analysis (optional)
        #-----------------------------------------------------------------
        if skip_uncertainty:
            logger.info("Uncertainty analysis skipped by configuration.")
            print("\n🔍 STEP 2: Uncertainty analysis skipped according to configuration.")
            
            if not skip_provenance or not skip_visualization:
                logger.error("Cannot perform subsequent analyses without uncertainty analysis.")
                print("\n❌ Cannot perform subsequent analyses without uncertainty analysis.")
                return True  # Return True because at least the predictions were generated
            
            return True
        
        logger.info("=== STARTING UNCERTAINTY ANALYSIS ===")
        print(f"\n🔍 STEP 2: Performing uncertainty analysis...")
        
        # Execute uncertainty analysis
        uncertainty_df = process_predictions_with_uncertainty(
            prediction_path=prediction_path,
            output_path=uncertainty_file,
            confidence_threshold=confidence_threshold
        )
        
        if uncertainty_df is None:
            logger.error("Failed the uncertainty analysis. Aborting subsequent analyses.")
            print("\n⚠️ The uncertainty analysis failed. Cannot continue.")
            return True  # Return True because at least the predictions were generated
        
        logger.info(f"Uncertainty analysis completed and saved in: {uncertainty_file}")
        print(f"✅ Uncertainty analysis completed.")
        print(f"   File: {uncertainty_file}")
        
        #-----------------------------------------------------------------
        # STEP 3: Provenance determination (optional)
        #-----------------------------------------------------------------
        if skip_provenance:
            logger.info("Provenance determination skipped by configuration.")
            print("\n🔎 STEP 3: Provenance determination skipped according to configuration.")
        else:
            logger.info("=== STARTING PROVENANCE DETERMINATION ===")
            print(f"\n🔎 STEP 3: Performing provenance determination...")
            
            # Execute provenance determination
            provenance_df = process_provenance_determination(
                uncertainty_df=uncertainty_df,
                output_path=provenance_file,
                confidence_threshold=confidence_threshold
            )
            
            if provenance_df is None:
                logger.error("Failed the provenance determination.")
                print("\n⚠️ The provenance determination failed.")
            else:
                logger.info(f"Provenance determination completed and saved in: {provenance_file}")
                print(f"✅ Provenance determination completed.")
                print(f"   File: {provenance_file}")
        
        #-----------------------------------------------------------------
        # STEP 4: Visualization (optional)
        #-----------------------------------------------------------------
        if skip_visualization:
            logger.info("Visualization generation skipped by configuration.")
            print("\n📈 STEP 4: Visualization generation skipped according to configuration.")
        else:
            logger.info("=== STARTING VISUALIZATION GENERATION ===")
            print(f"\n📈 STEP 4: Generating entropy visualizations...")
            
            # Generate visualizations
            vis_results = generate_visualization(
                uncertainty_df=uncertainty_df,
                output_dir=OUTPUT_DIR,
                entropy_col='Entropy'  
            )
            
            if not vis_results:
                logger.error("Failed the visualization generation.")
                print("\n⚠️ The visualization generation failed.")
            else:
                logger.info("Visualizations generated correctly.")
                print(f"✅ Visualizations generated correctly.")
                
                if 'visualization' in vis_results:
                    print(f"   Plot: {vis_results['visualization']}")
                
                if 'statistics' in vis_results and 'excel' in vis_results['statistics']:
                    print(f"   Statistics: {vis_results['statistics']['excel']}")
        
        print("\n🏆 COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")
        return True
        
    except Exception as e:
        logger.error(f"Error in the pipeline: {str(e)}")
        print(f"\n❌ Error executing the pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete archaeological analysis pipeline')
    parser.add_argument('--data', help='Path to the Excel file with archaeological data')
    parser.add_argument('--models', help='Directory containing trained models')
    parser.add_argument('--output', help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold (default: 0.7)')
    parser.add_argument('--no-uncertainty', action='store_true', help='Skip uncertainty analysis')
    parser.add_argument('--no-provenance', action='store_true', help='Skip provenance determination')
    parser.add_argument('--no-visualization', action='store_true', help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Update configuration if arguments are provided
    if args.data:
        DATA_PATH = args.data
    if args.models:
        MODELS_DIR = args.models
    if args.output:
        OUTPUT_DIR = args.output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Actualizar rutas de salida
        prediction_file = os.path.join(OUTPUT_DIR, f"archaeological_predictions_{current_date}.xlsx")
        uncertainty_file = os.path.join(OUTPUT_DIR, f"uncertainty_analysis_{current_date}.xlsx")
        provenance_file = os.path.join(OUTPUT_DIR, f"provenance_analysis_{current_date}.xlsx")
        visualization_file = os.path.join(OUTPUT_DIR, f"site_entropy_distribution_{current_date}.png")
        statistics_file = os.path.join(OUTPUT_DIR, f"site_statistics_{current_date}")
    
    # Execute pipeline
    success = run_complete_pipeline(
        skip_uncertainty=args.no_uncertainty,
        skip_provenance=args.no_provenance,
        skip_visualization=args.no_visualization,
        confidence_threshold=args.threshold
    )
    
    # Exit with appropriate code
    exit(0 if success else 1)
