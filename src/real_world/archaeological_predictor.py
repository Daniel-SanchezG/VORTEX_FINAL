#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Archaeological Predictor Module
---------------------------------
Modular system to load archaeological data from different sites,
select appropriate models and generate consolidated predictions.

This script is designed to, generate a single Excel file with predictions for all sites.
"""



import os
import logging
import datetime
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"archaeological_predictions_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ArchaeologicalPredictor")

class ArchaeologicalPredictor:
    """
    Class to manage the loading of archaeological data, 
    model selection and prediction generation.
    """
    
    def __init__(self, data_path, models_dir):
        """
        Initializes the predictor with paths to data and models.
        
        Args:
            data_path (str): Path to the Excel file with archaeological data.
            models_dir (str): Directory containing trained models.
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.data_pool = {}
        self.model_pool = {}
        self.site_model_mapping = {
            'pq': 'PQModel',            # Quiruelas
            'vdh': 'VdHModel',          # V_Higueras
            'da': 'full_model',         # Alberite
            'pa': 'full_model',         # Paternanbidea
            'cg': 'full_model',         # Can_Gambus
            'cs': 'full_model',         # Can_Sandurni
            'cc': 'full_model',         # Cova_Cassinmanya
            'ls': 'full_model',         # La_Serreta
            'rl': 'full_model',         # Roca_Livet
            'a': 'FrenchModel',         # Auverne
            'j': 'FrenchModel',         # Josseliere
            'k': 'FrenchModel',         # Kervilor
            'l': 'FrenchModel',         # Luffang
            'p': 'FrenchModel',         # Plinchacourt
            'StM': 'FrenchModel'        # SaintMichel
                         
        }
        self.site_names = {
            'pq': 'Quiruelas',
            'vdh': 'V_Higueras',
            'da': 'Alberite',
            'pa': 'Paternanbidea',
            'cg': 'Can_Gambus',
            'ls': 'La_Serreta',
            'cs': 'Can_Sandurni',
            'cc': 'Cova_Cassinmanya',
            'rl': 'Roca_Livet',
            'a': 'Auverne',
            'j': 'Josseliere',
            'k': 'Kervilor',
            'l': 'Luffang',
            'p': 'Plinchacourt',
            'StM': 'SaintMichel'
            
        }
    
    def load_datasets(self):
        """
        Loads all datasets from the Excel file.
        Each sheet is a different archaeological site.
        
        Returns:
            bool: True if the loading was successful, False otherwise.
        """
        logger.info(f"Loading data from: {self.data_path}")
        try:
            sheets = {
                'pq': 'Quiruelas', 
                'vdh': 'V_higueras', 
                'da': 'Alberite', 
                'pa': 'Paternanbidea',
                'cg': 'Can_Gambus',
                'ls': 'La_Serreta',
                'cs': 'Can_Sandurni',
                'cc': 'Cova_Cassinmanya',
                'rl': 'Roca_Livet',
                'a': 'Auverne',
                'j': 'Josseliere',
                'k': 'Kervilor',
                'l': 'Luffang',
                'p': 'Plinchacourt',
                'StM': 'SaintMichel'
            }
            
            for key, sheet_name in sheets.items():
                try:
                    df = pd.read_excel(self.data_path, sheet_name=sheet_name, engine='openpyxl')
                    
                    # Save site information
                    site_name = self.site_names[key]
                    
                    # Extract metadata and prepare DataFrame for prediction
                    metadata = {}
                    
                    # Handle ID
                    
                    metadata['id'] = df['id'].tolist()
                    df = df.drop(columns=['id'])
                
                    # Handle Yac (yacimiento)
                    if 'Yac' in df.columns:
                        metadata['Yac'] = df['Yac'].tolist()
                    else:
                        metadata['Yac'] = [site_name] * len(df)
                    
                    # Save metadata and data separately
                    self.data_pool[key] = {
                        'data': df,
                        'metadata': metadata,
                        'site_name': site_name
                    }
                    
                    logger.info(f"Cargado sitio '{sheet_name}' con {len(df)} filas")
                except Exception as e:
                    logger.error(f"Error cargando sitio '{sheet_name}': {str(e)}")
            
            return len(self.data_pool) > 0
            
        except Exception as e:
            logger.error(f"Error crítico accediendo al archivo Excel: {str(e)}")
            return False
    
    def load_models(self):
        """
        Loads all necessary models for predictions.
        
        Returns:
            bool: True if the loading was successful, False otherwise.
        """
        logger.info(f"Loading models from: {self.models_dir}")
        
        model_paths = {
            'full_model': os.path.join(self.models_dir, 'final_model'),
            'VdHModel': os.path.join(self.models_dir, 'rf_VdH'),
            'PQModel': os.path.join(self.models_dir, 'rf_Quiruelas'),
            'FrenchModel': os.path.join(self.models_dir, 'rf_French')
        }
        
        success = True
        for model_name, model_path in model_paths.items():
            try:
                model = load_model(model_path) #Using Pycaret's load_model function
                self.model_pool[model_name] = model
                logger.info(f"Loaded model '{model_name}' from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model '{model_name}': {str(e)}")
                success = False
        
        return success and len(self.model_pool) > 0
    
    def make_predictions(self):
        """
        Makes predictions for all sites using the corresponding models.
        
        Returns:
            pd.DataFrame: Consolidated DataFrame with all predictions
        """
        all_predictions = []
        
        for site_key, site_data in self.data_pool.items():
            df = site_data['data']
            metadata = site_data['metadata']
            site_name = site_data['site_name']
            
            model_name = self.site_model_mapping.get(site_key, 'full_model')
            model = self.model_pool.get(model_name)
            
            if model is None:
                logger.warning(f"No model found for site {site_key}, using full_model")
                model = self.model_pool.get('full_model')
                
                if model is None:
                    logger.error(f"Cannot make predictions for {site_key}: model not available")
                    continue
            
            logger.info(f"Making predictions for {site_key} using {model_name}")
            
            try:
                # Check and add columns that the model might expect
                try:
                    model_cols = model.get_params().get('feature_names', [])
                    if model_cols:
                        for col in model_cols:
                            if col not in df.columns and col != 'Yac' and col != 'id' and col != 'Site':
                                logger.warning(f"Adding missing column '{col}' for {site_key}")
                                df[col] = 0
                except Exception as e:
                    logger.warning(f"Could not check model columns: {str(e)}")
                
                # Make prediction with scores
                
                predictions = predict_model(model, data=df, raw_score=True) #Using Pycaret's predict_model function
                
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
                    result_df['Prediction'] = predictions[prediction_col]
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
                
                # Add to the results set
                all_predictions.append(result_df)
                
                logger.info(f"Completed predictions for {site_key}: {len(result_df)} rows")
                
            except Exception as e:
                logger.error(f"Error making predictions for {site_key}: {str(e)}")
        
        # Consolidate all results
        if not all_predictions:
            logger.error("No predictions generated for any site")
            return pd.DataFrame()
        
        consolidated_results = pd.concat(all_predictions, ignore_index=True)
        
        # Ensure all required columns exist
        required_columns = ['id', 'Site', 'Yac', 'Prediction', 
                          'prediction_score_CT', 'prediction_score_PCM', 'prediction_score_PDLC']
        
        for col in required_columns:
            if col not in consolidated_results.columns:
                consolidated_results[col] = np.nan
        
        # Reorder columns
        consolidated_results = consolidated_results[required_columns]
        
        logger.info(f"Consolidadas todas las predicciones: {len(consolidated_results)} filas en total")
        
        return consolidated_results
    
    def run_prediction_pipeline(self, output_path=None):
        """
        Executes the entire prediction process and saves the results.
        
        Args:
            output_path (str, optional): Path to save the results file.
                                         If None, it will be generated automatically.
        
        Returns:
            str: Path to the results file or None if failed.
        """
        # Generate output path if not provided
        if output_path is None:
            base_dir = os.path.dirname(self.data_path)
            base_name = os.path.splitext(os.path.basename(self.data_path))[0]
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(base_dir, f"{base_name}_{current_date}_consolidated_predictions.xlsx")
        
        # Execute pipeline
        logger.info("Starting prediction pipeline")
        
        if not self.load_datasets():
            logger.error("Data loading failed. Aborting pipeline.")
            return None
        
        if not self.load_models():
            logger.error("Model loading failed. Aborting pipeline.")
            return None
        
        results = self.make_predictions()
        
        if results.empty:
            logger.error("No results generated. Aborting pipeline.")
            return None
        
        # Save results
        try:
            results.to_excel(output_path, index=False)
            logger.info(f"Results saved in: {output_path}")
            
            # Also save as CSV
            csv_path = output_path.replace('.xlsx', '.csv')
            results.to_csv(csv_path, index=False)
            logger.info(f"Results also saved in: {csv_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return None


def main():
    """
    Main function to execute the system from the command line.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Prediction Process')
    parser.add_argument('--data', required=True, help='Path to the Excel file with real world data')
    parser.add_argument('--models', required=True, help='Directory containing trained models')
    parser.add_argument('--output', help='Path to the results file (optional)')
    
    args = parser.parse_args()
    
    predictor = ArchaeologicalPredictor(args.data, args.models)
    output_file = predictor.run_prediction_pipeline(args.output)
    
    if output_file:
        print(f"Pipeline completed successfully. Results in: {output_file}")
        return 0
    else:
        print("The pipeline failed. Check the logs for more details.")
        return 1


if __name__ == "__main__":
    exit(main())