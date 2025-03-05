#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de Predicción Arqueológica
---------------------------------
Sistema modular para cargar datos arqueológicos de diferentes sitios,
seleccionar modelos apropiados y generar predicciones consolidadas.

Este script está diseñado para ser implementado en producción,
generando un archivo Excel único con predicciones para todos los sitios.
"""

# Este archivo debe guardarse como archaeological_predictor.py

import os
import logging
import datetime
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model

# Configuración de logging
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
    Clase para gestionar la carga de datos arqueológicos, 
    selección de modelos y generación de predicciones.
    """
    
    def __init__(self, data_path, models_dir):
        """
        Inicializa el predictor con rutas a datos y modelos.
        
        Args:
            data_path (str): Ruta al archivo Excel con datos arqueológicos.
            models_dir (str): Directorio que contiene los modelos entrenados.
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
            'cs': 'full_model',         # CatalonianSites
            'fs': 'FrenchModel'         # FrenchSites
        }
        self.site_names = {
            'pq': 'Quiruelas',
            'vdh': 'V_Higueras',
            'da': 'Alberite',
            'pa': 'Paternanbidea',
            'cg': 'Can_Gambus',
            'cs': 'CatalonianSites',
            'fs': 'FrenchSites'
        }
    
    def load_datasets(self):
        """
        Carga todos los conjuntos de datos desde el archivo Excel.
        Cada hoja es un sitio arqueológico diferente.
        
        Returns:
            bool: True si la carga fue exitosa, False en caso contrario.
        """
        logger.info(f"Cargando datos desde: {self.data_path}")
        try:
            sheets = {
                'pq': 'quiruelas', 
                'vdh': 'v_higueras', 
                'da': 'Alberite', 
                'pa': 'Paternanbidea',
                'cg': 'Can_Gambus', 
                'cs': 'CatalonianSites', 
                'fs': 'FrenchSites'
            }
            
            for key, sheet_name in sheets.items():
                try:
                    df = pd.read_excel(self.data_path, sheet_name=sheet_name, engine='openpyxl')
                    
                    # Guardar información del sitio
                    site_name = self.site_names[key]
                    
                    # Extraer metadatos y preparar DataFrame para predicción
                    metadata = {}
                    
                    # Manejar ID
                    if 'id' not in df.columns:
                        metadata['id'] = [f"{key}_{i}" for i in range(len(df))]
                    else:
                        metadata['id'] = df['id'].tolist()
                        df = df.drop(columns=['id'])
                    
                    # Manejar Yac (yacimiento)
                    if 'Yac' in df.columns:
                        metadata['Yac'] = df['Yac'].tolist()
                    else:
                        metadata['Yac'] = [site_name] * len(df)
                    
                    # Guardar metadatos y datos por separado
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
        Carga todos los modelos necesarios para las predicciones.
        
        Returns:
            bool: True si la carga fue exitosa, False en caso contrario.
        """
        logger.info(f"Cargando modelos desde: {self.models_dir}")
        
        model_paths = {
            'full_model': os.path.join(self.models_dir, 'final_model'),
            'VdHModel': os.path.join(self.models_dir, '20250227_VdHSpecific'),
            'PQModel': os.path.join(self.models_dir, '20250227_QuiruelasSpecific'),
            'FrenchModel': os.path.join(self.models_dir, '20250227_FrenchSpecific')
        }
        
        success = True
        for model_name, model_path in model_paths.items():
            try:
                model = load_model(model_path)
                self.model_pool[model_name] = model
                logger.info(f"Cargado modelo '{model_name}' desde {model_path}")
            except Exception as e:
                logger.error(f"Error cargando modelo '{model_name}': {str(e)}")
                success = False
        
        return success and len(self.model_pool) > 0
    
    def make_predictions(self):
        """
        Realiza predicciones para todos los sitios usando los modelos correspondientes.
        
        Returns:
            pd.DataFrame: DataFrame consolidado con todas las predicciones
        """
        all_predictions = []
        
        for site_key, site_data in self.data_pool.items():
            df = site_data['data']
            metadata = site_data['metadata']
            site_name = site_data['site_name']
            
            model_name = self.site_model_mapping.get(site_key, 'full_model')
            model = self.model_pool.get(model_name)
            
            if model is None:
                logger.warning(f"No se encontró modelo para el sitio {site_key}, usando full_model")
                model = self.model_pool.get('full_model')
                
                if model is None:
                    logger.error(f"No se puede hacer predicción para {site_key}: modelo no disponible")
                    continue
            
            logger.info(f"Haciendo predicciones para {site_key} usando {model_name}")
            
            try:
                # Comprobar y añadir columnas que el modelo podría esperar
                try:
                    model_cols = model.get_params().get('feature_names', [])
                    if model_cols:
                        for col in model_cols:
                            if col not in df.columns and col != 'Yac' and col != 'id' and col != 'Site':
                                logger.warning(f"Añadiendo columna faltante '{col}' para {site_key}")
                                df[col] = 0
                except Exception as e:
                    logger.warning(f"No se pudieron verificar columnas del modelo: {str(e)}")
                
                # Realizar predicción con puntuaciones
                predictions = predict_model(model, data=df, raw_score=True)
                
                # Crear un nuevo DataFrame para los resultados
                result_df = pd.DataFrame()
                
                # Añadir metadatos
                result_df['id'] = metadata['id']
                result_df['Site'] = site_name
                result_df['Yac'] = metadata['Yac']
                
                # Manejar diferentes convenciones de nombres de columnas de predicción
                prediction_col = None
                if 'Label' in predictions.columns:
                    prediction_col = 'Label'
                elif 'prediction_label' in predictions.columns:
                    prediction_col = 'prediction_label'
                
                if prediction_col:
                    result_df['Prediction'] = predictions[prediction_col]
                else:
                    result_df['Prediction'] = "Unknown"
                
                # Extraer y renombrar columnas de puntuación si existen
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
                
                # Añadir al conjunto de resultados
                all_predictions.append(result_df)
                
                logger.info(f"Completadas predicciones para {site_key}: {len(result_df)} filas")
                
            except Exception as e:
                logger.error(f"Error haciendo predicciones para {site_key}: {str(e)}")
        
        # Consolidar todos los resultados
        if not all_predictions:
            logger.error("No se generaron predicciones para ningún sitio")
            return pd.DataFrame()
        
        consolidated_results = pd.concat(all_predictions, ignore_index=True)
        
        # Asegurar que existan todas las columnas requeridas
        required_columns = ['id', 'Site', 'Yac', 'Prediction', 
                          'prediction_score_CT', 'prediction_score_PCM', 'prediction_score_PDLC']
        
        for col in required_columns:
            if col not in consolidated_results.columns:
                consolidated_results[col] = np.nan
        
        # Reordenar columnas
        consolidated_results = consolidated_results[required_columns]
        
        logger.info(f"Consolidadas todas las predicciones: {len(consolidated_results)} filas en total")
        
        return consolidated_results
    
    def run_prediction_pipeline(self, output_path=None):
        """
        Ejecuta todo el proceso de predicción y guarda los resultados.
        
        Args:
            output_path (str, optional): Ruta para guardar el archivo de resultados.
                                         Si es None, se genera automáticamente.
        
        Returns:
            str: Ruta al archivo de resultados o None si falló.
        """
        # Generar ruta de salida si no se proporciona
        if output_path is None:
            base_dir = os.path.dirname(self.data_path)
            base_name = os.path.splitext(os.path.basename(self.data_path))[0]
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(base_dir, f"{base_name}_{current_date}_consolidated_predictions.xlsx")
        
        # Ejecutar pipeline
        logger.info("Iniciando pipeline de predicción")
        
        if not self.load_datasets():
            logger.error("Falló la carga de datos. Abortando pipeline.")
            return None
        
        if not self.load_models():
            logger.error("Falló la carga de modelos. Abortando pipeline.")
            return None
        
        results = self.make_predictions()
        
        if results.empty:
            logger.error("No se generaron resultados. Abortando pipeline.")
            return None
        
        # Guardar resultados
        try:
            results.to_excel(output_path, index=False)
            logger.info(f"Resultados guardados en: {output_path}")
            
            # También guardar como CSV
            csv_path = output_path.replace('.xlsx', '.csv')
            results.to_csv(csv_path, index=False)
            logger.info(f"Resultados guardados también en: {csv_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error guardando resultados: {str(e)}")
            return None


def main():
    """
    Función principal para ejecutar el sistema desde línea de comandos.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Predicción Arqueológica')
    parser.add_argument('--data', required=True, help='Ruta al archivo Excel con datos arqueológicos')
    parser.add_argument('--models', required=True, help='Directorio que contiene los modelos entrenados')
    parser.add_argument('--output', help='Ruta para el archivo de resultados (opcional)')
    
    args = parser.parse_args()
    
    predictor = ArchaeologicalPredictor(args.data, args.models)
    output_file = predictor.run_prediction_pipeline(args.output)
    
    if output_file:
        print(f"Pipeline completado con éxito. Resultados en: {output_file}")
        return 0
    else:
        print("El pipeline falló. Consulte los logs para más detalles.")
        return 1


if __name__ == "__main__":
    exit(main())