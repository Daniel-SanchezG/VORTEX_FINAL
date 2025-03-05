#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar predicciones arqueológicas con análisis de incertidumbre
----------------------------------------------------------------------------
Script que integra el sistema de predicción y el análisis de incertidumbre.

Ejemplo de uso:
    python run_predictions_with_uncertainty.py
"""

import os
import sys
import datetime
import logging

# Asegurarse de que el directorio actual esté en el path de Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar los módulos necesarios
from archaeological_predictor import ArchaeologicalPredictor
from uncertainty_analysis import process_predictions_with_uncertainty

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"archaeological_pipeline_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ArchaeologicalPipeline")

# Configuración de rutas - ajustar según la estructura en producción
BASE_DIR = '/home/dsg/VORTEX_FINAL/PRODUCTION'
DATA_PATH = os.path.join(BASE_DIR, 'DATA/real_world/real_world_data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

# Crear directorio de resultados si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generar nombres de archivos de salida
current_date = datetime.datetime.now().strftime("%Y%m%d")
prediction_file = os.path.join(OUTPUT_DIR, f"archaeological_predictions_{current_date}.xlsx")
uncertainty_file = os.path.join(OUTPUT_DIR, f"uncertainty_analysis_{current_date}.xlsx")

def run_predictions_with_uncertainty(skip_uncertainty=False, confidence_threshold=0.7):
    """Ejecuta el proceso de predicción completo con análisis de incertidumbre opcional"""
    print(f"Iniciando proceso de predicción arqueológica...")
    print(f"Datos: {DATA_PATH}")
    print(f"Modelos: {MODELS_DIR}")
    print(f"Salida: {prediction_file}")
    
    try:
        # Paso 1: Generar predicciones
        predictor = ArchaeologicalPredictor(DATA_PATH, MODELS_DIR)
        result_path = predictor.run_prediction_pipeline(prediction_file)
        
        if not result_path:
            print(f"\n❌ El proceso de predicción falló.")
            print("   Consulta los logs para más detalles.")
            return False
            
        print(f"\n✅ Proceso de predicción completado con éxito.")
        print(f"📊 Predicciones guardadas en: {result_path}")
        
        # Paso 2: Análisis de incertidumbre (opcional)
        if skip_uncertainty:
            print("\n🔍 Análisis de incertidumbre omitido según configuración.")
            return True
        
        print("\n🔍 Iniciando análisis de incertidumbre...")
        
        # Ejecutar análisis de incertidumbre
        uncertainty_results = process_predictions_with_uncertainty(
            prediction_path=result_path,
            output_path=uncertainty_file,
            confidence_threshold=confidence_threshold
        )
        
        if uncertainty_results is None:
            print(f"\n⚠️ El análisis de incertidumbre falló, pero las predicciones están disponibles.")
            return True
        
        print(f"\n✅ Análisis de incertidumbre completado con éxito.")
        print(f"📊 Resultados guardados en: {uncertainty_file}")
        
        return True
            
    except Exception as e:
        print(f"\n❌ Error ejecutando el proceso: {str(e)}")
        logger.error(f"Error en el pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sistema de Predicción Arqueológica con Análisis de Incertidumbre')
    parser.add_argument('--data', help='Ruta al archivo Excel con datos arqueológicos')
    parser.add_argument('--models', help='Directorio que contiene los modelos entrenados')
    parser.add_argument('--output', help='Directorio para guardar resultados')
    parser.add_argument('--no-uncertainty', action='store_true', help='Omitir análisis de incertidumbre')
    parser.add_argument('--threshold', type=float, default=0.7, help='Umbral de confianza (0.7 por defecto)')
    
    args = parser.parse_args()
    
    # Actualizar configuración si se proporcionan argumentos
    if args.data:
        DATA_PATH = args.data
    if args.models:
        MODELS_DIR = args.models
    if args.output:
        OUTPUT_DIR = args.output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        prediction_file = os.path.join(OUTPUT_DIR, f"archaeological_predictions_{current_date}.xlsx")
        uncertainty_file = os.path.join(OUTPUT_DIR, f"uncertainty_analysis_{current_date}.xlsx")
    
    # Ejecutar pipeline
    success = run_predictions_with_uncertainty(
        skip_uncertainty=args.no_uncertainty,
        confidence_threshold=args.threshold
    )
    
    # Salir con código apropiado
    exit(0 if success else 1)