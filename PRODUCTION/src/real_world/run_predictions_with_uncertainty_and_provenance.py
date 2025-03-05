#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar el pipeline completo de análisis arqueológico
-----------------------------------------------------------------
Integra predicción, análisis de incertidumbre y determinación de procedencia.

Ejemplo de uso:
    python run_predictions_with_uncertainty_and_provenance.py
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
from provenance_determination import process_provenance_determination

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
provenance_file = os.path.join(OUTPUT_DIR, f"provenance_analysis_{current_date}.xlsx")

def run_complete_pipeline(skip_uncertainty=False, skip_provenance=False, confidence_threshold=0.7):
    """
    Ejecuta el pipeline completo: predicciones, incertidumbre y procedencia.
    
    Args:
        skip_uncertainty (bool): Si True, omite el análisis de incertidumbre.
        skip_provenance (bool): Si True, omite la determinación de procedencia.
        confidence_threshold (float): Umbral de confianza para análisis.
    
    Returns:
        bool: True si el proceso completó con éxito, False en caso contrario.
    """
    print(f"Iniciando pipeline completo de análisis arqueológico...")
    print(f"Datos: {DATA_PATH}")
    print(f"Modelos: {MODELS_DIR}")
    print(f"Salida: {OUTPUT_DIR}")
    
    # Variables para almacenar DataFrames intermedios
    prediction_df = None
    uncertainty_df = None
    
    try:
        #-----------------------------------------------------------------
        # PASO 1: Generar predicciones
        #-----------------------------------------------------------------
        logger.info("=== INICIANDO SISTEMA DE PREDICCIÓN ARQUEOLÓGICA ===")
        print(f"\n📊 PASO 1: Generando predicciones arqueológicas...")
        
        predictor = ArchaeologicalPredictor(DATA_PATH, MODELS_DIR)
        prediction_path = predictor.run_prediction_pipeline(prediction_file)
        
        if not prediction_path:
            logger.error("Falló el proceso de predicción arqueológica. Abortando pipeline.")
            print("\n❌ El proceso de predicción falló. Abortando pipeline.")
            return False
        
        logger.info(f"Predicciones guardadas en: {prediction_path}")
        print(f"✅ Predicciones generadas correctamente.")
        print(f"   Archivo: {prediction_path}")
        
        #-----------------------------------------------------------------
        # PASO 2: Análisis de incertidumbre (opcional)
        #-----------------------------------------------------------------
        if skip_uncertainty:
            logger.info("Análisis de incertidumbre omitido por configuración.")
            print("\n🔍 PASO 2: Análisis de incertidumbre omitido según configuración.")
            
            if skip_provenance:
                return True
            else:
                logger.error("No se puede realizar análisis de procedencia sin análisis de incertidumbre.")
                print("\n❌ No se puede realizar análisis de procedencia sin análisis de incertidumbre.")
                return False
        
        logger.info("=== INICIANDO ANÁLISIS DE INCERTIDUMBRE ===")
        print(f"\n🔍 PASO 2: Realizando análisis de incertidumbre...")
        
        # Ejecutar análisis de incertidumbre
        uncertainty_df = process_predictions_with_uncertainty(
            prediction_path=prediction_path,
            output_path=uncertainty_file,
            confidence_threshold=confidence_threshold
        )
        
        if uncertainty_df is None:
            logger.error("Falló el análisis de incertidumbre. Abortando procedencia.")
            print("\n⚠️ El análisis de incertidumbre falló. No se puede continuar con la procedencia.")
            return True  # Retornar True porque al menos las predicciones se generaron
        
        logger.info(f"Análisis de incertidumbre completado y guardado en: {uncertainty_file}")
        print(f"✅ Análisis de incertidumbre completado.")
        print(f"   Archivo: {uncertainty_file}")
        
        #-----------------------------------------------------------------
        # PASO 3: Determinación de procedencia (opcional)
        #-----------------------------------------------------------------
        if skip_provenance:
            logger.info("Determinación de procedencia omitida por configuración.")
            print("\n🔎 PASO 3: Determinación de procedencia omitida según configuración.")
            return True
        
        logger.info("=== INICIANDO DETERMINACIÓN DE PROCEDENCIA ===")
        print(f"\n🔎 PASO 3: Realizando determinación de procedencia...")
        
        # Ejecutar determinación de procedencia
        provenance_df = process_provenance_determination(
            uncertainty_df=uncertainty_df,
            output_path=provenance_file,
            confidence_threshold=confidence_threshold
        )
        
        if provenance_df is None:
            logger.error("Falló la determinación de procedencia.")
            print("\n⚠️ La determinación de procedencia falló.")
            return True  # Retornar True porque al menos las predicciones e incertidumbre se generaron
        
        logger.info(f"Determinación de procedencia completada y guardada en: {provenance_file}")
        print(f"✅ Determinación de procedencia completada.")
        print(f"   Archivo: {provenance_file}")
        
        print("\n🏆 PIPELINE COMPLETO EJECUTADO CON ÉXITO!")
        return True
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}")
        print(f"\n❌ Error ejecutando el pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline completo de análisis arqueológico')
    parser.add_argument('--data', help='Ruta al archivo Excel con datos arqueológicos')
    parser.add_argument('--models', help='Directorio que contiene los modelos entrenados')
    parser.add_argument('--output', help='Directorio para guardar resultados')
    parser.add_argument('--threshold', type=float, default=0.7, help='Umbral de confianza (por defecto: 0.7)')
    parser.add_argument('--no-uncertainty', action='store_true', help='Omitir análisis de incertidumbre')
    parser.add_argument('--no-provenance', action='store_true', help='Omitir determinación de procedencia')
    
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
        provenance_file = os.path.join(OUTPUT_DIR, f"provenance_analysis_{current_date}.xlsx")
    
    # Ejecutar pipeline
    success = run_complete_pipeline(
        skip_uncertainty=args.no_uncertainty,
        skip_provenance=args.no_provenance,
        confidence_threshold=args.threshold
    )
    
    # Salir con código apropiado
    exit(0 if success else 1)