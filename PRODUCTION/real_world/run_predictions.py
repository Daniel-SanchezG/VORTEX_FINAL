#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar predicciones arqueológicas
-----------------------------------------------
Script sencillo para ejecutar el sistema de predicción en producción.

Ejemplo de uso:
    python run_predictions.py
"""

import os
import sys

# Asegurarse de que el directorio actual esté en el path de Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar la clase del módulo
from archaeological_predictor import ArchaeologicalPredictor

# Configuración de rutas - ajustar según la estructura en producción
BASE_DIR = '/home/dsg/VORTEX_FINAL/PRODUCTION'
DATA_PATH = os.path.join(BASE_DIR, 'DATA/real_world/real_world_data.xlsx')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

# Crear directorio de resultados si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generar ruta de salida
import datetime
output_file = os.path.join(
    OUTPUT_DIR, 
    f"archaeological_predictions_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
)

def run_predictions():
    """Ejecuta el proceso de predicción completo"""
    print(f"Iniciando proceso de predicción arqueológica...")
    print(f"Datos: {DATA_PATH}")
    print(f"Modelos: {MODELS_DIR}")
    print(f"Salida: {output_file}")
    
    try:
        # Inicializar y ejecutar el predictor
        predictor = ArchaeologicalPredictor(DATA_PATH, MODELS_DIR)
        result_path = predictor.run_prediction_pipeline(output_file)
        
        if result_path:
            print(f"\n✅ Proceso completado con éxito.")
            print(f"📊 Resultados guardados en: {result_path}")
            return True
        else:
            print(f"\n❌ El proceso de predicción falló.")
            print("   Consulta los logs para más detalles.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error ejecutando el proceso: {str(e)}")
        return False

if __name__ == "__main__":
    run_predictions()