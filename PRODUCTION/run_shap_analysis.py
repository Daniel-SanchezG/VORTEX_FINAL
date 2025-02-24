#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script independiente para ejecutar análisis SHAP en archivos Excel.
Este script puede ejecutarse de forma independiente del flujo principal
para generar visualizaciones SHAP y análisis de importancia de características.
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

# Importamos la clase ShapAnalyzer
from src.analysis.shap_analyzer import ShapAnalyzer

def setup_logging(output_dir):
    """Configura el sistema de logging para el análisis SHAP."""
    # Crear directorio si no existe
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar el logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'shap_analysis_{timestamp}.log'
    
    # Formato del log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Configurar logger raíz
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def parse_arguments():
    """Analiza los argumentos de línea de comandos para el análisis SHAP."""
    parser = argparse.ArgumentParser(
        description='Análisis SHAP independiente para archivos Excel'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Ruta al archivo Excel con los datos a analizar'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./shap_results',
        help='Directorio donde se guardarán los resultados del análisis'
    )
    parser.add_argument(
        '--sheets',
        type=str,
        nargs='+',
        default=None,
        help='Nombres de las hojas a analizar (por defecto: todas)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Ruta a un modelo pre-entrenado guardado (opcional)'
    )
    parser.add_argument(
        '--top-features',
        type=int,
        default=20,
        help='Número de características principales a mostrar en gráficos'
    )
    return parser.parse_args()

def main():
    """Función principal para el análisis SHAP independiente."""
    # Analizar argumentos
    args = parse_arguments()
    
    # Configurar directorios
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar logging
    logger = setup_logging(output_dir)
    
    # Registrar inicio
    logger.info("Iniciando análisis SHAP independiente")
    logger.info(f"Archivo de entrada: {args.input}")
    logger.info(f"Directorio de salida: {output_dir}")
    
    try:
        # Verificar que el archivo de entrada existe
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"El archivo de entrada {input_path} no existe")
            sys.exit(1)
        
        # Cargar modelo pre-entrenado si se proporciona
        model = None
        if args.model_path:
            model_path = Path(args.model_path)
            if model_path.exists():
                logger.info(f"Cargando modelo pre-entrenado desde {model_path}")
                try:
                    import joblib
                    model = joblib.load(model_path)
                    logger.info("Modelo cargado exitosamente")
                except Exception as e:
                    logger.error(f"Error al cargar el modelo: {str(e)}")
                    logger.warning("Continuando sin modelo pre-entrenado")
            else:
                logger.warning(f"La ruta del modelo {model_path} no existe. Continuando sin modelo pre-entrenado")
        
        # Crear el analizador SHAP
        analyzer = ShapAnalyzer(output_dir=output_dir)
        
        # Obtener lista de hojas si no se especifican
        if args.sheets is None:
            try:
                excel = pd.ExcelFile(input_path)
                sheets = excel.sheet_names
                logger.info(f"Hojas encontradas en el archivo: {', '.join(sheets)}")
            except Exception as e:
                logger.error(f"Error al leer hojas del archivo Excel: {str(e)}")
                sys.exit(1)
        else:
            sheets = args.sheets
            logger.info(f"Analizando hojas especificadas: {', '.join(sheets)}")
        
        # Ejecutar el análisis
        results = analyzer.analyze_multiple_sheets(
            excel_path=input_path,
            sheet_names=sheets,
            model=model
        )
        
        # Mostrar resumen de resultados
        if 'combined_importance' in results:
            combined_df = results['combined_importance']
            top_n = min(args.top_features, len(combined_df))
            
            logger.info(f"\nTop {top_n} características más importantes (promedio):")
            for i, (_, row) in enumerate(combined_df.head(top_n).iterrows()):
                if 'avg_importance' in row:
                    logger.info(f"{i+1}. {row['feature']}: {row['avg_importance']:.4f}")
        
        logger.info(f"\nAnálisis SHAP completado con éxito.")
        logger.info(f"Resultados guardados en: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error durante el análisis SHAP: {str(e)}")
        raise

if __name__ == "__main__":
    main()