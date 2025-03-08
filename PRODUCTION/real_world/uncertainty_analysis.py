#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de Análisis de Incertidumbre
-----------------------------------
Complemento para el sistema de predicción arqueológica que añade
análisis de incertidumbre a las predicciones generadas.
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
import logging
import os
import datetime

# Configuración de logging
logger = logging.getLogger("Uncertainty")

def analyze_uncertainty(prediction_df, confidence_threshold=0.7):
    """
    Realiza análisis de incertidumbre sobre un DataFrame de predicciones.

    Args:
        prediction_df (pd.DataFrame): DataFrame con predicciones y puntuaciones.
        confidence_threshold (float): Umbral de confianza para marcar como incierta.

    Returns:
        pd.DataFrame: DataFrame con resultados del análisis.
    """
    logger.info("Iniciando análisis de incertidumbre...")
    
    try:
        # Copiar DataFrame para no modificar el original
        df = prediction_df.copy()
        
        # Identificar columnas de puntuación
        score_cols = [col for col in df.columns if col.startswith('prediction_score_')]
        
        if not score_cols:
            logger.warning("No se encontraron columnas de puntuación.")
            return None
        
        logger.info(f"Usando columnas de puntuación: {score_cols}")
        
        # Convertir score_cols a números si son strings
        for col in score_cols:
            if df[col].dtype == object:  # Si es string
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Obtener probabilidades
        probas = df[score_cols].values
        
        # Obtener etiquetas de predicción y confianza
        prediction_col = 'Prediction' if 'Prediction' in df.columns else 'predicciones'
        predictions = df[prediction_col].values
        confidences = np.max(probas, axis=1)
        
        # Marcar predicciones bajo el umbral como inciertas
        uncertain_mask = confidences < confidence_threshold
        predictions_with_uncertainty = predictions.copy()
        predictions_with_uncertainty[uncertain_mask] = 'uncertain'
        
        # Calcular entropía
        entropies = np.array([entropy(probs, base=2) for probs in probas])
        
        # Crear DataFrame con resultados
        results_df = df.copy()
        
        # Añadir nuevas columnas de análisis
        results_df['Original_predictions'] = predictions
        results_df['Confidence'] = confidences
        results_df['Uncertainty_threshold_predictions'] = predictions_with_uncertainty
        results_df['Entropy'] = entropies
        
        # Calcular métricas globales
        n_uncertain = np.sum(uncertain_mask)
        uncertain_percent = (n_uncertain / len(df) * 100)
        mean_entropy = entropies.mean()
        
        logger.info(f"Predicciones inciertas: {n_uncertain}/{len(df)} ({uncertain_percent:.1f}%)")
        logger.info(f"Entropía media: {mean_entropy:.3f}")
        
        # Calcular entropía mediana por sitio si existe la columna Site
        if 'Site' in results_df.columns:
            entropy_median_by_site = results_df.groupby('Site')['Entropy'].median()
            for site, median in entropy_median_by_site.items():
                logger.info(f"Entropía mediana para {site}: {median:.3f}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error en análisis de incertidumbre: {str(e)}")
        return None

def save_uncertainty_results(results_df, output_path=None):
    """
    Guarda los resultados del análisis en archivos Excel y CSV.

    Args:
        results_df (pd.DataFrame): DataFrame con resultados del análisis.
        output_path (str, optional): Ruta base para los archivos de resultados.
            Si es None, se genera automáticamente.

    Returns:
        dict: Diccionario con rutas a los archivos generados o None si falló.
    """
    if results_df is None:
        logger.error("No hay resultados para guardar.")
        return None
    
    try:
        # Generar ruta base si no se proporciona
        if output_path is None:
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_path = f"uncertainty_analysis_{current_date}"
        else:
            output_path = os.path.splitext(output_path)[0]
        
        # Rutas de salida
        excel_path = f"{output_path}.xlsx"
        csv_path = f"{output_path}.csv"
        
        # Guardar resultados
        results_df.to_excel(excel_path, index=False)
        results_df.to_csv(csv_path, index=False)
        
        logger.info(f"Resultados guardados en Excel: {excel_path}")
        logger.info(f"Resultados guardados en CSV: {csv_path}")
        
        return {
            'excel': excel_path,
            'csv': csv_path
        }
        
    except Exception as e:
        logger.error(f"Error guardando resultados: {str(e)}")
        return None

def process_predictions_with_uncertainty(prediction_df=None, prediction_path=None, 
                                        output_path=None, confidence_threshold=0.7):
    """
    Procesa las predicciones y realiza análisis de incertidumbre.

    Args:
        prediction_df (pd.DataFrame, optional): DataFrame con predicciones.
        prediction_path (str, optional): Ruta al archivo con predicciones.
        output_path (str, optional): Ruta para guardar resultados.
        confidence_threshold (float): Umbral de confianza.

    Returns:
        pd.DataFrame: DataFrame con resultados de incertidumbre.
    """
    # Cargar predicciones si se proporciona ruta
    if prediction_df is None and prediction_path:
        try:
            file_ext = os.path.splitext(prediction_path)[1].lower()
            if file_ext == '.csv':
                prediction_df = pd.read_csv(prediction_path)
            elif file_ext in ['.xlsx', '.xls']:
                prediction_df = pd.read_excel(prediction_path)
            else:
                logger.error(f"Formato de archivo no soportado: {file_ext}")
                return None
                
            logger.info(f"Cargadas {len(prediction_df)} predicciones desde {prediction_path}")
        except Exception as e:
            logger.error(f"Error cargando predicciones: {str(e)}")
            return None
    
    # Verificar que tenemos un DataFrame para analizar
    if prediction_df is None:
        logger.error("No se proporcionó un DataFrame ni una ruta válida")
        return None
    
    # Realizar análisis de incertidumbre
    results_df = analyze_uncertainty(prediction_df, confidence_threshold)
    
    # Guardar resultados si se proporciona ruta
    if results_df is not None and output_path:
        save_uncertainty_results(results_df, output_path)
    
    return results_df

# Para pruebas directas del módulo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import sys
    if len(sys.argv) > 1:
        prediction_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        
        results = process_predictions_with_uncertainty(
            prediction_path=prediction_path,
            output_path=output_path,
            confidence_threshold=confidence_threshold
        )
        
        if results is not None:
            print(f"Análisis completado. Registradas {len(results)} filas.")
        else:
            print("Error en el análisis de incertidumbre.")
    else:
        print("Uso: python uncertainty_analysis.py <archivo_predicciones> [archivo_salida] [umbral_confianza]")