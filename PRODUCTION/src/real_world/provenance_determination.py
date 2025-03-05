#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de Determinación de Procedencia
--------------------------------------
Complemento para el sistema de predicción arqueológica que analiza
los resultados de incertidumbre y determina la procedencia por consenso.
"""

import pandas as pd
import numpy as np
import logging
import os
import datetime

# Configuración de logging
logger = logging.getLogger("Provenance")

def determine_provenance(uncertainty_df, confidence_threshold=0.7):
    """
    Realiza un análisis de determinación de procedencia basado en consenso.
    
    Args:
        uncertainty_df (pd.DataFrame): DataFrame con resultados del análisis de incertidumbre.
        confidence_threshold (float): Umbral de confianza para considerar predicciones fiables.
    
    Returns:
        pd.DataFrame: DataFrame con resultados de procedencia por sitio.
    """
    logger.info("Iniciando análisis de determinación de procedencia...")
    
    try:
        # Copiar DataFrame para no modificar el original
        df = uncertainty_df.copy()
        
        # Verificar que el DataFrame tiene las columnas necesarias
        required_cols = ['Site', 'Original_predictions', 'Entropy']
        score_cols = [col for col in df.columns if col.startswith('prediction_score_')]
        
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Columna requerida {col} no encontrada en el DataFrame")
                return None
        
        if not score_cols:
            logger.error("No se encontraron columnas de puntuación en el DataFrame")
            return None
        
        # Calcular probabilidad máxima para cada muestra
        df['max_prob'] = df[score_cols].max(axis=1)
        
        # Verificar columna de predicciones con umbral
        if 'Uncertainty_threshold_predictions' not in df.columns:
            logger.warning("No se encontró columna de predicciones con umbral, se generará")
            uncertain_mask = df['max_prob'] < confidence_threshold
            df['Uncertainty_threshold_predictions'] = df['Original_predictions'].copy()
            df.loc[uncertain_mask, 'Uncertainty_threshold_predictions'] = 'uncertain'
        
        # Lista para almacenar resultados por sitio
        results = []
        
        # Analizar cada sitio por separado
        for site in df['Site'].unique():
            site_data = df[df['Site'] == site]
            avg_uncertain = round(sum(site_data['max_prob'] < confidence_threshold) / len(site_data) * 100, 2)
            high_conf = site_data[site_data['max_prob'] > confidence_threshold]
            median_entropy = site_data['Entropy'].median() if 'Entropy' in site_data.columns else np.nan
            
            # Determinar consenso basado en muestras de alta confianza
            if len(high_conf) > 0:
                # Usar el valor más frecuente como consenso
                consensus_counts = high_conf['Uncertainty_threshold_predictions'].value_counts()
                consensus = consensus_counts.index[0]
                consistency = consensus_counts.iloc[0] / len(high_conf)
                n_consensus_pred = len(high_conf)
            else:
                consensus = 'No consensus'
                consistency = 0
                n_consensus_pred = 0
            
            # Contar predicciones por clase
            ct_count = len(site_data[site_data['Original_predictions'] == 'CT'])
            pcm_count = len(site_data[site_data['Original_predictions'] == 'PCM'])
            pdlc_count = len(site_data[site_data['Original_predictions'] == 'PDLC'])
            
            # Añadir resultados para este sitio
            results.append({
                'Site': site,
                'Samples_analyzed': len(site_data),
                'Gavá': ct_count,
                'Encinasola': pcm_count,
                'Aliste': pdlc_count,
                'Uncertain(%)': round(avg_uncertain,2),
                'Samples_for_provenance': n_consensus_pred,
                'Median_entropy': round(median_entropy, 2) if not np.isnan(median_entropy) else np.nan,
                'Consensus': consensus,
                'Homogeneity': round(consistency, 2)
            })
        
        # Crear DataFrame con resultados
        result_df = pd.DataFrame(results)
        logger.info(f"Análisis de procedencia completado para {len(result_df)} sitios")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error en análisis de procedencia: {str(e)}")
        return None

def save_provenance_results(provenance_df, output_path=None):
    """
    Guarda los resultados del análisis de procedencia en archivos Excel y CSV.
    
    Args:
        provenance_df (pd.DataFrame): DataFrame con resultados de procedencia.
        output_path (str, optional): Ruta base para los archivos de resultados.
            Si es None, se genera automáticamente.
    
    Returns:
        dict: Diccionario con rutas a los archivos generados o None si falló.
    """
    if provenance_df is None:
        logger.error("No hay resultados de procedencia para guardar")
        return None
    
    try:
        # Generar ruta base si no se proporciona
        if output_path is None:
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_path = f"provenance_analysis_{current_date}"
        else:
            output_path = os.path.splitext(output_path)[0]
        
        # Rutas de salida
        excel_path = f"{output_path}.xlsx"
        csv_path = f"{output_path}.csv"
        
        # Guardar resultados
        provenance_df.to_excel(excel_path, index=False)
        provenance_df.to_csv(csv_path, index=False)
        
        logger.info(f"Resultados de procedencia guardados en Excel: {excel_path}")
        logger.info(f"Resultados de procedencia guardados en CSV: {csv_path}")
        
        return {
            'excel': excel_path,
            'csv': csv_path
        }
    
    except Exception as e:
        logger.error(f"Error guardando resultados de procedencia: {str(e)}")
        return None

def process_provenance_determination(uncertainty_df=None, uncertainty_path=None, 
                                    output_path=None, confidence_threshold=0.7):
    """
    Procesa los resultados de incertidumbre y realiza análisis de procedencia.
    
    Args:
        uncertainty_df (pd.DataFrame, optional): DataFrame con resultados de incertidumbre.
        uncertainty_path (str, optional): Ruta al archivo con resultados de incertidumbre.
        output_path (str, optional): Ruta para guardar resultados.
        confidence_threshold (float): Umbral de confianza.
    
    Returns:
        pd.DataFrame: DataFrame con resultados de procedencia.
    """
    # Cargar resultados de incertidumbre si se proporciona ruta
    if uncertainty_df is None and uncertainty_path:
        try:
            file_ext = os.path.splitext(uncertainty_path)[1].lower()
            if file_ext == '.csv':
                uncertainty_df = pd.read_csv(uncertainty_path)
            elif file_ext in ['.xlsx', '.xls']:
                uncertainty_df = pd.read_excel(uncertainty_path)
            else:
                logger.error(f"Formato de archivo no soportado: {file_ext}")
                return None
            
            logger.info(f"Cargados resultados de incertidumbre desde {uncertainty_path}")
        except Exception as e:
            logger.error(f"Error cargando resultados de incertidumbre: {str(e)}")
            return None
    
    # Verificar que tenemos un DataFrame para analizar
    if uncertainty_df is None:
        logger.error("No se proporcionó un DataFrame ni una ruta válida")
        return None
    
    # Realizar análisis de procedencia
    provenance_df = determine_provenance(uncertainty_df, confidence_threshold)
    
    # Guardar resultados si se proporciona ruta
    if provenance_df is not None and output_path:
        save_provenance_results(provenance_df, output_path)
    
    return provenance_df

# Para pruebas directas del módulo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import sys
    if len(sys.argv) > 1:
        uncertainty_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        
        results = process_provenance_determination(
            uncertainty_path=uncertainty_path,
            output_path=output_path,
            confidence_threshold=confidence_threshold
        )
        
        if results is not None:
            print(f"Análisis de procedencia completado. Analizados {len(results)} sitios.")
        else:
            print("Error en el análisis de procedencia.")
    else:
        print("Uso: python provenance_determination.py <archivo_incertidumbre> [archivo_salida] [umbral_confianza]")