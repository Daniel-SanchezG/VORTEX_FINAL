#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de Visualización para Análisis Arqueológico
-------------------------------------------------
Complemento para crear visualizaciones de entropía y probabilidades
basadas en los resultados del análisis de incertidumbre.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# Configuración de logging
logger = logging.getLogger("Visualization")

def plot_site_entropy_distribution(uncertainty_df, output_file=None, 
                                  entropy_col='Entropy', score_cols=None):
    """
    Crea un gráfico que muestra la distribución de probabilidades y entropía por sitio.
    
    Args:
        uncertainty_df (pd.DataFrame): DataFrame con resultados del análisis de incertidumbre.
        output_file (str, optional): Ruta del archivo para guardar el gráfico.
        entropy_col (str): Nombre de la columna de entropía.
        score_cols (list): Lista de columnas con puntuaciones de probabilidad.
    
    Returns:
        pd.DataFrame: DataFrame con estadísticas por sitio.
    """
    logger.info("Generando visualización de distribución de entropía por sitio...")
    
    try:
        # Copiar DataFrame para no modificar el original
        df = uncertainty_df.copy()
        
        # Identificar columnas de puntuación si no se especifican
        if score_cols is None:
            score_cols = [col for col in df.columns if col.startswith('prediction_score_')]
            
            if not score_cols:
                logger.error("No se encontraron columnas de puntuación")
                return None
        
        logger.info(f"Usando columnas de puntuación: {score_cols}")
        
        # Asegurar que las columnas son numéricas
        for col in score_cols:
            if df[col].dtype == object:  # Si es string
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                
        # Asegurar que la columna de entropía existe
        if entropy_col not in df.columns:
            logger.error(f"Columna de entropía '{entropy_col}' no encontrada")
            return None
        
        # Calcular medianas por sitio
        site_medians = df.groupby('Site')[score_cols + [entropy_col]].median()
        
        # Configurar estilo del gráfico
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Crear gráfico de barras apiladas
        bottom = np.zeros(len(site_medians))
        colors = ['#4B0082', '#228B22', '#B8860B']  # Colores para CT, PCM, PDLC
        
        # Usar nombres de sitios como posiciones en x
        x = np.arange(len(site_medians.index))

        class_names = {
        'CT': 'Can Tintorer',
        'PCM': 'Encinasola',
        'PDLC': 'Aliste'
        }
        
        for i, col in enumerate(score_cols):
            class_code = col.replace('prediction_score_', '')
            class_label = class_names.get(class_code, class_code)  # Usar el nombre completo o el código si no está en el mapeo
            
            ax.bar(x, site_medians[col], bottom=bottom, 
                label=class_label,
                color=colors[i], alpha=0.7)
            bottom += site_medians[col]
        
        # Añadir línea de entropía
        ax2 = ax.twinx()
        ax2.plot(x, site_medians[entropy_col], 
                color='red', linewidth=2, label='Entropy', 
                marker='o')
        
        # Configurar ejes y etiquetas
        ax.set_xlabel('Sites', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Distribution (Median)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold', color='red')
        
        # Rotar etiquetas del eje x
        ax.set_xticks(x)
        ax.set_xticklabels(site_medians.index, rotation=45, ha='right')
        
        # Ajustar leyendas
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, 
                 loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Ajustar diseño para evitar que las etiquetas se corten
        plt.subplots_adjust(bottom=0.2)
        
        # Generar ruta para guardar si no se proporciona
        if output_file is None:
            import datetime
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_file = f"site_entropy_distribution_{current_date}.png"
        
        # Guardar gráfico
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualización guardada en: {output_file}")
        
        # Crear y mostrar estadísticas completas por sitio
        stats = pd.DataFrame({
            'median_CT': site_medians['prediction_score_CT'] if 'prediction_score_CT' in site_medians else np.nan,
            'median_PCM': site_medians['prediction_score_PCM'] if 'prediction_score_PCM' in site_medians else np.nan,
            'median_PDLC': site_medians['prediction_score_PDLC'] if 'prediction_score_PDLC' in site_medians else np.nan,
            'median_entropy': site_medians[entropy_col],
            'n_samples': df.groupby('Site').size(),
            'mean_entropy': df.groupby('Site')[entropy_col].mean(),
            'std_entropy': df.groupby('Site')[entropy_col].std()
        }).round(3)
        
        logger.info("Estadísticas por sitio generadas")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error generando visualización: {str(e)}")
        return None

def save_statistics(stats_df, output_path=None):
    """
    Guarda las estadísticas en archivos Excel y CSV.
    
    Args:
        stats_df (pd.DataFrame): DataFrame con estadísticas por sitio.
        output_path (str, optional): Ruta base para los archivos de resultados.
    
    Returns:
        dict: Diccionario con rutas a los archivos generados o None si falló.
    """
    if stats_df is None:
        logger.error("No hay estadísticas para guardar")
        return None
    
    try:
        # Generar ruta base si no se proporciona
        if output_path is None:
            import datetime
            current_date = datetime.datetime.now().strftime("%Y%m%d")
            output_path = f"site_statistics_{current_date}"
        else:
            output_path = os.path.splitext(output_path)[0]
        
        # Rutas de salida
        excel_path = f"{output_path}.xlsx"
        csv_path = f"{output_path}.csv"
        
        # Guardar resultados
        stats_df.to_excel(excel_path)
        stats_df.to_csv(csv_path)
        
        logger.info(f"Estadísticas guardadas en Excel: {excel_path}")
        logger.info(f"Estadísticas guardadas en CSV: {csv_path}")
        
        return {
            'excel': excel_path,
            'csv': csv_path
        }
        
    except Exception as e:
        logger.error(f"Error guardando estadísticas: {str(e)}")
        return None

def generate_visualization(uncertainty_df=None, uncertainty_path=None, 
                          output_dir=None, entropy_col='Entropy'):
    """
    Genera visualización y estadísticas basadas en resultados de incertidumbre.
    
    Args:
        uncertainty_df (pd.DataFrame, optional): DataFrame con resultados de incertidumbre.
        uncertainty_path (str, optional): Ruta al archivo con resultados de incertidumbre.
        output_dir (str, optional): Directorio para guardar resultados.
        entropy_col (str): Nombre de la columna de entropía.
    
    Returns:
        dict: Diccionario con rutas a los archivos generados.
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
    
    results = {}
    
    # Generar rutas de salida
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        import datetime
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        vis_file = os.path.join(output_dir, f"site_entropy_distribution_{current_date}.png")
        stats_file = os.path.join(output_dir, f"site_statistics_{current_date}")
    else:
        vis_file = None
        stats_file = None
    
    # Generar visualización
    stats_df = plot_site_entropy_distribution(
        uncertainty_df, 
        output_file=vis_file,
        entropy_col=entropy_col
    )
    
    if vis_file:
        results['visualization'] = vis_file
    
    # Guardar estadísticas
    if stats_df is not None:
        stats_files = save_statistics(stats_df, output_path=stats_file)
        if stats_files:
            results['statistics'] = stats_files
    
    return results if results else None

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
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        results = generate_visualization(
            uncertainty_path=uncertainty_path,
            output_dir=output_dir
        )
        
        if results:
            print("Visualización y estadísticas generadas correctamente.")
            for key, value in results.items():
                print(f"{key}: {value}")
        else:
            print("Error generando visualización y estadísticas.")
    else:
        print("Uso: python visualization.py <archivo_incertidumbre> [directorio_salida]")