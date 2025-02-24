import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import shap
from shap.plots import beeswarm
import os
from pathlib import Path
import logging

class ShapAnalyzer:
    """
    Class for analyzing feature importance using SHAP values.
    Simplified for binary classification only.
    """
    
    def __init__(self, output_dir, random_state=123):
        """
        Initialize the SHAP analyzer.
        
        Args:
            output_dir: Directory where results will be saved
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.random_state = random_state
        
        # Mapeo de nombres de hojas a nombres de columnas objetivo
        self.target_column_map = {
            'aliste': 'target_PDLC',
            'can_tintorer': 'target_CT',
            'encinasola': 'target_PCM'
        }
        
        # Setup logging if available, otherwise use print
        try:
            self.logger = logging.getLogger(__name__)
        except:
            self.logger = None
        
        # Create output directories
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    
    def log(self, message, level='info'):
        """Helper method to log or print messages"""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
        else:
            print(message)
    
    def process_excel_sheet(self, excel_path, sheet_name, train_existing_model=None):
        """
        Procesa una hoja de Excel para análisis SHAP:
        - Carga datos
        - Preprocesamiento
        - Balanceo con SMOTE
        - Entrenamiento de modelo Random Forest (o uso de modelo existente)
        - Análisis SHAP
        - Guarda gráfico SHAP
        
        Args:
            excel_path (str): Ruta al archivo Excel
            sheet_name (str): Nombre de la hoja a procesar
            train_existing_model: Modelo pre-entrenado (opcional)
            
        Returns:
            Dictionary with analysis results
        """
        # Definir directorios específicos para plots y tablas
        plots_dir = self.output_dir / 'plots'
        tables_dir = self.output_dir / 'tables'
        
        # Asegurarse de que los directorios existen
        plots_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.log(f"Procesando hoja: {sheet_name}")
        
        # 1. Cargar datos
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')
            self.log(f"Datos cargados. Forma: {df.shape}")
        except Exception as e:
            self.log(f"Error al cargar la hoja {sheet_name}: {str(e)}", level='error')
            raise
        
        # 2. Preprocesamiento básico
        if 'Site' in df.columns:
            df = df.drop(['Site'], axis=1)
        if 'suma' in df.columns:
            df = df.drop(['suma'], axis=1)
        
        # Verificar que no hay valores nulos
        if df.isnull().any().any():
            self.log("¡Advertencia! Hay valores nulos en los datos", level='warning')
        
        # 3. Determinar la columna objetivo
        target_column = self.target_column_map.get(sheet_name)
        if target_column is None:
            self.log(f"No se encontró un mapeo para la hoja {sheet_name}, buscando columnas 'target_'", level='warning')
            # Buscar cualquier columna que comience con 'target_'
            target_columns = [col for col in df.columns if col.startswith('target_')]
            if target_columns:
                target_column = target_columns[0]
                self.log(f"Se usará la columna {target_column} como objetivo")
            else:
                raise ValueError(f"No se encontró una columna objetivo para la hoja {sheet_name}")
        
        if target_column not in df.columns:
            self.log(f"Columna objetivo {target_column} no encontrada en la hoja {sheet_name}", level='error')
            # Mostrar las columnas disponibles para ayudar a depurar
            self.log(f"Columnas disponibles: {', '.join(df.columns)}")
            raise ValueError(f"Columna objetivo {target_column} no encontrada")
        
        # 4. Separar características y objetivo
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        
        # Mostrar distribución del objetivo
        target_counts = y.value_counts()
        self.log(f"Distribución del objetivo antes de SMOTE:\n{target_counts}")
        
        # 5. Balanceo con SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        
        self.log(f"Distribución del objetivo después de SMOTE:\n{y_smote.value_counts()}")
        
        # 6. División en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X_smote, y_smote, test_size=0.2, random_state=self.random_state, stratify=y_smote
        )
        
        # 7. Entrenamiento de RandomForestClassifier o uso de modelo existente
        if train_existing_model is None:
            # Si no hay modelo pre-entrenado, entrenamos uno nuevo
            self.log("Entrenando nuevo modelo Random Forest...")
            rf_model = RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced",
                criterion='entropy',
                random_state=self.random_state,
                n_jobs=-1  # Usar todos los núcleos disponibles
            )
            
            rf_model.fit(X_train, y_train)
            
            # Evaluación del modelo
            y_pred = rf_model.predict(X_test)
            report = classification_report(y_test, y_pred)
            self.log(f"Evaluación del modelo:\n{report}")
        else:
            # Si hay un modelo pre-entrenado, lo usamos
            rf_model = train_existing_model
            self.log("Usando modelo pre-entrenado para análisis SHAP")
        
        # 8. Análisis SHAP
        self.log("Calculando valores SHAP...")
        explainer = shap.Explainer(rf_model)
        shap_values = explainer(X_train)
        
        # 9. Generar y guardar plot SHAP
        
        
        # 9. Generar y guardar plot SHAP
        self.log("Generando visualización SHAP...")
        plt.figure(figsize=(12, 8))

        # Verificamos la forma de shap_values para determinar cómo visualizarlos
        if len(shap_values.shape) == 3 and shap_values.shape[2] > 1:
            # Caso de clasificación binaria o multiclase
            beeswarm(shap_values[:,:,1], max_display=20, color=plt.get_cmap("viridis"), show=False)
        else:
            # Caso de regresión o salida única
            beeswarm(shap_values, max_display=20, color=plt.get_cmap("viridis"), show=False)
            
        plt.title(f"SHAP Values - {sheet_name} ({target_column})")

        # Asegurarse de que el gráfico se dibuje completamente
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)  # Pequeña pausa para asegurar que el gráfico se procese

        # Guardar gráfico en la carpeta plots
        output_path = plots_dir / f"SHAP_local_{sheet_name}.png"
        plt.savefig(output_path, bbox_inches='tight')
        self.log(f"Gráfico guardado en: {output_path}")
        plt.close()
        
        # 10. Calcular y guardar importancia de características
        if len(shap_values.shape) == 3 and shap_values.shape[2] > 1:
            # Para clasificación binaria, usar valores de clase 1
            feature_importance = np.abs(shap_values.values[:,:,1]).mean(0)
        else:
            # Para regresión o salida única
            feature_importance = np.abs(shap_values.values).mean(0)
        
        # Crear DataFrame de importancia
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Guardar tabla de importancia en la carpeta tables
        importance_path = tables_dir / f"shap_importance_{sheet_name}.csv"
        importance_df.to_csv(importance_path, index=False)
        
        self.log(f"Tabla de importancia guardada en: {importance_path}")
        
        # 11. Retornar resultados
        result = {
            'model': rf_model,
            'shap_values': shap_values,
            'X_train': X_train,
            'feature_importance': importance_df,
            'plot_path': output_path,
            'importance_path': importance_path
        }
        
        # Añadir elementos adicionales si se entrenó un nuevo modelo
        if train_existing_model is None:
            result.update({
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'classification_report': report
            })
            
        return result
    
    def analyze_multiple_sheets(self, excel_path, sheet_names=None, model=None):
        """
        Analyze SHAP values for multiple sheets in an Excel file.
        
        Args:
            excel_path: Path to Excel file
            sheet_names: List of sheet names to analyze (if None, all sheets)
            model: Pre-trained model to use for all sheets (optional)
            
        Returns:
            Dictionary of results for each sheet
        """
        excel_path = Path(excel_path) if isinstance(excel_path, str) else excel_path
        self.log(f"Analizando valores SHAP para múltiples hojas en {excel_path}")
        
        # Verificar que el archivo existe
        if not excel_path.exists():
            raise FileNotFoundError(f"El archivo {excel_path} no existe")
        
        try:
            # If no sheet names provided, get all sheets
            if sheet_names is None:
                try:
                    xls = pd.ExcelFile(excel_path)
                    sheet_names = xls.sheet_names
                    self.log(f"Hojas encontradas: {', '.join(sheet_names)}")
                except Exception as e:
                    self.log(f"Error al leer las hojas del archivo: {str(e)}", level='error')
                    raise
            
            # Corregir el nombre de "encinsasola" a "encinasola" si es necesario
            corrected_sheet_names = []
            for sheet in sheet_names:
                if sheet == "encinsasola":
                    self.log("Corrigiendo nombre de hoja 'encinsasola' a 'encinasola'")
                    corrected_sheet_names.append("encinasola")
                else:
                    corrected_sheet_names.append(sheet)
            
            # Process each sheet
            results = {}
            successful_sheets = 0
            
            for sheet in corrected_sheet_names:
                try:
                    self.log(f"Procesando hoja: {sheet}")
                    sheet_result = self.process_excel_sheet(
                        excel_path=excel_path,
                        sheet_name=sheet,
                        train_existing_model=model
                    )
                    results[sheet] = sheet_result
                    successful_sheets += 1
                    self.log(f"Completado procesamiento de {sheet}")
                except Exception as e:
                    self.log(f"Error procesando hoja {sheet}: {str(e)}", level='error')
            
            # Create a combined importance table if at least one sheet was processed successfully
            if results:
                self.log("Generando tabla de importancia combinada...")
                combined_importance = pd.DataFrame()
                for sheet, result in results.items():
                    if 'feature_importance' in result:
                        importance = result['feature_importance'].copy()
                        importance.columns = ['feature', f'{sheet}_importance']
                        
                        if combined_importance.empty:
                            combined_importance = importance
                        else:
                            combined_importance = combined_importance.merge(
                                importance, on='feature', how='outer'
                            )
                
                # Calculate average importance across sheets
                importance_columns = [col for col in combined_importance.columns if col.endswith('_importance')]
                if importance_columns:
                    combined_importance['avg_importance'] = combined_importance[importance_columns].mean(axis=1)
                    combined_importance = combined_importance.sort_values('avg_importance', ascending=False)
                
                # Save combined table
                tables_dir = self.output_dir / 'tables'
                combined_table_path = tables_dir / "shap_importance_combined.csv"
                combined_importance.to_csv(combined_table_path, index=False)
                
                self.log(f"Tabla de importancia combinada guardada en: {combined_table_path}")
                results['combined_importance'] = combined_importance
            
            self.log(f"Procesadas correctamente {successful_sheets} de {len(corrected_sheet_names)} hojas")
            return results
            
        except Exception as e:
            self.log(f"Error en análisis SHAP de múltiples hojas: {str(e)}", level='error')
            raise

# Este bloque solo se ejecuta si se llama directamente al script
if __name__ == "__main__":
    # Ruta al archivo Excel
    excel_path = '/home/dsg/vortex/PRODUCTION/DATA/processed/training_data.xlsx'
    
    # Lista de hojas a procesar (corregido el nombre de "encinsasola")
    sheet_names = ['aliste', 'can_tintorer', 'encinasola']
    
    # Crear analizador
    output_dir = Path('/home/dsg/vortex/PRODUCTION/resultados_shap')
    analyzer = ShapAnalyzer(output_dir=output_dir)
    
    # Procesar cada hoja
    results = analyzer.analyze_multiple_sheets(excel_path, sheet_names)
    print(f"Proceso completado con éxito. Resultados guardados en {output_dir}")