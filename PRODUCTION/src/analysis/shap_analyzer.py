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
import sys

class ShapAnalyzer:
    """
    Class for analyzing feature importance using SHAP values.
    Simplified for binary classification only.
    """
    
    def __init__(self, output_dir, random_state=123, img_format='png', dpi=300):
        """
        Initialize the SHAP analyzer.
        
        Args:
            output_dir: Directory where results will be saved
            random_state: Random seed for reproducibility
            img_format: Format for saving images ('png', 'jpg', 'svg', etc.)
            dpi: Resolution for saved images (dots per inch)
        """
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.random_state = random_state
        self.img_format = img_format
        self.dpi = dpi
        
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
        
        # Create output directories once
        self.plots_dir = self.output_dir / 'plots'
        self.tables_dir = self.output_dir / 'tables'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.log(f"Directorio de resultados configurado en: {self.output_dir}")
        self.log(f"Los gráficos se guardarán en: {self.plots_dir}")
        self.log(f"Las tablas se guardarán en: {self.tables_dir}")
    
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
            
            # Mostrar importancia de características nativa del modelo
            if hasattr(rf_model, 'feature_importances_'):
                importance_df_rf = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.log(f"Top 10 características según importancia del modelo:\n{importance_df_rf.head(10)}")
        else:
            # Si hay un modelo pre-entrenado, lo usamos
            rf_model = train_existing_model
            self.log("Usando modelo pre-entrenado para análisis SHAP")
        
        # 8. Análisis SHAP
        self.log("Calculando valores SHAP...")
        try:
            # Usar TreeExplainer que es específico para modelos basados en árboles como Random Forest
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_train)
            
            # Verificar si shap_values es una lista (formato antiguo de shap) o un objeto Explanation
            if isinstance(shap_values, list):
                self.log(f"Valores SHAP en formato lista con {len(shap_values)} elementos")
                # Para clasificación binaria, shap_values es una lista con 2 elementos (uno para cada clase)
                if len(shap_values) == 2:
                    # Usamos shap_values[1] para la clase positiva (1)
                    shap_data = shap_values[1]
                    self.log(f"Usando valores SHAP para clase positiva, forma: {shap_data.shape}")
                else:
                    # Si solo hay un elemento, usamos ese
                    shap_data = shap_values[0]
                    self.log(f"Usando valores SHAP para clase única, forma: {shap_data.shape}")
            else:
                # Formato más reciente (objeto Explanation)
                self.log(f"Valores SHAP en formato Explanation, forma: {shap_values.shape}")
                # Para objetos tipo Explanation, extracto de los valores
                shap_data = shap_values
        except Exception as e:
            self.log(f"Error al calcular valores SHAP: {str(e)}", level='error')
            raise
        
        # 9. Generar y guardar plot SHAP
        self.log("Generando visualización SHAP...")
        plt.figure(figsize=(14, 10))
        
        try:
            # Intentar generar el gráfico beeswarm
            if isinstance(shap_values, list):
                # Para formato antiguo de SHAP (lista de arrays)
                shap.summary_plot(
                    shap_values[1] if len(shap_values) > 1 else shap_values[0],
                    X_train,
                    max_display=20,
                    plot_type=None,
                    show=False,
                    color=plt.get_cmap("viridis")
                )
            else:
                # Para formato más reciente (objeto Explanation)
                beeswarm(shap_data, max_display=20, color=plt.get_cmap("viridis"), show=False)
            
                plt.title(f"SHAP Values - {sheet_name} ({target_column})")
                plt.tight_layout()
            
            # Forzar actualización para mostrar datos
            plt.draw()
            plt.pause(0.5)  # Pausa más larga para asegurar renderizado
            
            # Guardar gráfico con alta calidad
            output_path = self.plots_dir / f"SHAP_local_{sheet_name}.{self.img_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            self.log(f"Gráfico guardado en: {output_path}")
        except Exception as e:
            self.log(f"Error al generar visualización SHAP: {str(e)}", level='error')
            
            # Intentar un enfoque alternativo
            plt.cla()  # Limpiar eje actual
            plt.clf()  # Limpiar figura
            
            # Crear un gráfico de barras simple con importancias
            try:
                if isinstance(shap_values, list):
                    # Formato antiguo (lista)
                    feature_imp = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0]).mean(0)
                else:
                    # Formato nuevo (Explanation)
                    if hasattr(shap_values, 'values'):
                        if len(shap_values.shape) == 3:
                            feature_imp = np.abs(shap_values.values[:,:,1]).mean(0)
                        else:
                            feature_imp = np.abs(shap_values.values).mean(0)
                    else:
                        feature_imp = np.abs(shap_values).mean(0)
                
                # Ordenar importancias
                indices = np.argsort(feature_imp)[-20:]  # Top 20
                features = X_train.columns[indices]
                importances = feature_imp[indices]
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(features)), importances)
                plt.yticks(range(len(features)), features)
                plt.title(f"SHAP Values - {sheet_name} ({target_column})")
                plt.xlabel('Mean |SHAP value|')
                plt.tight_layout()
                
                # Guardar gráfico alternativo
                output_path = self.plots_dir / f"SHAP_alt_{sheet_name}.{self.img_format}"
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                self.log(f"Gráfico alternativo guardado en: {output_path}")
            except Exception as e2:
                self.log(f"Error al generar visualización alternativa: {str(e2)}", level='error')
                plt.text(0.5, 0.5, f"Error en visualización SHAP:\n{str(e)}\n\nError en alternativa:\n{str(e2)}", 
                        ha='center', va='center', fontsize=12, wrap=True)
                output_path = self.plots_dir / f"SHAP_error_{sheet_name}.{self.img_format}"
                plt.savefig(output_path, dpi=self.dpi)
                self.log(f"Imagen de error guardada en: {output_path}")
        
        # Cerrar todas las figuras para liberar memoria
        plt.close('all')
        
        # 10. Calcular y guardar importancia de características
        try:
            if isinstance(shap_values, list):
                # Para formato antiguo (lista)
                feature_importance = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0]).mean(0)
            else:
                # Para formato nuevo (Explanation)
                if hasattr(shap_values, 'values'):
                    if len(shap_values.shape) == 3 and shap_values.shape[2] > 1:
                        feature_importance = np.abs(shap_values.values[:,:,1]).mean(0)
                    else:
                        feature_importance = np.abs(shap_values.values).mean(0)
                else:
                    feature_importance = np.abs(shap_values).mean(0)
            
            # Crear DataFrame de importancia
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Guardar tabla de importancia en la carpeta tables
            importance_path = self.tables_dir / f"shap_importance_{sheet_name}.csv"
            importance_df.to_csv(importance_path, index=False)
            
            self.log(f"Tabla de importancia guardada en: {importance_path}")
            self.log(f"Top 14 características según SHAP:\n{importance_df.head(14)}")
        except Exception as e:
            self.log(f"Error al calcular importancia de características: {str(e)}", level='error')
            importance_df = pd.DataFrame()
            importance_path = None
        
        # 11. Retornar resultados
        result = {
            'model': rf_model,
            'shap_values': shap_values,
            'X_train': X_train,
            'feature_importance': importance_df,
            'plot_path': output_path if 'output_path' in locals() else None,
            'importance_path': importance_path if 'importance_path' in locals() else None
        }
        
        # Añadir elementos adicionales si se entrenó un nuevo modelo
        if train_existing_model is None:
            result.update({
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'classification_report': report if 'report' in locals() else None
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
            self.log(f"Error: El archivo {excel_path} no existe", level='error')
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
            failed_sheets = 0
            
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
                    failed_sheets += 1
                    self.log(f"Error procesando hoja {sheet}: {str(e)}", level='error')
                    # Continuamos con la siguiente hoja
            
            # Create a combined importance table if at least one sheet was processed successfully
            if results:
                self.log("Generando tabla de importancia combinada...")
                try:
                    combined_importance = pd.DataFrame()
                    for sheet, result in results.items():
                        if 'feature_importance' in result and not result['feature_importance'].empty:
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
                    combined_table_path = self.tables_dir / "shap_importance_combined.csv"
                    combined_importance.to_csv(combined_table_path, index=False)
                    
                    self.log(f"Tabla de importancia combinada guardada en: {combined_table_path}")
                    self.log(f"Top 10 características promedio:\n{combined_importance.head(10)}")
                    results['combined_importance'] = combined_importance
                except Exception as e:
                    self.log(f"Error al generar tabla combinada: {str(e)}", level='error')
            
            self.log(f"Resumen del procesamiento:")
            self.log(f"- Hojas procesadas correctamente: {successful_sheets}")
            self.log(f"- Hojas con errores: {failed_sheets}")
            self.log(f"- Total de hojas: {len(corrected_sheet_names)}")
            
            if successful_sheets == 0:
                self.log("Advertencia: No se procesó correctamente ninguna hoja", level='warning')
            
            return results
            
        except Exception as e:
            self.log(f"Error en análisis SHAP de múltiples hojas: {str(e)}", level='error')
            raise

def setup_logging(output_dir=None, level=logging.INFO):
    """
    Configura el sistema de logging.
    
    Args:
        output_dir: Directorio para guardar el archivo de log
        level: Nivel de logging
    
    Returns:
        Logger configurado
    """
    # Crear formateador
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configurar logger raíz
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (si se proporciona output_dir)
    if output_dir:
        output_dir = Path(output_dir)
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'shap_analysis_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        print(f"Log file: {log_file}")
    
    return logger

# Este bloque solo se ejecuta si se llama directamente al script
if __name__ == "__main__":
    try:
        # Obtener directorio raíz del proyecto (2 niveles arriba del script actual)
        current_script_dir = Path(__file__).resolve().parent
        project_root = current_script_dir.parent.parent  # Subir dos niveles desde src/analysis
        
        # Configuración básica de argumentos
        import argparse
        parser = argparse.ArgumentParser(description='Análisis SHAP para modelos de clasificación.')
        parser.add_argument('--input', type=str, 
                          default=str(project_root / 'DATA/processed/training_data.xlsx'),
                          help='Ruta al archivo Excel de entrada (por defecto: DATA/processed/training_data.xlsx)')
        parser.add_argument('--output-dir', type=str, 
                          default=str(project_root / 'resultados_shap'),
                          help='Directorio para guardar resultados (por defecto: resultados_shap en la raíz)')
        parser.add_argument('--sheets', nargs='+', 
                          default=['aliste', 'can_tintorer', 'encinasola'],
                          help='Hojas a procesar (por defecto: aliste, can_tintorer, encinasola)')
        parser.add_argument('--format', type=str, default='png',
                          help='Formato de imágenes (png, jpg, svg) (por defecto: png)')
        parser.add_argument('--dpi', type=int, default=300,
                          help='Resolución de imágenes (por defecto: 300)')
        
        args = parser.parse_args()
        
        # Configurar logging
        logger = setup_logging(args.output_dir)
        
        # Mostrar parámetros
        logger.info(f"Ejecutando análisis SHAP con los siguientes parámetros:")
        logger.info(f"- Archivo Excel: {args.input}")
        logger.info(f"- Directorio de salida: {args.output_dir}")
        logger.info(f"- Hojas a procesar: {args.sheets}")
        logger.info(f"- Formato de imágenes: {args.format}")
        logger.info(f"- Resolución (DPI): {args.dpi}")
        
        # Crear analizador
        output_dir = Path(args.output_dir)
        analyzer = ShapAnalyzer(
            output_dir=output_dir,
            img_format=args.format,
            dpi=args.dpi
        )
        
        # Procesar cada hoja
        results = analyzer.analyze_multiple_sheets(
            excel_path=args.input,
            sheet_names=args.sheets
        )
        
        logger.info(f"Proceso completado con éxito. Resultados guardados en {output_dir}")
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Error durante la ejecución: {str(e)}", exc_info=True)
        else:
            print(f"Error durante la ejecución: {str(e)}")
        sys.exit(1)