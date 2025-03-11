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
        
        self.log(f"Results directory set up in: {self.output_dir}")
        self.log(f"Plots will be saved in: {self.plots_dir}")
        self.log(f"Tables will be saved in: {self.tables_dir}")
    
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
        Process an Excel sheet for SHAP analysis:
        - Load data
        - Preprocessing
        - Balance with SMOTE
        - Training of Random Forest model (or use existing model)
        - SHAP analysis
        - Save SHAP plot
        
        Args:
            excel_path (str): Path to Excel file
            sheet_name (str): Name of the sheet to process
            train_existing_model: Pre-trained model (optional)
            
        Returns:
            Dictionary with analysis results
        """
        self.log(f"Processing sheet: {sheet_name}")
        
        # 1. Cargar datos
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')
            self.log(f"Data loaded. Shape: {df.shape}")
        except Exception as e:
            self.log(f"Error loading sheet {sheet_name}: {str(e)}", level='error')
            raise
        
        # 2. Basic preprocessing
        if 'Site' in df.columns:
            df = df.drop(['Site'], axis=1)
        if 'suma' in df.columns:
            df = df.drop(['suma'], axis=1)
        
        # Verificar que no hay valores nulos
        if df.isnull().any().any():
            self.log("Warning! There are null values in the data", level='warning')
        
        # 3. Determinar la columna objetivo
        target_column = self.target_column_map.get(sheet_name)
        if target_column is None:
            self.log(f"No mapping found for sheet {sheet_name}, searching for columns 'target_'", level='warning')
            # Buscar cualquier columna que comience con 'target_'
            target_columns = [col for col in df.columns if col.startswith('target_')]
            if target_columns:
                target_column = target_columns[0]
                self.log(f"Using column {target_column} as target")
            else:
                raise ValueError(f"No target column found for sheet {sheet_name}")
        
        if target_column not in df.columns:
            self.log(f"Target column {target_column} not found in sheet {sheet_name}", level='error')
            # Mostrar las columnas disponibles para ayudar a depurar
            self.log(f"Available columns: {', '.join(df.columns)}")
            raise ValueError(f"Target column {target_column} not found")
        
        # 4. Separar características y objetivo
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        
        # Show target distribution
        target_counts = y.value_counts()
        self.log(f"Target distribution before SMOTE:\n{target_counts}")
        
        # 5. Balance with SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        
        self.log(f"Target distribution after SMOTE:\n{y_smote.value_counts()}")
        
        # 6. Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_smote, y_smote, test_size=0.2, random_state=self.random_state, stratify=y_smote
        )
        
        # 7. Training of RandomForestClassifier or use existing model
        if train_existing_model is None:
            # If no pre-trained model, train a new one
            self.log("Training new Random Forest model...")
            rf_model = RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced",
                criterion='entropy',
                random_state=self.random_state,
                n_jobs=-1  # Use all available cores
            )
            
            rf_model.fit(X_train, y_train)
            
            # Evaluación del modelo
            y_pred = rf_model.predict(X_test)
            report = classification_report(y_test, y_pred)
            self.log(f"Model evaluation:\n{report}")
            
            # Show feature importance of native model
            if hasattr(rf_model, 'feature_importances_'):
                importance_df_rf = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.log(f"Top 14 features according to model importance:\n{importance_df_rf.head(14)}")
        else:
            # If there is a pre-trained model, use it
            rf_model = train_existing_model
            self.log("Using pre-trained model for SHAP analysis")
        
        # 8. SHAP analysis
        self.log("Calculating SHAP values...")
        try:
            # Use TreeExplainer that is specific to tree-based models like Random Forest
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_train)
            
            # Check if shap_values is a list (old shap format) or an Explanation object
            if isinstance(shap_values, list):
                self.log(f"SHAP values in list format with {len(shap_values)} elements")
                # For binary classification, shap_values is a list with 2 elements (one for each class)
                if len(shap_values) == 2:
                    # We use shap_values[1] for the positive class (1)
                    shap_data = shap_values[1]
                    self.log(f"Using SHAP values for positive class, shape: {shap_data.shape}")
                else:
                    # If there is only one element, we use that
                    shap_data = shap_values[0]
                    self.log(f"Using SHAP values for single class, shape: {shap_data.shape}")
            else:
                # New format (Explanation object)
                self.log(f"SHAP values in Explanation format, shape: {shap_values.shape}")
                # For Explanation objects, extract the values
                shap_data = shap_values
        except Exception as e:
            self.log(f"Error calculating SHAP values: {str(e)}", level='error')
            raise
        
        # 9. Generate and save SHAP plot
        self.log("Generating SHAP plot...")
        plt.figure(figsize=(14, 10))
        
        try:
            # Intentar generar el gráfico beeswarm
            if isinstance(shap_values, list):
                # For old SHAP format (list of arrays)
                shap.summary_plot(
                    shap_values[1] if len(shap_values) > 1 else shap_values[0],
                    X_train,
                    max_display=20,
                    cmap=plt.cm.viridis,
                    plot_type=None,
                    show=False
                )
            else:
                
                # For new SHAP format (Explanation object)
                beeswarm(shap_data, max_display=20, color=plt.get_cmap("viridis"), show=False)
            
            plt.title(f"SHAP Values - {sheet_name} ({target_column})")
            plt.tight_layout()
            
            # Force update to show data
            plt.draw()
            plt.pause(0.5)  # Pause longer to ensure rendering
            
            # Save plot with high quality
            output_path = self.plots_dir / f"SHAP_local_{sheet_name}.{self.img_format}"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            self.log(f"Plot saved in: {output_path}")
        except Exception as e:
            self.log(f"Error generating SHAP plot: {str(e)}", level='error')
            
            # Try an alternative approach
            plt.cla()  # Clear current axis
            plt.clf()  # Clear figure
            
            # Create a simple bar plot with importances
            try:
                if isinstance(shap_values, list):
                    # Old format (list)
                    feature_imp = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0]).mean(0)
                else:
                    # New format (Explanation)
                    if hasattr(shap_values, 'values'):
                        if len(shap_values.shape) == 3:
                            feature_imp = np.abs(shap_values.values[:,:,1]).mean(0)
                        else:
                            feature_imp = np.abs(shap_values.values).mean(0)
                    else:
                        feature_imp = np.abs(shap_values).mean(0)
                
                # Sort importances
                indices = np.argsort(feature_imp)[-20:]  # Top 20
                features = X_train.columns[indices]
                importances = feature_imp[indices]
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(features)), importances)
                plt.yticks(range(len(features)), features)
                plt.title(f"SHAP Values - {sheet_name} ({target_column})")
                plt.xlabel('Mean |SHAP value|')
                plt.tight_layout()
                
                # Save alternative plot
                output_path = self.plots_dir / f"SHAP_alt_{sheet_name}.{self.img_format}"
                plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                self.log(f"Alternative plot saved in: {output_path}")
            except Exception as e2:
                self.log(f"Error generating alternative plot: {str(e2)}", level='error')
                plt.text(0.5, 0.5, f"Error in SHAP visualization:\n{str(e)}\n\nError in alternative plot:\n{str(e2)}", 
                        ha='center', va='center', fontsize=12, wrap=True)
                output_path = self.plots_dir / f"SHAP_error_{sheet_name}.{self.img_format}"
                plt.savefig(output_path, dpi=self.dpi)
                self.log(f"Error plot saved in: {output_path}")
        
        # Close all figures to free memory
        plt.close('all')
        
        # 10. Calculate and save feature importance
        try:
            if isinstance(shap_values, list):
                # For old format (list)
                feature_importance = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0]).mean(0)
            else:
                # For new format (Explanation)
                if hasattr(shap_values, 'values'):
                    if len(shap_values.shape) == 3 and shap_values.shape[2] > 1:
                        feature_importance = np.abs(shap_values.values[:,:,1]).mean(0)
                    else:
                        feature_importance = np.abs(shap_values.values).mean(0)
                else:
                    feature_importance = np.abs(shap_values).mean(0)
            
            # Create DataFrame of importance
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Save importance table in the tables folder
            importance_path = self.tables_dir / f"shap_importance_{sheet_name}.csv"
            importance_df.to_csv(importance_path, index=False)
            
            self.log(f"Importance table saved in: {importance_path}")
            self.log(f"Top 14 features according to SHAP:\n{importance_df.head(14)}")
        except Exception as e:
            self.log(f"Error calculating feature importance: {str(e)}", level='error')
            importance_df = pd.DataFrame()
            importance_path = None
        
        # 11. Return results
        result = {
            'model': rf_model,
            'shap_values': shap_values,
            'X_train': X_train,
            'feature_importance': importance_df,
            'plot_path': output_path if 'output_path' in locals() else None,
            'importance_path': importance_path if 'importance_path' in locals() else None
        }
        
        # Add additional elements if a new model was trained
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
        
        # Check if the file exists
        if not excel_path.exists():
            self.log(f"Error: The file {excel_path} does not exist", level='error')
            raise FileNotFoundError(f"The file {excel_path} does not exist")
        
        try:
            # If no sheet names provided, get all sheets
            if sheet_names is None:
                try:
                    xls = pd.ExcelFile(excel_path)
                    sheet_names = xls.sheet_names
                    self.log(f"Found sheets: {', '.join(sheet_names)}")
                except Exception as e:
                    self.log(f"Error reading sheets from file: {str(e)}", level='error')
                    raise
            
            # Corregir el nombre de "encinsasola" a "encinasola" si es necesario
            corrected_sheet_names = []
            for sheet in sheet_names:
                if sheet == "encinsasola":
                    self.log("Correcting sheet name 'encinsasola' to 'encinasola'")
                    corrected_sheet_names.append("encinasola")
                else:
                    corrected_sheet_names.append(sheet)
            
            # Process each sheet
            results = {}
            successful_sheets = 0
            failed_sheets = 0
            
            for sheet in corrected_sheet_names:
                try:
                    self.log(f"Processing sheet: {sheet}")
                    sheet_result = self.process_excel_sheet(
                        excel_path=excel_path,
                        sheet_name=sheet,
                        train_existing_model=model
                    )
                    results[sheet] = sheet_result
                    successful_sheets += 1
                    self.log(f"Completed processing of {sheet}")
                except Exception as e:
                    failed_sheets += 1
                    self.log(f"Error processing sheet {sheet}: {str(e)}", level='error')
                    # Continue with the next sheet
            
            # Create a combined importance table if at least one sheet was processed successfully
            if results:
                self.log("Generating combined importance table...")
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
                    
                    self.log(f"Combined importance table saved in: {combined_table_path}")
                    self.log(f"Top 10 average features:\n{combined_importance.head(10)}")
                    results['combined_importance'] = combined_importance
                except Exception as e:
                    self.log(f"Error generating combined table: {str(e)}", level='error')
            
            self.log(f"Summary of processing:")
            self.log(f"- Correctly processed sheets: {successful_sheets}")
            self.log(f"- Sheets with errors: {failed_sheets}")
            self.log(f"- Total sheets: {len(corrected_sheet_names)}")
            
            if successful_sheets == 0:
                self.log("Warning: No sheets were processed correctly", level='warning')
            
            return results
            
        except Exception as e:
            self.log(f"Error in SHAP analysis of multiple sheets: {str(e)}", level='error')
            raise

def setup_logging(output_dir=None, level=logging.INFO):
    """
    Configures the logging system.
    
    Args:
        output_dir: Directory to save the log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Clean existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if output_dir is provided)
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

# This block only runs if the script is called directly
if __name__ == "__main__":
    try:
        # Get the project root directory (2 levels above the current script)
        current_script_dir = Path(__file__).resolve().parent
        project_root = current_script_dir.parent.parent  # Go up two levels from src/analysis
        
        # Basic argument configuration
        import argparse
        parser = argparse.ArgumentParser(description='SHAP analysis for classification models.')
        parser.add_argument('--input', type=str, 
                          default=str(project_root / 'DATA/processed/training_data.xlsx'),
                          help='Path to the input Excel file (default: DATA/processed/training_data.xlsx)')
        parser.add_argument('--output-dir', type=str, 
                          default=str(project_root / 'shap_results'),
                          help='Directory to save results (default: shap_results in the root)')
        parser.add_argument('--sheets', nargs='+', 
                          default=['aliste', 'can_tintorer', 'encinasola'],
                          help='Sheets to process (default: aliste, can_tintorer, encinasola)')
        parser.add_argument('--format', type=str, default='png',
                          help='Image format (png, jpg, svg) (default: png)')
        parser.add_argument('--dpi', type=int, default=300,
                          help='Image resolution (default: 300)')
        
        args = parser.parse_args()
        
        # Configure logging
        logger = setup_logging(args.output_dir)
        
        # Show parameters
        logger.info(f"Running SHAP analysis with the following parameters:")
        logger.info(f"- Excel file: {args.input}")
        logger.info(f"- Output directory: {args.output_dir}")
        logger.info(f"- Sheets to process: {args.sheets}")
        logger.info(f"- Image format: {args.format}")
        logger.info(f"- Image resolution (DPI): {args.dpi}")
        
        # Create analyzer
        output_dir = Path(args.output_dir)
        analyzer = ShapAnalyzer(
            output_dir=output_dir,
            img_format=args.format,
            dpi=args.dpi
        )
        
        # Process each sheet
        results = analyzer.analyze_multiple_sheets(
            excel_path=args.input,
            sheet_names=args.sheets
        )
        
        logger.info(f"Process completed successfully. Results saved in {output_dir}")
        
    except Exception as e:
        if 'logger' in locals():
                logger.error(f"Error during execution: {str(e)}", exc_info=True)
        else:
            print(f"Error during execution: {str(e)}")
        sys.exit(1)