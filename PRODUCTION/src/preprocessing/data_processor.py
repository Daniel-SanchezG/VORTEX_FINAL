# src/preprocessing/data_processor.py

from typing import Dict, List, Optional, Tuple
import pandas as pd # type: ignore
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Clase para el preprocesamiento de datos XRF, incluyendo limpieza,
    eliminación de casos y balanceo de clases.
    """
    
    def __init__(
        self,
        random_state: int = 786,
        min_class_size: int = 10,
        validation_split: float = 0.1
    ):
        """
        Inicializa el preprocesador de datos.
        
        Args:
            random_state: Semilla para reproducibilidad
            min_class_size: Tamaño mínimo de clase para mantener
            validation_split: Fracción de datos para validación final
        """
        self.random_state = random_state
        self.min_class_size = min_class_size
        self.validation_split = validation_split
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos desde archivo Excel o CSV.
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            DataFrame con datos cargados
        """
        path = Path(file_path)
        try:
            if path.suffix == '.xlsx':
                df = pd.read_excel(file_path, engine='openpyxl')
            elif path.suffix == '.csv':
                df = pd.read_csv(file_path, encoding='latin-1')
            else:
                raise ValueError(f"Formato de archivo no soportado: {path.suffix}")
                
            logger.info(f"Datos cargados exitosamente: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
            
    def clean_initial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza limpieza inicial de datos.
        
        Args:
            df: DataFrame con datos crudos
            
        Returns:
            DataFrame limpio
        """
        # Eliminar columnas iniciales innecesarias (primeras 22 columnas)
        data = df.drop(df.iloc[:, :22], axis=1)
        
        # Agregar columnas importantes
        data['Site'] = df['Site']
        data['id'] = df['ID']
        
        # Verificar valores faltantes
        if data.isnull().any().any():
            logger.warning("Se encontraron valores faltantes en los datos")
            
        return data
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina registros duplicados basados en ID.
        
        Args:
            df: DataFrame con posibles duplicados
            
        Returns:
            DataFrame sin duplicados
        """
        # Identificar duplicados
        duplicados = df['id'].duplicated().sum()
        if duplicados > 0:
            logger.info(f"Encontrados {duplicados} IDs duplicados")
            
        # Eliminar duplicados
        df_clean = df.drop_duplicates(subset='id', keep='first')
        df_clean.reset_index(drop=True, inplace=True)
        
        logger.info(f"Registros únicos después de eliminar duplicados: {df_clean.shape[0]}")
        return df_clean
        
    def remove_small_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina clases con menos casos que min_class_size.
        
        Args:
            df: DataFrame con todas las clases
            
        Returns:
            DataFrame sin clases pequeñas
        """
        # Identificar clases pequeñas
        class_counts = df['Site'].value_counts()
        small_classes = class_counts[class_counts < self.min_class_size].index
        
        if len(small_classes) > 0:
            logger.info(f"Eliminando {len(small_classes)} clases con menos de {self.min_class_size} casos")
            df_filtered = df[~df['Site'].isin(small_classes)]
            df_filtered.reset_index(drop=True, inplace=True)
            return df_filtered
        
        return df
        
    def split_validation(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en conjuntos de entrenamiento y validación final.
        
        Args:
            df: DataFrame completo
            
        Returns:
            Tuple de (datos_entrenamiento, datos_validacion)
        """
        # Dividir datos
        train_data = df.sample(
            frac=1-self.validation_split,
            random_state=self.random_state
        )
        val_data = df.drop(train_data.index)
        
        # Resetear índices
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        
        logger.info(f"División de datos - Entrenamiento: {train_data.shape}, Validación: {val_data.shape}")
        return train_data, val_data
        
    def apply_smote(
        self,
        df: pd.DataFrame,
        target_col: str = 'Site',
        exclude_cols: List[str] = ['Site', 'id']
    ) -> pd.DataFrame:
        """
        Aplica SMOTE para balancear las clases.
        
        Args:
            df: DataFrame para balancear
            target_col: Nombre de la columna objetivo
            exclude_cols: Columnas a excluir del balanceo
            
        Returns:
            DataFrame balanceado
        """
        # Preparar datos para SMOTE
        X = df.drop(exclude_cols, axis=1)
        y = df[target_col]
        
        # Aplicar SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Crear DataFrame balanceado
        balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
        balanced_df[target_col] = y_balanced
        
        logger.info(f"Datos balanceados - Shape final: {balanced_df.shape}")
        return balanced_df
        
    def process_data(
        self,
        input_path: str,
        output_train_path: Optional[str] = None,
        output_val_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta el pipeline completo de preprocesamiento.
        
        Args:
            input_path: Ruta al archivo de entrada
            output_train_path: Ruta opcional para guardar datos de entrenamiento
            output_val_path: Ruta opcional para guardar datos de validación
            
        Returns:
            Tuple de (datos_entrenamiento_procesados, datos_validacion)
        """
        # 1. Cargar datos
        df = self.load_data(input_path)
        
        # 2. Limpieza inicial
        df_clean = self.clean_initial_data(df)
        
        # 3. Eliminar duplicados
        df_unique = self.remove_duplicates(df_clean)
        
        # 4. Eliminar clases pequeñas
        df_filtered = self.remove_small_classes(df_unique)
        
        # 5. Dividir en train/validation
        train_data, val_data = self.split_validation(df_filtered)
        
        # 6. Aplicar SMOTE a los datos de entrenamiento
        train_balanced = self.apply_smote(train_data)
        
        # Guardar datos si se especifican rutas
        if output_train_path:
            train_balanced.to_excel(output_train_path, index=False)
            logger.info(f"Datos de entrenamiento guardados en {output_train_path}")
            
        if output_val_path:
            val_data.to_excel(output_val_path, index=False)
            logger.info(f"Datos de validación guardados en {output_val_path}")
            
        return train_balanced, val_data

# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar preprocesador
    preprocessor = DataPreprocessor(
        random_state=786,
        min_class_size=10,
        validation_split=0.1
    )
    
    # Procesar datos
    train_data, val_data = preprocessor.process_data(
        input_path="data/raw/input_data.xlsx",
        output_train_path="data/processed/train_data.xlsx",
        output_val_path="data/processed/validation_data.xlsx"
    )