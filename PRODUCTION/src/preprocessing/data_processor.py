# src/preprocessing/data_processor.py

from typing import Dict, List, Optional, Tuple
import pandas as pd # type: ignore
import numpy as np
from imblearn.over_sampling import SMOTE
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for XRF data pre-processing, including cleaning,
    case elimination and class balancing.
    """
    
    def __init__(
        self,
        random_state: int = 786,
        min_class_size: int = 10,
        validation_split: float = 0.1
    ):
        """
        Initialises the data preprocessor.
        
        Args:
            random_state: Seed for reproducibility.
            min_class_size: minimum class size to maintain
            validation_split: data fraction for final validation
        """
        self.random_state = random_state
        self.min_class_size = min_class_size
        self.validation_split = validation_split
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from Excel or CSV file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with loaded data
        """
        path = Path(file_path)
        try:
            if path.suffix == '.xlsx':
                df = pd.read_excel(file_path, engine='openpyxl')
            elif path.suffix == '.csv':
                df = pd.read_csv(file_path, encoding='latin-1')
            else:
                raise ValueError(f"File format not supported: {path.suffix}")
                
            logger.info(f"Data successfully uploaded: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def clean_initial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs initial data cleansing.
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            Clean DataFrame
        """
        # Delete initial unnecessary columns
        columns_to_drop = list(df.iloc[:, :22].columns) + ['suma']
        data = df.drop(columns=columns_to_drop, axis=1)
        
        # Add important columns
        data['Site'] = df['Site']
        data['id'] = df['ID']
        
        # Check for missing values
        if data.isnull().any().any():
            logger.warning("Missing values found in the data")
            
        return data
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Delete duplicate records based on ID.
        
        Args:
            df: DataFrame with possible duplicates
            
        Returns:
            DataFrame without duplicates
        """
        # Identify duplicates
        duplicates = df['id'].duplicated().sum()
        if duplicates > 0:
            logger.info(f"Found {duplicates} duplicate IDs")
            
        # Delete duplicates
        df_clean = df.drop_duplicates(subset='id', keep='first')
        df_clean.reset_index(drop=True, inplace=True)
        
        logger.info(f"Unique records after deleting duplicates: {df_clean.shape[0]}")
        return df_clean
        
    def remove_small_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Delete classes with less cases than min_class_size.
        
        Args:
            df: DataFrame with all classes
            
        Returns:
            DataFrame without small classes
        """
        # Identify small classes
        class_counts = df['Site'].value_counts()
        small_classes = class_counts[class_counts < self.min_class_size].index
        
        if len(small_classes) > 0:
            logger.info(f"Deleting {len(small_classes)} classes with less than {self.min_class_size} cases")
            df_filtered = df[~df['Site'].isin(small_classes)]
            df_filtered.reset_index(drop=True, inplace=True)
            return df_filtered
        
        return df
        
    def split_validation(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide the data into training and final validation sets. Creates Final Validation Set (FVS) 
        
        Args:
            df: Complete DataFrame
            
        Returns:
            Tuple of (training data, final validation data)
        """
        # Split data
        train_data = df.sample(
            frac=1-self.validation_split,
            random_state=self.random_state
        )
        val_data = df.drop(train_data.index)
        
        # Reset indices
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        
        logger.info(f"Data split - Training: {train_data.shape}, Validation: {val_data.shape}")
        return train_data, val_data
        
    def apply_smote(
        self,
        df: pd.DataFrame,
        target_col: str = 'Site',
        exclude_cols: List[str] = ['Site', 'id']
    ) -> pd.DataFrame:
        """
        Applies SMOTE to balance the classes.
        
        Args:
            df: DataFrame to balance
            target_col: Name of the target column
            exclude_cols: Columns to exclude from balancing
            
        Returns:
            Balanced DataFrame
        """
        # Prepare data for SMOTE
        X = df.drop(exclude_cols, axis=1)
        y = df[target_col]
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Create balanced DataFrame
        balanced_df = pd.DataFrame(X_balanced, columns=X.columns)
        balanced_df[target_col] = y_balanced
        
        logger.info(f"Balanced data - Final shape: {balanced_df.shape}")
        return balanced_df
        
    def process_data(
        self,
        input_path: str,
        output_train_path: Optional[str] = None,
        output_val_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the complete pre-processing pipeline.
        
        Args:
            input_path: Path to the input file
            output_train_path: Optional path to save training data
            output_val_path: Optional path to save validation data
            
        Returns:
            Tuple of (processed training data, validation data)
        """
        # 1. Load data
        df = self.load_data(input_path)
        
        # 2. Initial cleaning
        df_clean = self.clean_initial_data(df)
        
        # 3. Delete duplicates
        df_unique = self.remove_duplicates(df_clean)
        
        # 4. Delete small classes
        df_filtered = self.remove_small_classes(df_unique)
        
        # 5. Split into train/validation
        train_data, val_data = self.split_validation(df_filtered)
        
        # 6. Apply SMOTE to training data
        train_balanced = self.apply_smote(train_data)
        
        # Save data if paths are specified
        if output_train_path:
            train_balanced.to_excel(output_train_path, index=False)
            logger.info(f"Training data saved in {output_train_path}")
            
        if output_val_path:
            val_data.to_excel(output_val_path, index=False)
            logger.info(f"Validation data saved in {output_val_path}")
            
        return train_balanced, val_data

# Example of usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        random_state=786,
        min_class_size=10,
        validation_split=0.1
    )
    
    # Process data
    train_data, val_data = preprocessor.process_data(
        input_path="data/raw/input_data.xlsx",
        output_train_path="data/processed/train_data.xlsx",
        output_val_path="data/processed/validation_data.xlsx"
    )