# src/training/specific_models_trainer.py
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecificModelTrainer:
    """
    Class for training and evaluating specific models for different feature subsets.
    """
    
    def __init__(
        self,
        random_state: int = 123,
        target_column: str = 'Site',
        output_dir: str = 'outputs',
        class_names: List[str] = None
    ):
        """
        Initialize the specific model trainer.
        
        Args:
            random_state: Seed for reproducibility
            target_column: Name of target column
            output_dir: Directory for saving results
            class_names: Names of the classes for visualization
        """
        self.random_state = random_state
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.class_names = class_names or ['Can_Tintorer', 'Terena', 'Aliste']
        self.features_pool = None
        self.trained_models = {}
        self.calibrated_models = {}
        self.evaluation_reports = {}
        self.setup_dirs()

    def setup_dirs(self):
        """Create necessary directory structure."""
        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    
    def set_features_pool(self, features_pool: Dict[str, List[str]]):
        """
        Set the features pool dictionary.
        
        Args:
            features_pool: Dictionary with feature subset configurations
        """
        self.features_pool = features_pool
        logger.info(f"Features pool set with {len(features_pool)} configurations")
        
    def train_all_models(self, train_data: pd.DataFrame, validation_data: pd.DataFrame) -> Dict:
        """
        Train all models defined in features_pool.
        
        Args:
            train_data: Training data
            validation_data: Validation data
            
        Returns:
            Dictionary with evaluation reports
        """
        if self.features_pool is None:
            raise ValueError("Features pool must be set before training models")
            
        logger.info("Starting training of all specific models...")
        
        # SMOTE instance for reuse
        smote = SMOTE(random_state=self.random_state)
        
        # Iterate over each feature configuration
        for config_name, features in self.features_pool.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing configuration: {config_name}")
            logger.info(f"{'='*50}")
            
            # Train and evaluate the specific model
            self.train_and_evaluate(
                train_data=train_data,
                validation_data=validation_data,
                config_name=config_name,
                features=features,
                smote=smote
            )
            
        # Print performance summary
        self._print_performance_summary()
        
        return self.evaluation_reports
            
    def train_and_evaluate(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        config_name: str,
        features: List[str],
        smote: Optional[SMOTE] = None
    ) -> pd.DataFrame:
        """
        Train and evaluate a specific model configuration.
        
        Args:
            train_data: Training data
            validation_data: Validation data
            config_name: Name of the configuration
            features: List of features for this configuration
            smote: SMOTE instance (optional)
            
        Returns:
            DataFrame with validation predictions
        """
        try:
            # Extract feature subsets
            data_subset = train_data[features]
            fvs_subset = validation_data[features]
            
            # Separate features and target
            X = data_subset.drop([self.target_column], axis=1)
            y = data_subset[self.target_column]
            
            # Validation data
            X_val = fvs_subset.drop([self.target_column], axis=1)
            y_val = fvs_subset[self.target_column]
            
            logger.info(f"Training data shape: {X.shape}, class distribution: {y.value_counts()}")
            logger.info(f"Validation data shape: {X_val.shape}, class distribution: {y_val.value_counts()}")
            
            # Apply SMOTE for balancing
            if smote is None:
                smote = SMOTE(random_state=self.random_state)
                
            X_balanced, y_balanced = smote.fit_resample(X, y)
            logger.info(f"Balanced data shape: {X_balanced.shape}, distribution: {pd.Series(y_balanced).value_counts()}")
            
            # Split for calibration
            X_train, X_calib, y_train, y_calib = train_test_split(
                X_balanced, y_balanced,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_balanced
            )
            
            # Get model name from config
            model_name = config_name.replace('trainning_features_', 'rf_')
            logger.info(f"Training model {model_name}...")
            
            # Train model
            model = self._train_model(X_train, y_train)
            self.trained_models[model_name] = model
            
            # Calibrate model
            logger.info(f"Calibrating model {model_name}...")
            calibrated_model = self._calibrate_model(model, X_calib, y_calib)
            self.calibrated_models[model_name] = calibrated_model
            
            # Evaluate model
            logger.info(f"Evaluating model {model_name}...")
            predictions_df = self._evaluate_model(
                model=calibrated_model,
                validation_data=fvs_subset,
                X_val=X_val,
                model_name=model_name
            )
            
            # Save model
            self._save_model(calibrated_model, model_name)
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error in training and evaluation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train a Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained Random Forest model
        """
        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            criterion='entropy',
            random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        return model
        
    def _calibrate_model(
        self, 
        model: RandomForestClassifier, 
        X_calib: pd.DataFrame, 
        y_calib: pd.Series
    ) -> CalibratedClassifierCV:
        """
        Calibrate a model.
        
        Args:
            model: Model to calibrate
            X_calib: Calibration features
            y_calib: Calibration target
            
        Returns:
            Calibrated model
        """
        calibrator = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
        calibrator.fit(X_calib, y_calib)
        return calibrator
        
    def _evaluate_model(
        self,
        model: CalibratedClassifierCV,
        validation_data: pd.DataFrame,
        X_val: pd.DataFrame,
        model_name: str
    ) -> pd.DataFrame:
        """
        Evaluate model on validation data.
        
        Args:
            model: Model to evaluate
            validation_data: Complete validation data
            X_val: Validation features
            model_name: Name of the model
            
        Returns:
            DataFrame with predictions
        """
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Create predictions DataFrame
        predictions_df = validation_data.copy()
        predictions_df['predictions'] = y_pred
        
        # Generate classification report
        y_true = predictions_df[self.target_column]
        class_rep = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
        
        # Store and save report
        self.evaluation_reports[model_name] = class_rep
        report_path = self.output_dir / f"tables/{model_name}_classification_report.csv"
        class_rep.to_csv(report_path)
        logger.info(f"Classification report saved in: {report_path}")
        
        # Generate and save confusion matrix
        self._generate_confusion_matrix(y_true, y_pred, model_name)
        
        return predictions_df
    
    def _generate_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_name: str
    ) -> None:
        """
        Generate and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names
        )
        disp.plot(cmap='Greens')
        plt.title(f"Confusion Matrix for {model_name}")
        
        # Save confusion matrix
        cm_path = self.output_dir / f"plots/{model_name}_confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved in: {cm_path}")
    
    def _save_model(self, model: CalibratedClassifierCV, model_name: str) -> None:
        """
        Save model to file.
        
        Args:
            model: Model to save
            model_name: Name of the model
        """
        model_path = self.output_dir / f"models/{model_name}_calibrated.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved in: {model_path}")
    
    def _print_performance_summary(self) -> None:
        """Print a summary of all model performances."""
        logger.info("\n\n" + "="*80)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("="*80)
        
        for model_name, report in self.evaluation_reports.items():
            logger.info(f"\nModel: {model_name}")
            logger.info(f"Accuracy: {report.loc['accuracy']['f1-score']:.4f}")
            logger.info(f"F1-score (macro avg): {report.loc['macro avg']['f1-score']:.4f}")
            logger.info(f"F1-score (weighted avg): {report.loc['weighted avg']['f1-score']:.4f}")
            logger.info("-"*50)
        
        logger.info("\nProcess completed successfully!")