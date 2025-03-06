# src/training/specific_models_trainer_pycaret.py
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from pycaret.classification import (
    setup, create_model, tune_model, calibrate_model,
    finalize_model, save_model, compare_models, pull,
    predict_model, get_config
)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecificModelTrainerPyCaret:
    """
    Class for training and evaluating specific models for different feature subsets
    using PyCaret, maintaining compatibility with archaeological_predictor.py.
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
        self.current_experiment = None

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
                features=features
            )
            
        # Print performance summary
        self._print_performance_summary()
        
        return self.evaluation_reports
            
    def train_and_evaluate(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        config_name: str,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Train and evaluate a specific model configuration.
        
        Args:
            train_data: Training data
            validation_data: Validation data
            config_name: Name of the configuration
            features: List of features for this configuration
            
        Returns:
            DataFrame with validation predictions
        """
        try:
            # Extract feature subsets 
            # Make sure target column is included in features list
            if self.target_column not in features:
                features_with_target = features + [self.target_column]
            else:
                features_with_target = features

            # Create subsets with only the required features
            data_subset = train_data[features_with_target]
            val_subset = validation_data[features_with_target]
            
            logger.info(f"Training data shape: {data_subset.shape}")
            logger.info(f"Class distribution: {data_subset[self.target_column].value_counts()}")
            logger.info(f"Validation data shape: {val_subset.shape}")
            
            # Get model name from config
            model_name = config_name.replace('trainning_features_', 'rf_')
            logger.info(f"Training model {model_name}...")
            
            # Configure PyCaret experiment with feature subset
            logger.info("Setting up PyCaret experiment...")
            self.current_experiment = setup(
                data=data_subset,
                target=self.target_column,
                session_id=self.random_state,
                train_size=0.8,  # Will use 20% for internal validation
                normalize=True,
                transformation=True, 
                ignore_features=None,  # Already filtered features
                fix_imbalance=True,  # This replaces SMOTE from the original implementation
                verbose=False
            )
            
            # Train base model
            logger.info(f"Creating base Random Forest model...")
            base_model = create_model(
                'rf',
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced",
                criterion='entropy',
                random_state=self.random_state
            )
            
            # Save base model metrics
            rf_metrics = pull()
            rf_metrics.to_csv(
                self.output_dir / f"tables/{model_name}_base_metrics.csv"
            )

            # Tune hyperparameters
            logger.info(f"Tuning model {model_name}...")
            tuned_model = tune_model(
                base_model,
                n_iter=10,
                optimize='F1',
                custom_grid={'criterion': ['entropy'],
                             'min_samples_leaf':[5, 10, 15, 20, 25, 30]
                            }
            )
            
            # Save tuned model metrics
            tuned_metrics = pull()
            tuned_metrics.to_csv(
                self.output_dir / f"tables/{model_name}_tuned_metrics.csv"
            )
            
            # Calibrate model probabilities
            logger.info(f"Calibrating model {model_name}...")
            calibrated_model = calibrate_model(tuned_model, method='sigmoid')
            
            # Save calibrated model metrics
            cal_metrics = pull()
            cal_metrics.to_csv(
                self.output_dir / f"tables/{model_name}_calibrated_metrics.csv"
            )
            
            # Finalize model (train on entire dataset)
            logger.info(f"Finalizing model {model_name}...")
            final_model = finalize_model(calibrated_model)
            
            # Store models
            self.trained_models[model_name] = tuned_model
            self.calibrated_models[model_name] = final_model
            
            # Save model
            model_path = self.output_dir / f"models/{model_name}"
            save_model(final_model, model_path)
            logger.info(f"Model saved at: {model_path}")
            
            # Evaluate model on validation data
            logger.info(f"Evaluating model {model_name} on validation data...")
            predictions = predict_model(
                final_model, 
                data=val_subset,
                raw_score=True  # Get probability scores for each class
            )
            
            # Generate and save classification report
            y_true = predictions[self.target_column]
            y_pred = predictions['prediction_label']
            class_rep = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T
            
            # Store and save report
            self.evaluation_reports[model_name] = class_rep
            report_path = self.output_dir / f"tables/{model_name}_classification_report.csv"
            class_rep.to_csv(report_path)
            logger.info(f"Classification report saved in: {report_path}")
            
            # Generate and save confusion matrix
            self._generate_confusion_matrix(y_true, y_pred, model_name)
            
            # Save predictions for reference
            pred_path = self.output_dir / f"tables/{model_name}_predictions.csv"
            predictions.to_csv(pred_path, index=False)
            logger.info(f"Predictions saved in: {pred_path}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in training and evaluation: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _generate_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
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