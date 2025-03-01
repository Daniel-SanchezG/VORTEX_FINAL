# src/training/model_trainer.py

from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from pycaret.classification import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training and evaluating classification models.
    """
    
    def __init__(
        self,
        random_state: int = 123,
        target_column: str = 'Site',
        output_dir: str = 'outputs'
    ):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Seed for reproducibility
            target_column: Name of target column
            output_dir: Directory for saving results
        """
        self.random_state = random_state
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.model = None
        self.experiment_setup = False
        self.setup_dirs()
        
    def setup_dirs(self):
        """Create necessary directory structure."""
        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tables').mkdir(parents=True, exist_ok=True)
        
    def setup_experiment(
        self,
        data: pd.DataFrame,
        train_size: float = 0.8
    ) -> None:
        """
        Configure the training experiment.
        
        Args:
            data: DataFrame with the data
            train_size: Proportion of data for training
        """
        try:
            logger.info("Configuring training experiment...")
            
            # Prepare data
            features = self.prepare_data(data)
            
            # Configure experiment
            exp = setup(
                data=features,
                target=self.target_column,
                train_size=train_size,
                session_id=self.random_state,
                verbose=False
            )
            
            # Remove unnecessary metrics
            remove_metric('MCC')
            remove_metric('Kappa')
            remove_metric('AUC')
            
            self.experiment_setup = True
            logger.info("Experiment configured successfully")
            
        except Exception as e:
            logger.error(f"Error configuring experiment: {str(e)}")
            raise
            
    def train_model(self) -> None:
        """
        Train the Random Forest model with optimized parameters.
        """
        if not self.experiment_setup:
            raise ValueError("Must run setup_experiment before training the model")
            
        try:
            logger.info("Starting model training...")
            
            # Create and train base model
            self.base_model = create_model(
                'rf',
                n_estimators=200,
                min_samples_leaf=5,
                class_weight="balanced",
                criterion='entropy',
                random_state=42

            )
            
            # Save base model metrics
            rf_metrics = pull()
            rf_metrics.to_csv(
                self.output_dir / 'tables/rf_model_score_grid.csv'
            )
            
            # Optimize hyperparameters and save the tuned model explicitly
            logger.info("Optimizing hyperparameters...")
            self.tuned_model = tune_model(
                self.base_model,
                n_iter=10,
                optimize='F1',
                custom_grid={'criterion': ['entropy'],
                             'min_samples_leaf':[5, 10, 15, 20, 25, 30],
                             'n_estimators':[10,50,100,200]
                             }
            )
            
            # Save tuned model metrics
            tuned_metrics = pull()
            tuned_metrics.to_csv(
                self.output_dir / 'tables/tuned_model_score_grid.csv'
            )
            
            # Save tuned model before calibration for feature importance
            logger.info("Saving tuned model before calibration...")
            save_model(
                self.tuned_model,
                self.output_dir / 'models/tuned_model',
                verbose=True
            )
            
            # Calibrate model
            logger.info("Calibrating model probabilities...")
            self.calibrated_model = calibrate_model(self.tuned_model, method='sigmoid')
            
            # Save calibrated model metrics
            cal_metrics = pull()
            cal_metrics.to_csv(
                self.output_dir / 'tables/calibrated_model_score_grid.csv'
            )
            
            # Finalize and save calibrated model
            self.final_model = finalize_model(self.calibrated_model)
            save_model(
                self.final_model,
                self.output_dir / 'models/final_model',
                verbose=True
            )
            
            self.model = self.final_model

            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def get_tuned_model(self):
        """
        Returns the tuned model before calibration.
        
        Returns:
            The tuned Random Forest model
        
        Raises:
            ValueError: If the model hasn't been trained yet
        """
        try:
            return self.tuned_model
        except AttributeError:
            raise ValueError("Model has not been trained yet. Run train_model() first.")
            
    def evaluate_model(
        self,
        validation_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Evaluate model on validation data.
        
        Args:
            validation_data: DataFrame with validation data
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        try:
            logger.info("Evaluating model on validation data...")
            
            # Prepare validation data
            val_features = self.prepare_data(validation_data)
            
            # Make predictions
            predictions = predict_model(self.model, data=val_features)
            
            # Save predictions
            predictions.to_csv(
                self.output_dir / 'tables/validation_predictions.csv',
                index=False
            )
            
            # Create confusion matrix
            y_true = predictions[self.target_column]
            y_pred = predictions['prediction_label']
            
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 5.5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['Can_tintorer', 'Terena', 'Aliste']
            )
            disp.plot(cmap='Greens')
            plt.savefig(
                self.output_dir / 'plots/confusion_matrix.png',
                bbox_inches='tight'
            )
            plt.close()
            
            # Save classification report
            report = classification_report(
                y_true,
                y_pred,
                output_dict=True
            )
            pd.DataFrame(report).T.to_csv(
                self.output_dir / 'tables/classification_report.csv'
            )
            
            logger.info("Evaluation completed")
            return predictions
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for model.
        
        Args:
            data: DataFrame to prepare
            
        Returns:
            Prepared DataFrame
        """
        # Remove non-feature columns
        columns_to_drop = ['id'] if 'id' in data.columns else []
        features = data.drop(columns=columns_to_drop, errors='ignore')
        return features
            
    def train_and_evaluate(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run complete training and evaluation pipeline.
        
        Args:
            train_data: Training data
            validation_data: Validation data
            
        Returns:
            DataFrame with validation predictions
        """
        # Ensure setup is done first
        self.setup_experiment(train_data)
        
        # Continue with pipeline
        self.train_model()
        predictions = self.evaluate_model(validation_data)
        return predictions