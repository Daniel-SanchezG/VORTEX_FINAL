# src/analysis/feature_importance_analyzer.py

from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from collections import Counter
import shap
from matplotlib.colors import LinearSegmentedColormap
import traceback
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """Class for analyzing feature importance using multiple methods."""
    
    def __init__(
        self,
        output_dir: str,
        n_top_features: int = 14,
        class_names: List[str] = None,
        random_state: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.n_top_features = n_top_features
        self.class_names = class_names or ['Class_1', 'Class_2', 'Class_3']
        self.random_state = random_state
        
        # Create output directories if they don't exist
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tables').mkdir(parents=True, exist_ok=True)
        
    def analyze_feature_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_runs: int = 10
    ) -> Dict:
        """Run complete feature importance analysis."""
        try:
            logger.info("Starting feature importance analysis...")
            logger.info(f"Data shape: X={X.shape}, y={y.shape}")
            logger.info(f"Output directory: {self.output_dir}")
            
            # 1. Basic Random Forest Feature Importance
            logger.info("Calculating Random Forest feature importance...")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(
                self.output_dir / 'tables/rf_feature_importance.csv',
                index=False
            )
            
            # 2. Feature Importance Plot
            logger.info("Creating feature importance plot...")
            self._plot_feature_importance_dotplot(feature_importance)
            
            # 3. RFECV Analysis
            logger.info("Starting RFECV analysis...")
            rfecv_results = self._run_rfecv_analysis(model, X, y, n_runs)
            
            # 4. SHAP Analysis
            logger.info("Running SHAP analysis...")
            self._run_shap_analysis(model, X)
            
            # Combine results
            results = {
                'feature_importance': feature_importance,
                'rfecv_results': rfecv_results
            }
            
            logger.info("Feature importance analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _run_rfecv_analysis(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_runs: int
    ) -> Dict:
        """Run RFECV analysis multiple times."""
        try:
            logger.info(f"Running RFECV with {n_runs} iterations...")
            
            results = {
                'n_features_list': [],
                'grid_scores_list': [],
                'feature_masks': []
            }
            
            for i in range(n_runs):
                logger.info(f"RFECV iteration {i+1}/{n_runs}")
                
                cv = StratifiedKFold(
                    n_splits=5,  # Reducido para pruebas
                    shuffle=True,
                    random_state=self.random_state + i
                )
                
                rfecv = RFECV(
                    estimator=clone(model),
                    step=1,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                rfecv.fit(X, y)
                
                results['n_features_list'].append(rfecv.n_features_)
                # Usar cv_results_ en lugar de grid_scores_
                results['grid_scores_list'].append(rfecv.cv_results_['mean_test_score'])
                results['feature_masks'].append(rfecv.support_)
                
                logger.info(f"Iteration {i+1} completed: selected {rfecv.n_features_} features")
            
            # Calculate feature importance frequencies
            feature_freq = np.mean(results['feature_masks'], axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'selection_frequency': feature_freq
            })
            
            # Sort by frequency
            feature_importance_df = feature_importance_df.sort_values(
                'selection_frequency', 
                ascending=False
            )
            
            # Save feature selection frequencies
            feature_importance_df.to_csv(
                self.output_dir / 'tables/rfecv_feature_frequencies.csv',
                index=False
            )
            
            # Plot results
            self._plot_rfecv_results(results['grid_scores_list'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in RFECV analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _plot_rfecv_results(self, grid_scores_list: List) -> None:
        """Plot RFECV results with confidence intervals."""
        try:
            logger.info("Creating RFECV plot...")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            grid_scores_array = np.array(grid_scores_list)
            mean_scores = np.mean(grid_scores_array, axis=0)
            std_scores = np.std(grid_scores_array, axis=0)
            
            x = range(1, len(mean_scores) + 1)
            
            # Plot confidence interval
            ax.fill_between(
                x,
                mean_scores - std_scores,
                mean_scores + std_scores,
                alpha=0.3,
                color='gray',
                label='Standard Deviation'
            )
            
            # Plot mean line
            ax.plot(
                x,
                mean_scores,
                'o-',
                color='gray',
                linewidth=2,
                markersize=4,
                label='Mean CV Score'
            )
            
            ax.set_xlabel('Number of Selected Features', fontsize=12)
            ax.set_ylabel('Cross-validation Score', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(fontsize=10)
            
            plt.tight_layout()
            plt.savefig(
                self.output_dir / 'plots/rfecv_analysis.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            logger.info("RFECV plot saved successfully")
            
        except Exception as e:
            logger.error(f"Error creating RFECV plot: {str(e)}")
            logger.error(traceback.format_exc())
            raise    

    def _plot_feature_importance_dotplot(self, importance_df: pd.DataFrame) -> None:
        """
        Create a dot plot of feature importances.
        
        Args:
            importance_df: DataFrame with feature importances
        """
        try:
            logger.info("Creating feature importance dot plot...")
            
            # Select top features
            top_features = importance_df.head(self.n_top_features)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot horizontal lines and dots
            y_pos = range(len(top_features))
            ax.hlines(
                y=y_pos,
                xmin=0,
                xmax=top_features['importance'],
                color='skyblue',
                alpha=0.7,
                linewidth=2
            )
            ax.plot(
                top_features['importance'],
                y_pos,
                'o',
                markersize=8,
                color='darkblue'
            )
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features['feature'], fontsize=12)
            ax.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
            ax.set_ylabel('Features', fontsize=14, fontweight='bold')
            ax.grid(True, axis='x', linestyle='--', alpha=0.3)
            
            # Add importance values as text
            for i, v in enumerate(top_features['importance']):
                ax.text(v, i, f' {v:.4f}', va='center', fontsize=10)
            
            # Invert y-axis to show most important features at top
            ax.invert_yaxis()
            
            # Set title
            plt.title('Top Feature Importance', fontsize=16, pad=20)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(
                self.output_dir / 'plots/feature_importance_dotplot.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            logger.info("Feature importance plot saved successfully")
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            logger.error(traceback.format_exc())
            raise    

    def _run_shap_analysis(self,model, X: pd.DataFrame) -> None:
        """
        Run SHAP analysis and create summary plot.
        
        Args:
            model: Trained model
            X: Feature matrix
        """
        try:
            logger.info("Running SHAP analysis...")
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Create SHAP summary plot
            plt.figure(figsize=(12, 8))
            
            # For multi-class cases, we want to show all classes
            shap.summary_plot(
                shap_values,
                X,
                plot_type="bar",
                class_names=self.class_names,
                max_display=self.n_top_features,
                plot_size=(12, 8),
                show=False
            )
            
            # Customize plot
            plt.title('SHAP Feature Importance by Class', fontsize=14, pad=20)
            plt.xlabel('mean(|SHAP value|)', fontsize=12)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(
                self.output_dir / 'plots/shap_summary.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
            
            # Save SHAP values for each class
            for i, class_name in enumerate(self.class_names):
                shap_df = pd.DataFrame({
                    'feature': X.columns,
                    f'mean_shap_value_{class_name}': np.mean(np.abs(shap_values[i]), axis=0)
                }).sort_values(f'mean_shap_value_{class_name}', ascending=False)
                
                shap_df.to_csv(
                    self.output_dir / f'tables/shap_values_{class_name}.csv',
                    index=False
                )
            
            # Save overall SHAP values
            mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            overall_shap_df = pd.DataFrame({
                'feature': X.columns,
                'mean_shap_value': np.mean(mean_shap, axis=0)
            }).sort_values('mean_shap_value', ascending=False)
            
            overall_shap_df.to_csv(
                self.output_dir / 'tables/shap_values_overall.csv',
                index=False
            )
            
            logger.info("SHAP analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise