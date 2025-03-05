#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real World Use Case Module

This module provides functionality for performing model predictions,
uncertainty analysis, and provenance determination for archaeological
green phosphate samples.
"""

import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Union, Optional
import datetime
from pycaret.classification import load_model, predict_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import seaborn as sns


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealWorldUseCase:
    """
    Class for performing model predictions, uncertainty analysis,
    and provenance determination.
    """
    
    def __init__(
        self,
        random_state: int = 123,
        target_column: str = 'Site',
        output_dir: str = 'outputs',
        class_names: Optional[List[str]] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the use case processor.
        
        Args:
            random_state: Seed for reproducibility
            target_column: Name of target column
            output_dir: Directory for saving results
            class_names: Names of the classes for visualization
            confidence_threshold: Threshold to mark prediction as uncertain
        """
        self.random_state = random_state
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.class_names = class_names or ['CT', 'PCM', 'PDLC']
        self.class_labels = {
            'CT': 'Gavá',
            'PCM': 'Encinasola',
            'PDLC': 'Aliste'
        }
        self.confidence_threshold = confidence_threshold
        
        # These will be set later
        self.data_pool = None
        self.model_pool = None
        self.score_columns = [f'prediction_score_{cls}' for cls in self.class_names]
        
        # Define mapping between datasets and models
        self.data_model_mapping = {
            'pq': 'PQModel',
            'vdh': 'VdHModel',
            'fs': 'FrenchModel',
            # All others will use 'full_model' by default
        }
        
        # Storage for combined results
        self.all_predictions = []
        self.all_uncertainty_results = []
        self.all_provenance_results = []
        
        # Create directory structure
        self.setup_dirs()
        
        logger.info(f"Initialized RealWorldUseCase with {len(self.class_names)} classes")
        
    def setup_dirs(self):
        """Create necessary directory structure."""
        (self.output_dir / 'real_world_plots').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'real_world_tables').mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directories in {self.output_dir}")
    
    def set_data_pool(self, data_pool: Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]):
        """
        Set the data pool dictionary.
        
        Args:
            data_pool: Dictionary with different data subset paths or DataFrames
        """
        self.data_pool = data_pool
        logger.info(f"Data pool set with {len(data_pool)} configurations")
        
    def set_model_pool(self, model_pool: Dict[str, Union[object, List[object], str, List[str]]]):
        """
        Set the model pool dictionary.
        
        Args:
            model_pool: Dictionary with different models or model paths
        """
        # Load models if paths are provided
        processed_models = {}
        for key, value in model_pool.items():
            if isinstance(value, list):
                # Handle list of model paths
                if isinstance(value[0], str):
                    # Load each model from path
                    loaded_models = []
                    for model_path in value:
                        try:
                            loaded_models.append(load_model(model_path))
                            logger.info(f"Loaded model from {model_path}")
                        except Exception as e:
                            logger.error(f"Error loading model from {model_path}: {str(e)}")
                    processed_models[key] = loaded_models
                else:
                    # Already loaded models
                    processed_models[key] = value
            
                
        self.model_pool = processed_models
        logger.info(f"Model pool set with {len(processed_models)} configurations")
        
    
            
    def make_predictions(
        self, 
        model_name: str, 
        data_name: str = None, 
        file_path: str = None, 
        df: pd.DataFrame = None,
        exclude_columns: List[str] = None,
        output_column: str = 'predictions',
        save_output: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions using a specified model on the specified data.
        
        Args:
            model_name: Name of the model in the model pool
            data_name: Name of the dataset in the data pool (optional)
            file_path: Path to an Excel or CSV file (optional)
            df: DataFrame to use for predictions (optional)
            exclude_columns: List of columns to exclude as input
            output_column: Name of the column for the predictions
            save_output: Whether to save the results to a file
            
        Returns:
            DataFrame with predictions
        """
        
        # Input validation - need at least one data source
        if sum(x is not None for x in [data_name, file_path, df]) != 1:
            raise ValueError("Exactly one of data_name, file_path, or df must be provided")
        
        # Get the model
        if model_name not in self.model_pool:
            raise ValueError(f"Model '{model_name}' not found in model pool")
        
        model = self.model_pool[model_name]
        if isinstance(model, list):
            model = model[0]  # Take the first model if a list is provided
            
        # Get the data
        if data_name is not None:
            if self.data_pool is None or data_name not in self.data_pool:
                raise ValueError(f"Data '{data_name}' not found in data pool")
            
            data = self.data_pool[data_name]
            if isinstance(data, list):
                data = data[0]  # Take the first dataset if a list is provided
        elif file_path is not None:
            # Determine file type
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        else:
            data = df.copy()
        
        # Exclude columns safely if specified
        if exclude_columns:
            # Only drop columns that actually exist in the data
            columns_to_drop = [col for col in exclude_columns if col in data.columns]
            if columns_to_drop:
                data = data.drop(columns=columns_to_drop)
                logger.info(f"Dropped columns: {columns_to_drop}")
            
            # Log if any columns were not found
            columns_not_found = [col for col in exclude_columns if col not in data.columns]
            if columns_not_found:
                logger.warning(f"Columns not found for dropping: {columns_not_found}")
            
        # Make predictions based on model type
        logger.info(f"Making predictions using model '{model_name}'")
           
        try:    # Make predictions
            predictions = predict_model(model, data=data, raw_score=True)
            
            # Rename prediction column if needed
            if 'Label' in predictions.columns:
                predictions = predictions.rename(columns={'Label': output_column})
            elif 'prediction_label' in predictions.columns:
                predictions = predictions.rename(columns={'prediction_label': output_column})
            
            # Save results if requested
            if save_output and (file_path is not None):
                base_name = os.path.splitext(file_path)[0]
                output_xlsx = f"{base_name}_predictions.xlsx"
                output_csv = f"{base_name}_predictions.csv"
                
                predictions.to_excel(output_xlsx, index=False)
                predictions.to_csv(output_csv, index=False)
                logger.info(f"Predictions saved to {output_xlsx} and {output_csv}")
                
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with PyCaret model: {str(e)}")
            
    
    def normalize_prediction_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize prediction score columns to ensure they are numeric.
        
        Args:
            df: DataFrame with prediction scores
            
        Returns:
            DataFrame with normalized score columns
        """
        result_df = df.copy()
        
        # Process score columns
        for col in self.score_columns:
            if col in result_df.columns:
                # Check if conversion is needed
                if result_df[col].dtype == object:
                    # Try to handle comma as decimal separator
                    result_df[col] = result_df[col].str.replace(',', '.').astype(float)
                elif not np.issubdtype(result_df[col].dtype, np.number):
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        return result_df
    
    def analyze_uncertainty(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = None,
        id_column: str = 'id',
        prediction_column: str = 'predictions'
    ) -> pd.DataFrame:
        """
        Performs uncertainty analysis per site.
        
        Args:
            df: DataFrame with probabilistic predictions
            confidence_threshold: Threshold to mark prediction as uncertain (optional)
            id_column: Name of the ID column
            prediction_column: Name of the prediction column
            
        Returns:
            DataFrame with uncertainty analysis results
        """
        # Use instance threshold if none is provided
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        # Normalize data
        df = self.normalize_prediction_scores(df)
        
        # Ensure all score columns exist
        for col in self.score_columns:
            if col not in df.columns:
                logger.warning(f"Score column {col} not found in data. Adding with default values.")
                df[col] = 1.0 / len(self.score_columns)
        
        # Obtain probabilities
        probas = df[self.score_columns].values
        
        # Obtain prediction labels and confidence
        predictions = df[prediction_column].values if prediction_column in df.columns else None
        confidences = np.max(probas, axis=1)
        
        # Mark predictions below the threshold as uncertain
        uncertain_mask = confidences < confidence_threshold
        predictions_with_uncertainty = predictions.copy() if predictions is not None else np.argmax(probas, axis=1)
        predictions_with_uncertainty[uncertain_mask] = 'uncertain'
        
        # Calculate entropy
        entropies = np.array([entropy(probs, base=2) for probs in probas])
        
        # Create DataFrame with results
        results = {
            'entropy': entropies
        }
        
        # Add ID column if it exists
        if id_column in df.columns:
            results['id'] = df[id_column]
        else:
            results['id'] = range(len(df))
            
        # Add site column if it exists
        if self.target_column in df.columns:
            results['Site'] = df[self.target_column]
        else:
            # Use a default site if none exists (to avoid later errors)
            results['Site'] = 'Unknown'
        
        # Add original predictions if available
        if predictions is not None:
            results['Original_predictions'] = predictions
            
        # Add prediction scores
        for i, col in enumerate(self.score_columns):
            results[col] = df[col]
            
        # Add uncertainty threshold predictions
        results['Uncertainty_threshold_predictions'] = predictions_with_uncertainty
        
        results_df = pd.DataFrame(results)
        
        # Display summary statistics
        n_uncertain = np.sum(uncertain_mask)
        n_total = len(df)
        pct_uncertain = (n_uncertain / n_total) * 100
        
        logger.info(f"Uncertain predictions: {n_uncertain}/{n_total} ({pct_uncertain:.1f}%)")
        logger.info(f"Mean dataset entropy: {entropies.mean():.3f}")
        
        # Calculate and display median entropy per site
        if self.target_column in df.columns:
            logger.info("\nMedian entropy per site:")
            entropy_median_by_site = results_df.groupby('Site')['entropy'].median()
            for site, median in entropy_median_by_site.items():
                logger.info(f"{site}: {median:.3f}")
        
        return results_df
    
    def determine_provenance(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = None
    ) -> pd.DataFrame:
        """
        Performs consensus provenance determination based on a confidence threshold.
        Creates a summary statistics table.
        
        Args:
            df: DataFrame with probabilistic predictions
            confidence_threshold: Threshold to mark prediction as uncertain (optional)
            
        Returns:
            DataFrame with provenance determination results
        """
        # Use instance threshold if none is provided
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        # Normalize data if needed
        df = self.normalize_prediction_scores(df)
        
        # Calculate max probability for each sample
        df['max_prob'] = df[self.score_columns].max(axis=1)
        
        results = []
        for site in df['Site'].unique():
            site_data = df[df['Site'] == site]
            
            # Calculate uncertainty percentage
            n_uncertain = sum(site_data['max_prob'] < confidence_threshold)
            total_samples = len(site_data)
            
            # Prevent division by zero
            if total_samples > 0:
                pct_uncertain = n_uncertain / total_samples * 100
            else:
                pct_uncertain = 0
                logger.warning(f"No samples found for site {site}")
                continue
            
            # Get high confidence samples
            high_conf = site_data[site_data['max_prob'] >= confidence_threshold]
            
            # Calculate entropy if not already present
            if 'entropy' not in site_data.columns:
                # Calculate entropy 
                site_data['entropy'] = site_data[self.score_columns].apply(
                    lambda x: entropy(x, base=2), axis=1
                )
                
            median_entropy = site_data['entropy'].median()
            
            # Determine consensus
            if len(high_conf) > 0:
                # Get most frequent prediction among high confidence samples
                if 'Uncertainty_threshold_predictions' in high_conf.columns:
                    consensus_series = high_conf['Uncertainty_threshold_predictions'].mode()
                else:
                    # Use column with highest probability
                    consensus_series = high_conf[self.score_columns].idxmax(axis=1).mode()
                    
                consensus = consensus_series.iloc[0]
                if isinstance(consensus, str) and consensus.startswith('prediction_score_'):
                    consensus = consensus.replace('prediction_score_', '')
                    
                # Calculate consistency (percentage of high confidence samples with consensus prediction)
                if 'Uncertainty_threshold_predictions' in high_conf.columns:
                    n_consensus = sum(high_conf['Uncertainty_threshold_predictions'] == consensus)
                else:
                    n_consensus = sum(high_conf[self.score_columns].idxmax(axis=1) == f'prediction_score_{consensus}')
                    
                consistency = n_consensus / len(high_conf) if len(high_conf) > 0 else 0
                n_consensus_pred = len(high_conf)
            else:
                consensus = 'No consensus'
                consistency = 0
                n_consensus_pred = 0
            
            # Count predictions per class
            class_counts = {}
            if 'Original_predictions' in site_data.columns:
                for cls in self.class_names:
                    class_counts[self.class_labels.get(cls, cls)] = sum(site_data['Original_predictions'] == cls)
            else:
                # Use max probability column
                for cls in self.class_names:
                    score_col = f'prediction_score_{cls}'
                    if score_col in site_data.columns:
                        class_counts[self.class_labels.get(cls, cls)] = sum(
                            site_data[self.score_columns].idxmax(axis=1) == score_col
                        )
            
            # Create result row
            result = {
                'Site': site,
                'Samples_analyzed': len(site_data),
                'Uncertain(%)': round(pct_uncertain),
                'Samples_for_provenance': n_consensus_pred,
                'Median_entropy': round(median_entropy, 2),
                'Consensus': consensus,
                'Homogeneity': round(consistency, 2)
            }
            
            # Add class counts
            result.update(class_counts)
            
            results.append(result)
        
        # Return empty DataFrame if no results
        if not results:
            logger.warning("No provenance results generated")
            empty_results = {
                'Site': ['No sites'],
                'Samples_analyzed': [0],
                'Uncertain(%)': [0],
                'Samples_for_provenance': [0],
                'Median_entropy': [0],
                'Consensus': ['None'],
                'Homogeneity': [0]
            }
            # Add class counts
            for cls in self.class_names:
                empty_results[self.class_labels.get(cls, cls)] = [0]
                
            return pd.DataFrame(empty_results)
            
        return pd.DataFrame(results)
    
    def plot_entropy_distribution(
        self,
        df: pd.DataFrame,
        output_file: str = None,
        show_plot: bool = True
    ):
        """
        Creates an entropy and probability distribution plot.
        
        Args:
            df: DataFrame with probabilistic predictions
            output_file: File path to save the plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            DataFrame with site statistics
        """
        # Normalize data if needed
        df = self.normalize_prediction_scores(df)
        
        # Calculate entropy if not already present
        if 'entropy' not in df.columns:
            df['entropy'] = df[self.score_columns].apply(
                lambda x: entropy(x, base=2), axis=1
            )
        
        # Check if there are any valid sites
        if 'Site' not in df.columns or df['Site'].nunique() == 0:
            logger.warning("No site information available for plotting")
            # Return empty stats
            empty_stats = pd.DataFrame({
                'median_entropy': [0],
                'n_samples': [0],
                'mean_entropy': [0],
                'std_entropy': [0]
            }, index=['No sites'])
            return empty_stats
        
        # Calculate median values by site
        site_medians = df.groupby('Site')[self.score_columns + ['entropy']].median()
        
        # Configure the chart style
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create stacked bar chart
        bottom = np.zeros(len(site_medians))
        colors = ['#4B0082', '#228B22', '#B8860B']  # Colors for different classes
        
        # Direct use of site names as x-positions
        x = np.arange(len(site_medians.index))
        
        for i, col in enumerate(self.score_columns):
            ax.bar(x, site_medians[col], bottom=bottom, 
                label=col.replace('prediction_score_', ''),
                color=colors[i], alpha=0.7)
            bottom += site_medians[col]
            
        # Add entropy line
        ax2 = ax.twinx()
        ax2.plot(
            x, site_medians['entropy'],
            color='red', linewidth=2, label='Entropy',
            marker='o'
        )
        
        # Configure axes and labels
        ax.set_xlabel('Sites', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Distribution (Median)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold', color='red')
        
        # Rotate x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(site_medians.index, rotation=45, ha='right')
        
        # Adjust captions
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, labels1 + labels2,
            loc='upper right', bbox_to_anchor=(1.15, 1)
        )
        
        # Adjust layout to prevent labels from being cut off
        plt.subplots_adjust(bottom=0.2)
        
        # Save plot if output file is specified
        if output_file:
            if not output_file.endswith(('.png', '.jpg', '.pdf')):
                output_file += '.png'
                
            output_path = self.output_dir / 'real_world_plots' / output_file
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Plot saved to {output_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Create statistics DataFrame
        stats_dict = {
            'median_entropy': site_medians['entropy'],
            'n_samples': df.groupby('Site').size(),
            'mean_entropy': df.groupby('Site')['entropy'].mean(),
            'std_entropy': df.groupby('Site')['entropy'].std()
        }
        
        # Add median prediction scores
        for col in self.score_columns:
            if col in site_medians.columns:
                stats_dict[f'median_{col.replace("prediction_score_", "")}'] = site_medians[col]
        
        stats = pd.DataFrame(stats_dict).round(3)
        
        return stats
    
    def run_full_analysis(
        self,
        data: Union[str, pd.DataFrame],
        model_name: str = None,
        output_prefix: str = 'analysis',
        exclude_columns: List[str] = None,
        confidence_threshold: float = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run a complete analysis pipeline including predictions,
        uncertainty analysis, provenance determination, and visualization.
        
        Args:
            data: Data name in data pool or DataFrame
            model_name: Model name in model pool (optional, will use mapping if not provided)
            output_prefix: Prefix for output files
            exclude_columns: Columns to exclude from predictions
            confidence_threshold: Confidence threshold for uncertainty
            
        Returns:
            Dictionary with all result DataFrames
        """
        results = {}
        
        # Determine model name if not provided (using mapping)
        if model_name is None and isinstance(data, str):
            model_name = self.data_model_mapping.get(data, 'full_model')
            logger.info(f"Using mapped model '{model_name}' for dataset '{data}'")
        
        try:
            # Make predictions with better error handling
            if isinstance(data, str):
                predictions = self.make_predictions(
                    model_name=model_name,
                    data_name=data,
                    exclude_columns=exclude_columns
                )
            else:
                predictions = self.make_predictions(
                    model_name=model_name,
                    df=data,
                    exclude_columns=exclude_columns
                )
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
        
        # Add dataset name to predictions for tracking
        if isinstance(data, str):
            predictions['data_source'] = data
            
        results['predictions'] = predictions
        self.all_predictions.append(predictions)
        
        # Analyze uncertainty
        uncertainty = self.analyze_uncertainty(
            predictions,
            confidence_threshold=confidence_threshold
        )
        results['uncertainty'] = uncertainty
        self.all_uncertainty_results.append(uncertainty)
        
        # Save uncertainty results
        uncertainty_path = self.output_dir / 'real_world_tables' / f"{output_prefix}_uncertainty.xlsx"
        uncertainty.to_excel(uncertainty_path, index=False)
        logger.info(f"Uncertainty analysis saved to {uncertainty_path}")
        

        # Determine provenance with better error handling
        try:
            provenance = self.determine_provenance(
                uncertainty,
                confidence_threshold=confidence_threshold
            )
            results['provenance'] = provenance
            self.all_provenance_results.append(provenance)
            
            # Save provenance results
            provenance_path = self.output_dir / 'tables' / f"{output_prefix}_provenance.xlsx"
            provenance.to_excel(provenance_path, index=False)
            logger.info(f"Provenance determination saved to {provenance_path}")
        except Exception as e:
            logger.error(f"Error in provenance determination: {str(e)}")
            # Continue with the process despite the error
            empty_provenance = pd.DataFrame({
                'Site': ['Error'],
                'Samples_analyzed': [0],
                'Uncertain(%)': [0],
                'Samples_for_provenance': [0],
                'Median_entropy': [0],
                'Consensus': ['Error'],
                'Homogeneity': [0]
            })
            results['provenance'] = empty_provenance
            self.all_provenance_results.append(empty_provenance)

    def run_integrated_analysis(
        self,
        output_prefix: str = 'integrated_analysis',
        exclude_columns_dict: Dict[str, List[str]] = None,
        confidence_threshold: float = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run analysis on all datasets in the data pool, selecting the appropriate 
        model for each based on the mapping, and then combine results into
        integrated outputs.
        
        Args:
            output_prefix: Prefix for output files
            exclude_columns_dict: Dictionary mapping dataset names to columns to exclude
            confidence_threshold: Confidence threshold for uncertainty
            
        Returns:
            Dictionary with combined result DataFrames
        """
        if self.data_pool is None or self.model_pool is None:
            raise ValueError("Data pool and model pool must be set before running analysis")
        
        # Clear any previous results
        self.all_predictions = []
        self.all_uncertainty_results = []
        self.all_provenance_results = []
        
        # Process each dataset
        for data_name in self.data_pool.keys():
            # Get appropriate model name
            model_name = self.data_model_mapping.get(data_name, 'full_model')
            
            # Skip if model doesn't exist
            if model_name not in self.model_pool:
                logger.warning(f"Model '{model_name}' not found for dataset '{data_name}'. Skipping.")
                continue
                
            logger.info(f"Processing dataset '{data_name}' with model '{model_name}'")
            
            # Get dataset-specific exclude columns if available
            exclude_cols = None
            if exclude_columns_dict and data_name in exclude_columns_dict:
                exclude_cols = exclude_columns_dict[data_name]
                logger.info(f"Using dataset-specific exclude columns for {data_name}: {exclude_cols}")
            
            # Run analysis for this dataset
            try:
                self.run_full_analysis(
                    data=data_name,
                    model_name=model_name,
                    output_prefix=f"{data_name}_{output_prefix}",
                    exclude_columns=exclude_cols,
                    confidence_threshold=confidence_threshold
                )
            except Exception as e:
                logger.error(f"Error processing dataset '{data_name}': {str(e)}")
                continue
        
        # Create combined results
        integrated_results = {}
        
        # Combine all predictions
        if self.all_predictions:
            try:
                combined_predictions = pd.concat(self.all_predictions, ignore_index=True)
                integrated_results['combined_predictions'] = combined_predictions
                
                # Save combined predictions
                combined_predictions_path = self.output_dir / 'tables' / f"{output_prefix}_combined_predictions.xlsx"
                combined_predictions.to_excel(combined_predictions_path, index=False)
                logger.info(f"Combined predictions saved to {combined_predictions_path}")
            except Exception as e:
                logger.error(f"Error combining predictions: {str(e)}")
        
        # Combine all uncertainty results
        if self.all_uncertainty_results:
            try:
                combined_uncertainty = pd.concat(self.all_uncertainty_results, ignore_index=True)
                integrated_results['combined_uncertainty'] = combined_uncertainty
                
                # Save combined uncertainty results
                combined_uncertainty_path = self.output_dir / 'real_world_tables' / f"{output_prefix}_combined_uncertainty.xlsx"
                combined_uncertainty.to_excel(combined_uncertainty_path, index=False)
                logger.info(f"Combined uncertainty analysis saved to {combined_uncertainty_path}")
                
                # Create integrated plot using all data
                try:
                    combined_stats = self.plot_entropy_distribution(
                        combined_uncertainty,
                        output_file=f"{output_prefix}_combined_entropy_distribution.png",
                        show_plot=False
                    )
                    integrated_results['combined_stats'] = combined_stats
                    
                    # Save combined stats
                    combined_stats_path = self.output_dir / 'real_world_tables' / f"{output_prefix}_combined_statistics.csv"
                    combined_stats.to_csv(combined_stats_path)
                    logger.info(f"Combined statistics saved to {combined_stats_path}")
                except Exception as e:
                    logger.error(f"Error creating combined entropy plot: {str(e)}")
            except Exception as e:
                logger.error(f"Error combining uncertainty results: {str(e)}")
        
        # Combine all provenance results
        if self.all_provenance_results:
            try:
                combined_provenance = pd.concat(self.all_provenance_results, ignore_index=True)
                integrated_results['combined_provenance'] = combined_provenance
                
                # Save combined provenance results
                combined_provenance_path = self.output_dir / 'tables' / f"{output_prefix}_combined_provenance.xlsx"
                combined_provenance.to_excel(combined_provenance_path, index=False)
                logger.info(f"Combined provenance determination saved to {combined_provenance_path}")
            except Exception as e:
                logger.error(f"Error combining provenance results: {str(e)}")
        
        # Create a final summary of all processed datasets
        try:
            summary = pd.DataFrame({
                'Dataset': list(self.data_pool.keys()),
                'Model_used': [self.data_model_mapping.get(k, 'full_model') for k in self.data_pool.keys()],
                'Rows': [len(self.data_pool[k]) for k in self.data_pool.keys()],
                'Processed': [k in [p.get('data_source', None) for p in self.all_predictions if isinstance(p, pd.DataFrame) and 'data_source' in p.columns] for k in self.data_pool.keys()]
            })
            
            # Save summary
            summary_path = self.output_dir / 'real_world_tables' / f"{output_prefix}_processing_summary.xlsx"
            summary.to_excel(summary_path, index=False)
            logger.info(f"Processing summary saved to {summary_path}")
            integrated_results['processing_summary'] = summary
        except Exception as e:
            logger.error(f"Error creating processing summary: {str(e)}")
        
        return integrated_results

# Bloque main para el script
if __name__ == '__main__':
    # Example usage
    use_case = RealWorldUseCase(output_dir='./outputs')
    
    try:
        # Set up data and model pools with actual data
        data_pool = {}
        
        # Load data from Excel sheets with proper error handling
        try:
            excel_path = '/home/dsg/VORTEX_FINAL/PRODUCTION/DATA/real_world/real_world_data.xlsx'
            sheets = ['quiruelas', 'v_higueras', 'Alberite', 'Paternanbidea', 
                      'Can_Gambus', 'CatalonianSites', 'FrenchSites']
            
            # Load one by one with error handling
            for sheet, key in zip(sheets, ['pq', 'vdh', 'da', 'pa', 'cg', 'cs', 'fs']):
                try:
                    df = pd.read_excel(excel_path, sheet_name=sheet, engine='openpyxl')
                    

                    
                    # Store the processed dataframe
                    data_pool[key] = df
                    logger.info(f"Loaded sheet '{sheet}' with {len(df)} rows")
                except Exception as e:
                    logger.error(f"Error loading sheet '{sheet}': {str(e)}")
        except Exception as e:
            logger.error(f"Error accessing Excel file: {str(e)}")
            logger.info("Using empty data pool due to data loading errors")
        
        # Check if any data was loaded
        if not data_pool:
            logger.warning("No data was loaded.")
            
        # Set up model pool with error handling - using PyCaret load_model
        model_pool = {}
        model_paths = {
            'full_model': '/home/dsg/VORTEX_FINAL/PRODUCTION/notebooks/models/final_model_direct',
            'VdHModel': '/home/dsg/VORTEX_FINAL/PRODUCTION/models/20250227_VdHSpecific',
            'PQModel': '/home/dsg/VORTEX_FINAL/PRODUCTION/models/20250227_QuiruelasSpecific',
            'FrenchModel': '/home/dsg/VORTEX_FINAL/PRODUCTION/models/20250227_FrenchSpecific'
        }

        # Define dataset-specific columns to exclude
        exclude_columns_by_dataset = { }
        
        for model_name, model_path in model_paths.items():
            try:
                # Intentar primero con load_model de PyCaret
                try:
                    model = load_model(model_path)
                    logger.info(f"Loaded model '{model_name}' using PyCaret from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load with PyCaret: {str(e)}")
                    
                
                # Print model feature names if available (for debugging)
                if hasattr(model, 'feature_names_in_'):
                    features = model.feature_names_in_
                    logger.info(f"Model {model_name} expects features: {features[:5]}... (total: {len(features)})")
                
                model_pool[model_name] = model
                
            except Exception as e:
                logger.error(f"Error loading model '{model_name}': {str(e)}")
        
        # Set data and model pools
        use_case.set_data_pool(data_pool)
        use_case.set_model_pool(model_pool)
        
        #  run_integrated_analysis
        logger.info("Starting integrated analysis...")
        integrated_results = use_case.run_integrated_analysis(
            exclude_columns_dict=exclude_columns_by_dataset
        )
        logger.info("Integrated analysis completed successfully")
        print("Integrated analysis complete!")
        
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())