# main.py

import argparse
from pathlib import Path
import logging
import platform
from datetime import datetime
from src.preprocessing.data_processor import DataPreprocessor
from src.training.model_trainer import ModelTrainer
from src.analysis.feature_importance_analyzer import FeatureImportanceAnalyzer
from src.training.specific_models_trainer_pycaret import SpecificModelTrainerPyCaret

def setup_logging(output_dir: Path) -> None:
    """
    Configure logging system to write both to console and file.
    
    Args:
        output_dir: Directory where the log file will be saved
    """
    # Create formatter for logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Handler for console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler for file
    log_file = output_dir / 'logs' / f'log_of_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def parse_arguments():
    """
    Parse command line arguments for the XRF data processing pipeline.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='XRF data processing and training pipeline'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input file (Excel or CSV)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Base directory where results will be saved'
    )
    parser.add_argument(
        '--min-class-size',
        type=int,
        default=10,
        help='Minimum class size to keep'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.1,
        help='Fraction of data for validation (between 0 and 1)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full analysis including feature importance (step 3)'
    )
    return parser.parse_args()

def setup_directories(base_dir: Path, timestamp: str) -> Path:
    """
    Create necessary directories with timestamp.
    
    Args:
        base_dir: Base output directory
        timestamp: Timestamp for this experiment
        
    Returns:
        Path to the specific output directory for this experiment
    """
    # Create specific directory for this experiment
    output_dir = base_dir / f'experiment_{timestamp}'
    
    # Create required subdirectories
    (output_dir / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    return output_dir

def main():
    """
    Main function to run the XRF data processing and training pipeline.
    """
    # Get arguments
    args = parse_arguments()
    
    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure directories
    base_dir = Path(args.output_dir)
    output_dir = setup_directories(base_dir, timestamp)
    
    # Configure logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting experiment with timestamp: {timestamp}")
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Full analysis: {'Yes' if args.full else 'No'}")
        
        # 1. Preprocessing
        logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor(
            random_state=786,
            min_class_size=args.min_class_size,
            validation_split=args.validation_split
        )
        
        train_data, val_data = preprocessor.process_data(
            input_path=args.input,
            output_train_path=output_dir / f"data/processed/training_data_{timestamp}.xlsx",
            output_val_path=output_dir / f"data/processed/validation_data_{timestamp}.xlsx"
        )
        
        # 2. Training and evaluation
        logger.info("Starting model training...")
        trainer = ModelTrainer(
            random_state=123,
            output_dir=output_dir
        )
        
        # First, complete training and evaluation
        predictions = trainer.train_and_evaluate(
            train_data=train_data,
            validation_data=val_data
        )
  
        # Variables for storing information about feature analysis
        top_features = None
        feature_importance_info = ""
        
        # 3. Feature Importance Analysis (only if --full is enabled)
        if args.full:
            logger.info("Starting feature importance analysis...")
            
            # Get features and target for analysis
            X = train_data.drop(['Site', 'id'] if 'id' in train_data.columns else ['Site'], axis=1)
            y = train_data['Site']
            
            # Now we can obtain the tuned model.
            logger.info("Retrieving tuned model for feature importance analysis...")
            tuned_model = trainer.get_tuned_model()
            
            # Create analyzer
            analyzer = FeatureImportanceAnalyzer(
                output_dir=output_dir,
                class_names=['Can_Tintorer', 'Terena', 'Aliste'],
                random_state=123
            )
            
            # Run analysis
            logger.info("Running feature importance analysis...")
            results = analyzer.analyze_feature_importance(
                model=tuned_model,
                X=X,
                y=y,
                n_runs=10
            )
            
            # Save information on top features to include in experiment report
            importance_df = results['feature_importance']
            top_features = importance_df.head(14)
            
            # Prepare information for the report
            feature_importance_info = f"\nTop Features by Importance\n"
            feature_importance_info += f"========================\n"
            for _, row in top_features.iterrows():
                feature_importance_info += f"- {row['feature']}: {row['importance']:.4f}\n"
        else:
            logger.info("Skipping feature importance analysis (use --full to enable)")
            feature_importance_info = "\nFeature importance analysis was skipped. Use --full to enable.\n"

        # 4. Train specific models for different feature subsets
        logger.info("Starting specific models training...")

        # Define feature sets
        features_pool = {
            'trainning_features_Destilled': ['Ca', 'S', 'K', 'Ti', 'V', 'Cr', 'Cu', 
                                          'Zn', 'As', 'Se', 'Sr', 
                                          'Mo', 'Ba', 
                                          'Ta', 'Site'],
            'trainning_features_French': ['Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca',
                                       'Ti', 'V', 'Cr', 'Fe', 'Co', 'Cu', 'Zn',
                                       'As', 'Se', 'Rb', 'Sr', 'Zr', 'Mo', 'Ba',
                                       'Ta', 'Site'],
            'trainning_features_Quiruelas': ['Al', 'Si', 'P', 'K', 'Ca', 
                                           'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
                                           'Ga', 'Zn', 'As', 'Rb', 
                                           'Sr', 'Zr', 'Site'],
            'trainning_features_VdH': ['Al', 'Si', 'P', 'S', 'Cl', 'K',
                                     'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 
                                     'Ni', 'Cu', 'Zn', 'As', 'Rb', 
                                     'Sr', 'Zr', 'Site']
        }

        # Create specific model trainer
        specific_trainer = SpecificModelTrainerPyCaret(
            random_state=123,
            output_dir=output_dir,
            class_names=['Can_Tintorer', 'Terena', 'Aliste']
        )

        # Set features pool and train all models
        specific_trainer.set_features_pool(features_pool)
        specific_reports = specific_trainer.train_all_models(
            train_data=train_data,
            validation_data=val_data
        )

        # Log results summary
        if args.full and top_features is not None:
            logger.info("\nTop 14 most important features:")
            for _, row in top_features.iterrows():
                logger.info(f"- {row['feature']}: {row['importance']:.4f}")
        
        # Save experiment information
        with open(output_dir / 'experiment_info.txt', 'w') as f:
            f.write(f"Experiment Information\n")
            f.write(f"=====================\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Input file: {args.input}\n")
            f.write(f"Minimum class size: {args.min_class_size}\n")
            f.write(f"Validation split: {args.validation_split}\n")
            f.write(f"Full analysis: {'Yes' if args.full else 'No'}\n")
            f.write(f"Number of training samples: {len(train_data)}\n")
            f.write(f"Number of validation samples: {len(val_data)}\n")
            f.write(f"\nModel Configuration\n")
            f.write(f"\nSystem Information\n")
            f.write(f"==================\n")
            f.write(f"Operating System: {platform.system()}\n")
            f.write(f"OS Release: {platform.release()}\n")
            f.write(f"Architecture: {platform.machine()}\n")
            f.write(f"Python Version: {platform.python_version()}\n")
            f.write(f"===================\n")
            f.write(f"Random state (preprocessing): 786\n")
            f.write(f"Random state (training): 123\n")
            f.write(feature_importance_info)
        
        # Add specific models results to experiment info
        with open(output_dir / 'experiment_info.txt', 'a') as f:
            f.write(f"\nSpecific Models Performance\n")
            f.write(f"==========================\n")
            for model_name, report in specific_reports.items():
                f.write(f"\nModel: {model_name}\n")
                f.write(f"Accuracy: {report.loc['accuracy']['f1-score']:.4f}\n")
                f.write(f"F1-score (macro avg): {report.loc['macro avg']['f1-score']:.4f}\n")
                f.write(f"F1-score (weighted avg): {report.loc['weighted avg']['f1-score']:.4f}\n")
                f.write("-" * 30 + "\n")
        
        logger.info("Process completed successfully!")
        logger.info(f"Results saved in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during process: {str(e)}")
        raise

if __name__ == "__main__":
    main()