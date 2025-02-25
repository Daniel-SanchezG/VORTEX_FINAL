# main.py

import argparse
from pathlib import Path
import logging
from datetime import datetime
from src.preprocessing.data_processor import DataPreprocessor
from src.training.model_trainer import ModelTrainer
from src.analysis.feature_importance_analyzer import FeatureImportanceAnalyzer

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
        
        # Primero completamos el entrenamiento y evaluación
        predictions = trainer.train_and_evaluate(
            train_data=train_data,
            validation_data=val_data
        )
        
        # Parte relevante del main.py que necesita cambiar:

        # 2. Training and evaluation
        logger.info("Starting model training...")
        trainer = ModelTrainer(
            random_state=123,
            output_dir=output_dir
        )
        
        # Primero completamos el entrenamiento y evaluación
        predictions = trainer.train_and_evaluate(
            train_data=train_data,
            validation_data=val_data
        )
        
        # 3. Feature Importance Analysis
        logger.info("Starting feature importance analysis...")
        
        # Get features and target for analysis
        X = train_data.drop(['Site', 'id'] if 'id' in train_data.columns else ['Site'], axis=1)
        y = train_data['Site']
        
        # Ahora podemos obtener el modelo tunificado con seguridad
        logger.info("Retrieving tuned model for feature importance analysis...")
        tuned_model = trainer.get_tuned_model()
        
        # Create analyzer
        analyzer = FeatureImportanceAnalyzer(
            output_dir=output_dir,
            class_names=['Gavá', 'Terena', 'Aliste'],
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
        
        # Log results summary
        importance_df = results['feature_importance']
        top_features = importance_df.head(5)
        logger.info("\nTop 5 most important features:")
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
            f.write(f"Number of training samples: {len(train_data)}\n")
            f.write(f"Number of validation samples: {len(val_data)}\n")
            f.write(f"\nModel Configuration\n")
            f.write(f"===================\n")
            f.write(f"Random state (preprocessing): 786\n")
            f.write(f"Random state (training): 123\n")
            f.write(f"\nTop Features by Importance\n")
            f.write(f"========================\n")
            for _, row in top_features.iterrows():
                f.write(f"- {row['feature']}: {row['importance']:.4f}\n")
        
        logger.info("Process completed successfully!")
        logger.info(f"Results saved in: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during process: {str(e)}")
        raise

if __name__ == "__main__":
    main()