# main.py

from src.preprocessing.data_processor import DataPreprocessor
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Inicializar el preprocesador
    preprocessor = DataPreprocessor(
        random_state=786,
        min_class_size=10,
        validation_split=0.1
    )
    
    try:
        # Procesar los datos
        train_data, val_data = preprocessor.process_data(
            input_path="/home/dsg/vortex/PRODUCTION/DATA/raw/input_data.xlsx",  # Ajusta esta ruta
            output_train_path="/home/dsg/vortex/PRODUCTION/DATA/processed/training_data.xlsx",
            output_val_path="/home/dsg/vortex/PRODUCTION/DATA/processed/final_validation_set.xlsx"
        )
        
        # Imprimir información sobre los datos procesados
        print("\nResumen del procesamiento:")
        print(f"Datos de entrenamiento: {train_data.shape}")
        print(f"Datos de validación: {val_data.shape}")
        print("\nDistribución de clases en entrenamiento:")
        print(train_data['Site'].value_counts())
        
    except Exception as e:
        logging.error(f"Error durante el procesamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()