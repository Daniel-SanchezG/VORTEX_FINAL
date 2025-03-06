# Proyecto VORTEX - Sistema de Clasificación Arqueológica

Este repositorio contiene un sistema completo para el procesamiento, entrenamiento y análisis de modelos de clasificación en datos arqueológicos de XRF (Fluorescencia de Rayos X). El sistema permite entrenar modelos generales y específicos para diferentes conjuntos de características y analizar la importancia de las variables.

## Estructura del Proyecto

```
VORTEX_FINAL/
├── PRODUCTION/
│   ├── src/
│   │   ├── preprocessing/
│   │   │   └── data_processor.py
│   │   ├── training/
│   │   │   ├── model_trainer.py
│   │   │   ├── specific_models_trainer.py
│   │   │   └── specific_models_trainer_pycaret.py
│   │   └── analysis/
│   │       └── feature_importance_analyzer.py
│   ├── archaeological_predictor.py
│   ├── run_predictions.py
│   └── main.py
├── DATA/
│   └── real_world/
│       └── real_world_data.xlsx
├── models/
├── outputs/
└── mlruns/
```

## Componentes Principales

### 1. Entrenamiento y Evaluación (`main.py`)

El script principal que coordina el flujo de trabajo completo de procesamiento de datos, entrenamiento de modelos, análisis de importancia de características y entrenamiento de modelos específicos.

**Características principales:**
- Preprocesamiento de datos XRF
- Entrenamiento del modelo general
- Análisis opcional de importancia de características
- Entrenamiento de modelos específicos para diferentes conjuntos de características

**Uso:**
```bash
python main.py --input /ruta/a/datos.xlsx --output-dir outputs [--full] [--min-class-size 10] [--validation-split 0.1]
```

**Argumentos:**
- `--input`: Ruta al archivo de datos (Excel o CSV)
- `--output-dir`: Directorio base para guardar resultados (predeterminado: 'outputs')
- `--min-class-size`: Tamaño mínimo de clase para mantener (predeterminado: 10)
- `--validation-split`: Fracción de datos para validación (predeterminado: 0.1)
- `--full`: Flag para ejecutar el análisis completo incluyendo importancia de características (paso 3)

### 2. Entrenadores de Modelos

#### a. Entrenador General (`model_trainer.py`)

Utiliza PyCaret para entrenar un modelo Random Forest general con optimización de hiperparámetros y calibración de probabilidades.

#### b. Entrenador de Modelos Específicos

**Versión scikit-learn** (`specific_models_trainer.py`):
- Implementación original basada en scikit-learn
- Entrena modelos específicos para diferentes conjuntos de características
- Puede presentar problemas de compatibilidad con `archaeological_predictor.py`

**Versión PyCaret** (`specific_models_trainer_pycaret.py`):
- Reimplementación usando PyCaret para consistencia
- Mantiene compatibilidad completa con `archaeological_predictor.py`
- Genera los mismos análisis y evaluaciones que la versión original

### 3. Analizador de Importancia de Características (`feature_importance_analyzer.py`)

Realiza análisis detallados para identificar las características más importantes en la clasificación:
- Análisis RFECV (Recursive Feature Elimination with Cross-Validation)
- Visualización de importancia de características
- Análisis SHAP (SHapley Additive exPlanations)

### 4. Predictor Arqueológico (`archaeological_predictor.py`)

Sistema para cargar datos arqueológicos del mundo real, seleccionar modelos apropiados y generar predicciones consolidadas.

**Características principales:**
- Carga datos de múltiples sitios arqueológicos
- Selecciona automáticamente el modelo apropiado para cada sitio
- Genera predicciones con scores de probabilidad para cada clase
- Consolida resultados en un único archivo

**Uso:**
```bash
python archaeological_predictor.py --data /ruta/a/datos.xlsx --models /ruta/a/modelos [--output /ruta/salida.xlsx]
```

## Solución para Problemas de Compatibilidad

El sistema originalmente presentaba problemas de compatibilidad entre:
- Modelos entrenados con PyCaret en `model_trainer.py`
- Modelos entrenados directamente con scikit-learn en `specific_models_trainer.py`
- El sistema de predicción en `archaeological_predictor.py`

La solución implementada fue:

1. Crear `specific_models_trainer_pycaret.py` que:
   - Mantiene la misma funcionalidad que `specific_models_trainer.py`
   - Utiliza PyCaret para entrenar y guardar modelos
   - Garantiza compatibilidad con `archaeological_predictor.py`

2. Utilizar el formato de modelo de PyCaret para todos los modelos, lo que asegura:
   - Consistencia en la carga de modelos
   - Manejo adecuado de preprocesamiento de datos
   - Compatibilidad con la función `predict_model` de PyCaret

## MLflow y Tracking de Experimentos

El sistema utiliza MLflow (a través de PyCaret) para el seguimiento de experimentos. Los artefactos de MLflow se guardan en:
- El directorio `mlruns/` (por defecto)

## Requisitos

- Python 3.8+
- pandas
- numpy
- scikit-learn
- PyCaret
- matplotlib
- seaborn
- shap
- MLflow (utilizado internamente por PyCaret)

## Instalación

```bash
pip install pandas numpy scikit-learn pycaret matplotlib seaborn shap mlflow
```

## Notas Importantes

1. **Compatibilidad de Modelos**: Asegúrese de utilizar `specific_models_trainer_pycaret.py` para entrenar modelos específicos que sean compatibles con `archaeological_predictor.py`.

2. **Directorios de MLflow**: PyCaret crea automáticamente directorios `mlruns/` para el seguimiento de experimentos. Para cambiar la ubicación, puede configurar la variable de entorno `MLFLOW_TRACKING_URI` o modificar el código para establecer `mlflow.set_tracking_uri()`.

3. **Análisis de Importancia de Características**: Este análisis puede ser computacionalmente intensivo. Utilice el flag `--full` en `main.py` solo cuando necesite realizar este análisis.