# Sistema de Análisis Arqueológico VORTEX

## Descripción

El Sistema de Análisis Arqueológico VORTEX es una herramienta modular diseñada para analizar datos arqueológicos mediante aprendizaje automático. El sistema realiza predicciones sobre la procedencia de artefactos arqueológicos, evalúa la incertidumbre de estas predicciones, determina consensos de procedencia por sitio y visualiza los resultados.

## Estructura del Sistema

El sistema está compuesto por cuatro módulos principales, cada uno responsable de una fase específica del análisis:

1. **Predicción Arqueológica** (`archaeological_predictor.py`): Carga datos de diferentes sitios arqueológicos, aplica modelos de aprendizaje automático entrenados previamente y genera predicciones sobre la procedencia de los artefactos.

2. **Análisis de Incertidumbre** (`uncertainty_analysis.py`): Evalúa la confiabilidad de las predicciones, calcula métricas de incertidumbre (entropía) y marca predicciones como "inciertas" cuando no alcanzan un umbral de confianza predefinido.

3. **Determinación de Procedencia** (`provenance_determination.py`): Analiza las predicciones de alta confianza para determinar consensos de procedencia por sitio, calculando estadísticas como homogeneidad y proporción de muestras inciertas.

4. **Visualización** (`visualization.py`): Genera representaciones gráficas de la distribución de probabilidades y entropía por sitio, junto con estadísticas detalladas.

Todo esto se integra en un script principal (`run_predictions_with_uncertainty_and_provenance.py`) que ejecuta el pipeline completo.

## Requisitos

- Python 3.8+
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- pycaret

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/vortex-archaeological-analysis.git
cd vortex-archaeological-analysis

# Instalar dependencias
pip install -r requirements.txt
```

## Estructura de Directorios

```
/VORTEX_FINAL/PRODUCTION/
│
├── DATA/
│   └── real_world/
│       └── real_world_data.xlsx  # Datos arqueológicos por sitio
│
├── models/
│   ├── final_model               # Modelo general
│   ├── 20250227_VdHSpecific      # Modelo específico para V_Higueras
│   ├── 20250227_QuiruelasSpecific # Modelo específico para Quiruelas
│   └── 20250227_FrenchSpecific   # Modelo específico para sitios franceses
│
├── results/                      # Directorio para resultados
│
└── src/
    └── real_world/
        ├── archaeological_predictor.py
        ├── uncertainty_analysis.py
        ├── provenance_determination.py
        ├── visualization.py
        └── run_predictions_with_uncertainty_and_provenance.py
```

## Uso

### Ejecución Completa

Para ejecutar el pipeline completo desde la línea de comandos:

```bash
python run_predictions_with_uncertainty_and_provenance.py
```

### Opciones Disponibles

```bash
python run_predictions_with_uncertainty_and_provenance.py --help
```

Parámetros:
- `--data`: Ruta al archivo Excel con datos arqueológicos
- `--models`: Directorio que contiene los modelos entrenados
- `--output`: Directorio para guardar resultados
- `--threshold`: Umbral de confianza (predeterminado: 0.7)
- `--no-uncertainty`: Omitir análisis de incertidumbre
- `--no-provenance`: Omitir determinación de procedencia
- `--no-visualization`: Omitir generación de visualizaciones

### Ejemplo con Parámetros Personalizados

```bash
python run_predictions_with_uncertainty_and_provenance.py --threshold 0.8 --output /ruta/personalizada/resultados/
```

## Resultados

El sistema genera varios archivos de resultados:

1. **Predicciones** (`archaeological_predictions_YYYYMMDD.xlsx`): Predicciones crudas para cada muestra con puntuaciones de probabilidad.

2. **Análisis de Incertidumbre** (`uncertainty_analysis_YYYYMMDD.xlsx`): Predicciones con métricas de incertidumbre, incluyendo:
   - Probabilidades por clase (CT, PCM, PDLC)
   - Predicciones originales
   - Nivel de confianza
   - Predicciones con umbral de incertidumbre
   - Entropía (en bits)

3. **Determinación de Procedencia** (`provenance_analysis_YYYYMMDD.xlsx`): Análisis a nivel de sitio, incluyendo:
   - Conteo de muestras por categoría
   - Porcentaje de incertidumbre
   - Muestras utilizadas para determinar procedencia
   - Entropía mediana
   - Procedencia por consenso
   - Homogeneidad del consenso

4. **Visualización** (`site_entropy_distribution_YYYYMMDD.png`): Gráfico que muestra:
   - Distribución de probabilidades medianas por sitio
   - Entropía mediana por sitio

5. **Estadísticas Detalladas** (`site_statistics_YYYYMMDD.xlsx`): Métricas adicionales por sitio.

## Casos de Uso

El sistema se puede utilizar para:

1. **Análisis de Procedencia**: Determinar el origen más probable de artefactos arqueológicos.
2. **Evaluación de Incertidumbre**: Identificar muestras y sitios con predicciones de baja confianza.
3. **Análisis de Homogeneidad**: Evaluar la consistencia de las procedencias en cada sitio.
4. **Visualización de Patrones**: Identificar visualmente sitios con alta incertidumbre o distribuciones de probabilidad particulares.

## Características Adicionales

- **Manejo de IDs**: El sistema preserva rigurosamente los IDs originales de las muestras para asegurar la trazabilidad.
- **Mapeo de Modelos**: Selecciona automáticamente el modelo más apropiado para cada sitio arqueológico.
- **Análisis Robusto**: Utiliza estadísticas medianas para resistir valores atípicos.
- **Configurabilidad**: Permite ajustar umbrales de confianza y otras opciones según necesidades específicas.

## Contribuir

Si deseas contribuir a este proyecto:

1. Haz un fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Haz push a la rama (`git push origin nueva-funcionalidad`)
5. Crea un Pull Request

## Licencia

Este proyecto está licenciado bajo [LICENCIA] - consulta el archivo LICENSE para más detalles.

## Contacto

[Tu Nombre] - [tu.email@ejemplo.com]

Enlace del proyecto: [https://github.com/tu-usuario/vortex-archaeological-analysis](https://github.com/tu-usuario/vortex-archaeological-analysis)
