# Guía de Integración del Análisis de Procedencia

He implementado una solución minimalista para integrar el análisis de determinación de procedencia en tu sistema. Esta implementación:

1. Toma los resultados del análisis de incertidumbre
2. Determina la procedencia por consenso para cada sitio 
3. Genera estadísticas detalladas por sitio arqueológico
4. Se integra perfectamente con el pipeline existente

## Estructura de Archivos

La solución consta de tres archivos principales:

1. **provenance_determination.py** - Nuevo módulo para determinación de procedencia
2. **run_predictions_with_uncertainty_and_provenance.py** - Script unificado que ejecuta todo el pipeline
3. Tu módulo existente **uncertainty_analysis.py**

## Instalación

1. Coloca los archivos en el mismo directorio:
   - `/home/dsg/VORTEX_FINAL/PRODUCTION/src/real_world/provenance_determination.py`
   - `/home/dsg/VORTEX_FINAL/PRODUCTION/src/real_world/run_predictions_with_uncertainty_and_provenance.py`

2. No es necesario instalar dependencias adicionales, el código utiliza las mismas bibliotecas que los módulos anteriores.

## Uso del Sistema

### Ejecución Completa

Para ejecutar el pipeline completo desde la línea de comandos:

```bash
python run_predictions_with_uncertainty_and_provenance.py
```

Esto ejecutará automáticamente los tres pasos en secuencia:
1. Predicciones arqueológicas
2. Análisis de incertidumbre
3. Determinación de procedencia

### Opciones Disponibles

```bash
python run_predictions_with_uncertainty_and_provenance.py --help
```

Parámetros:
- `--data` - Ruta al archivo Excel con datos
- `--models` - Directorio con modelos
- `--output` - Directorio para resultados
- `--threshold` - Umbral de confianza (predeterminado: 0.7)
- `--no-uncertainty` - Omitir análisis de incertidumbre
- `--no-provenance` - Omitir determinación de procedencia

### Ejemplo con Parámetros

```bash
python run_predictions_with_uncertainty_and_provenance.py --threshold 0.8 --output /home/dsg/VORTEX_FINAL/PRODUCTION/results/proyecto_especial/
```

### Módulo Individual

También puedes usar el módulo de determinación de procedencia de forma independiente:

```bash
python provenance_determination.py /ruta/a/resultados_incertidumbre.xlsx
```

## Resultados del Análisis de Procedencia

El módulo genera un DataFrame de resumen que contiene para cada sitio arqueológico:

- **Site**: Nombre del sitio arqueológico
- **Samples_analyzed**: Número total de muestras analizadas
- **Gavá, Encinasola, Aliste**: Conteo de muestras clasificadas en cada categoría
- **Uncertain(%)**: Porcentaje de predicciones inciertas
- **Samples_for_provenance**: Número de muestras usadas para determinar procedencia
- **Median_entropy**: Mediana de la entropía (incertidumbre)
- **Consensus**: Procedencia consensuada (categoría más frecuente)
- **Homogeneity**: Consistencia del consenso (0-1)

Estos resultados se guardan automáticamente en formatos Excel y CSV.

## Detalles de Implementación

### Enfoque de Consenso

La implementación:
- Identifica muestras de alta confianza (probabilidad > umbral)
- Determina la clase más frecuente como el consenso
- Calcula la homogeneidad (proporción de muestras que concuerdan con el consenso)
- Maneja casos donde no hay muestras de alta confianza

### Integración con el Pipeline

El script principal:
- Ejecuta los tres análisis en secuencia
- Pasa automáticamente los resultados entre módulos
- Mantiene cada etapa independiente pero integrada
- Permite omitir etapas según sea necesario
- Proporciona mensajes detallados de progreso

## Solución de Problemas

Si encuentras errores:

1. Verifica que el análisis de incertidumbre se haya ejecutado correctamente
2. Asegúrate de que los nombres de columnas coincidan con los esperados
3. Verifica que hay al menos un sitio con datos suficientes
4. Revisa los logs para mensajes de error específicos

## Personalización

El módulo se puede personalizar fácilmente:
- Ajustando el umbral de confianza según tus necesidades
- Modificando la definición de "consenso" si se requiere otro enfoque
- Añadiendo o modificando estadísticas en el DataFrame de resultados
