#!/bin/bash

# Asegurarse de que el directorio de salida existe
mkdir -p outputs
mkdir -p real_world_results

# Valores predeterminados
INPUT="DATA/raw/input_data.xlsx"
OUTPUT_DIR="outputs"
MIN_CLASS_SIZE=10
VALIDATION_SPLIT=0.1
FULL=false

# Procesar argumentos
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --min-class-size)
      MIN_CLASS_SIZE="$2"
      shift 2
      ;;
    --validation-split)
      VALIDATION_SPLIT="$2"
      shift 2
      ;;
    --full)
      FULL=true
      shift
      ;;
    --help)
      echo "VORTEX Docker Helper"
      echo "Usage: ./docker_run.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --input FILE              Path to input file (Excel or CSV)"
      echo "  --output-dir DIR          Directory where results will be saved"
      echo "  --min-class-size N        Minimum class size to keep (default: 10)"
      echo "  --validation-split N      Fraction of data for validation (default: 0.1)"
      echo "  --full                    Run full analysis including feature importance"
      echo "  --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Construir los argumentos para main.py
ARGS="--input \"$INPUT\" --output-dir \"$OUTPUT_DIR\" --min-class-size $MIN_CLASS_SIZE --validation-split $VALIDATION_SPLIT"
if [ "$FULL" = true ]; then
  ARGS="$ARGS --full"
fi

# Ejecutar Docker
echo "Running VORTEX with: $ARGS"
docker run -v "$(pwd)/$OUTPUT_DIR:/app/$OUTPUT_DIR" \
           -v "$(pwd)/DATA:/app/DATA" \
           vortex $ARGS