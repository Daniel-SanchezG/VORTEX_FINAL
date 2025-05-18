@echo off
setlocal enabledelayedexpansion

:: Asegurarse de que el directorio de salida existe
if not exist outputs mkdir outputs
if not exist real_world_results mkdir real_world_results

:: Valores predeterminados
set INPUT=DATA/raw/input_data.xlsx
set OUTPUT_DIR=outputs
set MIN_CLASS_SIZE=10
set VALIDATION_SPLIT=0.1
set FULL=

:parse_args
if "%1"=="" goto run
if "%1"=="--input" (
  set INPUT=%2
  shift
  shift
  goto parse_args
)
if "%1"=="--output-dir" (
  set OUTPUT_DIR=%2
  shift
  shift
  goto parse_args
)
if "%1"=="--min-class-size" (
  set MIN_CLASS_SIZE=%2
  shift
  shift
  goto parse_args
)
if "%1"=="--validation-split" (
  set VALIDATION_SPLIT=%2
  shift
  shift
  goto parse_args
)
if "%1"=="--full" (
  set FULL=--full
  shift
  goto parse_args
)
if "%1"=="--help" (
  echo VORTEX Docker Helper
  echo Usage: docker_run.bat [OPTIONS]
  echo.
  echo Options:
  echo   --input FILE              Path to input file (Excel or CSV)
  echo   --output-dir DIR          Directory where results will be saved
  echo   --min-class-size N        Minimum class size to keep (default: 10)
  echo   --validation-split N      Fraction of data for validation (default: 0.1)
  echo   --full                    Run full analysis including feature importance
  echo   --help                    Show this help message
  exit /b 0
)
echo Unknown option: %1
exit /b 1

:run
:: Construir los argumentos para main.py
set ARGS=--input "%INPUT%" --output-dir "%OUTPUT_DIR%" --min-class-size %MIN_CLASS_SIZE% --validation-split %VALIDATION_SPLIT% %FULL%

:: Ejecutar Docker
echo Running VORTEX with: %ARGS%
docker run -v "%cd%\%OUTPUT_DIR%:/app/%OUTPUT_DIR%" ^
           -v "%cd%\DATA:/app/DATA" ^
           vortex %ARGS%