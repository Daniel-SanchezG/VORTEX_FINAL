
[![GitHub License](https://img.shields.io/github/license/Daniel-SanchezG/VORTEX_FINAL)](https://github.com/Daniel-SanchezG/VORTEX_FINAL/blob/main/LICENSE)



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15000069.svg)](https://doi.org/10.5281/zenodo.15000069)




**VORTEX** (Variscite ORigin Technology X-ray based) is a modular tool designed to source archaeological variscites samples using machine learning. The system makes predictions about the provenance of archaeological artifacts, evaluates the uncertainty of these predictions, determines provenance consensus by site, and visualizes the results.


Authors:

**Daniel Sanchez-Gomez , José Ángel Garrido-Cordero, José María Martínez-Blanes, Rodrigo Villalobos García, Manuel Edo i Benaigues, Ana Catarina Sousa, María Dolores Zambrana Vega, Rosa Barroso Bermejo, Primitiva Bueno Ramírez, Carlos P. Odriozola**

---
## Features

- Automated data preprocessing pipeline
- Model training with Random Forest classifier
- Probability calibration for uncertainty estimation
- Comprehensive evaluation metrics and visualizations
- Timestamped experiments and logging
- Reproducible results with fixed random seeds
- SHAP analysis for feature importance interpretation
- Real-world proof-of-concept
- Uncertainty quantification and visualization

## Project Structure

```
VORTEX_FINAL/
│
├── DATA/
│   ├── raw/                  # Raw input data
│   ├── processed/            # Processed datasets
│   └── real_world/           # Real-world archaeological data
│
├── src/
│   ├── preprocessing/        # Data preprocessing modules
│   │   ├── __init__.py
│   │   └── data_processor.py
│   │
│   ├── training/             # Model training modules
│   │   ├── __init__.py
│   │   ├── model_trainer.py
│   │   ├── specific_models_trainer.py
│   │   └── specific_models_trainer_pycaret.py
│   │
│   ├── analysis/             # Analysis modules
│   │   ├── __init__.py
│   │   ├── feature_importance_analyzer.py
│   │   └── shap_analyzer.py
│   │
│   ├── real_world/           # Real-world prediction modules
│   │   ├── __init__.py
│   │   ├── archaeological_predictor.py
│   │   ├── provenance_determination.py
│   │   ├── run_predictions.py
│   │   ├── run_predictions_with_uncertainty.py
│   │   ├── uncertainty_analysis.py
│   │   └── visualization.py
│   │
│   ├── utils/                # Utility functions
│   ├── visualization/        # Visualization modules
│   └── __init__.py
│
├── models/                   # Saved trained models
│   ├── final_model       # Main prediction model
│   ├── rf_Destilled      # Destilled model
│   ├── rf_French         # French-specific model
│   ├── rf_Quiruelas      # Quiruelas-specific model
│   └── rf_VdH            # VdH-specific model
│
├── development_notebooks/                # Jupyter development notebooks
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   ├── Feature_importance.ipynb
│   ├── SHAP_local.ipynb
│   ├── Uncertainty_analysis.ipynb
│   └── real_world_predictions.ipynb
│
├── outputs/                  # Experiment outputs
│   └── experiment_YYYYMMDD_HHMMSS/
│       ├── data/             # Processed data
│       ├── models/           # Saved models
│       ├── plots/            # Generated visualizations
│       ├── tables/           # Metrics and results
│       └── logs/             # Experiment logs
│
├── resultados_shap/          # SHAP analysis results
│   ├── logs/                 # SHAP analysis logs
│   ├── plots/                # SHAP visualizations
│   └── tables/               # SHAP importance values
│
├── real_world_results/       # Results from real-world analyses
├── main.py                   # Main training pipeline script
├── real_world.py             # Real-world prediction script
├── run_shap_analysis.py      # SHAP analysis script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```


## Prerequisites

Before installing VORTEX, you need to have Python (version 3.8 or higher), pip, and virtualenv installed on your system.

### Installing Python

#### Windows

1. Download the latest Python installer from [python.org](https://www.python.org/downloads/)
2. Run the installer and make sure to check "Add Python to PATH" during installation
3. Verify the installation by opening Command Prompt and typing:

    ```cmd
    python3 --version
    ```

#### Linux/macOS

Most Linux distributions and macOS come with Python pre-installed. To verify:

```bash
python3 --version
```

If Python is not installed or you need a newer version:

**For Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install python3 python3-pip
```

**For macOS (using Homebrew):**

```bash
brew install python
```

### Installing pip

pip is the package installer for Python and usually comes bundled with Python installations.

#### Windows

If pip wasn't installed with Python:

```cmd
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

#### Linux/macOS

**For Ubuntu/Debian:**

```bash
sudo apt install python3-pip
```

**For macOS:**

```bash
python3 -m ensurepip --upgrade
```

Verify pip installation:

```bash
pip --version
# or
pip3 --version
```

### Installing virtualenv

virtualenv is a tool to create isolated Python environments.

#### Windows

```cmd
pip3 install virtualenv
```

#### Linux/macOS

```bash
pip3 install virtualenv
```

Or using the system package manager on Ubuntu/Debian:

```bash
sudo apt install python3-venv
```

## Installation

### Linux/macOS

1. Clone the repository:

```bash
git clone https://github.com/Daniel-SanchezG/VORTEX_FINAL
cd VORTEX
```

2. Create and activate a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

3. Install dependencies:

```bash
pip3 install -r requirements.txt
```

### Windows

1. Clone the repository:

```cmd
git clone https://github.com/Daniel-SanchezG/VORTEX_FINAL
cd VORTEX
```

2. Create and activate a virtual environment:

```cmd
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

3. Install dependencies:

```cmd
pip3 install -r requirements_windows.txt
```

4. Windows-specific considerations:
    - If you encounter issues with packages that require C++ compilation (like scikit-learn):
        
        - Install Microsoft C++ Build Tools from the [Microsoft website](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
        - Or use pre-compiled wheels: `pip install --only-binary :all: -r requirements-windows.txt`
    - For path issues, use backslashes or raw strings when specifying paths in commands:

```cmd
   python3 main.py --input "DATA\raw\input_data.xlsx"   
```

Or simply use forward slashes, which generally work in Windows as well:

```cmd

python3 main.py --input "DATA/raw/input_data.xlsx"

```

- If you encounter PowerShell execution policy issues when activating the environment:

```powershell

 Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
 
```

Then try activating the environment again.

---
## Installation

### Linux/macOS

1. Clone the repository:
```bash
git clone https://github.com/Daniel-SanchezG/VORTEX_FINAL
cd VORTEX
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

3. Install dependencies:
```bash
pip3 install -r requirements.txt
```

### Windows

1. Clone the repository:
```cmd
git clone https://github.com/Daniel-SanchezG/VORTEX_FINAL
cd VORTEX
```

2. Create and activate a virtual environment:
```cmd
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

3. Install dependencies:
```cmd
pip3 install -r requirements.txt
```

4. Windows-specific considerations:
   - If you encounter issues with packages that require C++ compilation (like scikit-learn):
     - Install Microsoft C++ Build Tools from the [Microsoft website](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
     - Or use pre-compiled wheels: `pip install --only-binary :all: -r requirements.txt`
   
   - For path issues, use backslashes or raw strings when specifying paths in commands:
     ```cmd
     python3 main.py --input "DATA\raw\input_data.xlsx"
     ```
     Or simply use forward slashes, which generally work in Windows as well:
     ```cmd
     python3 main.py --input "DATA/raw/input_data.xlsx"
     ```

## Usage

### Training Pipeline

Run the model training pipeline with default parameters:
```bash
python3 main.py --input "DATA/raw/input_data.xlsx"
```

#### Available Arguments:

- `--input`: Path to input file (Excel or CSV) [Required]
- `--output-dir`: Directory where results will be saved [Default: "outputs"]
- `--min-class-size`: Minimum class size to keep [Default: 10]
- `--validation-split`: Fraction of data for validation [Default: 0.1]
- `--full`: Run full analysis including feature importance analysis [Flag]

#### Example with All Parameters:

**Linux/macOS:**
```bash
python3 main.py \
    --input "DATA/raw/input_data.xlsx" \
    --output-dir "my_experiments" \
    --min-class-size 15 \
    --validation-split 0.2 \
    --full
```

**Windows:**
```cmd
python3 main.py ^
    --input "DATA\raw\input_data.xlsx" ^
    --output-dir "my_experiments" ^
    --min-class-size 15 ^
    --validation-split 0.2 ^
    --full
```

The`--full` flag activates the feature importance analysis through RFECV, which identifies and ranks the most important features. The process can be computationally intensive depending on the available resources, so it is disabled by default.

### **Real-World Prediction**

To perform real_world proof-of-concept:
```bash
python real_world.py --data "DATA/real_world/real_world_data.xlsx" --models "models" --output "real_world_results"
```

#### Available Arguments:

- `--data`: Path to the Excel file with archaeological data [Required]
- `--models`: Directory containing trained models [Required]
- `--output`: Directory to save results [Required]
- `--threshold`: Confidence threshold for analysis [Default: 0.7]
- `--no-uncertainty`: Skip uncertainty analysis [Flag]
- `--no-provenance`: Skip provenance determination [Flag]
- `--no-visualization`: Skip visualization generation [Flag]

#### Example with All Parameters:

**Linux/macOS:**
```bash
python3 real_world.py \
    --data "DATA/real_world/real_world_data.xlsx" \
    --models "models" \
    --output "real_world_results" \
    --threshold 0.7
```

**Windows:**
```cmd
python3 real_world.py 
    --data "DATA\real_world\real_world_data.xlsx" 
    --models "models" 
    --output "real_world_results" 
    --threshold 0.7
```

The real-world analysis pipeline consists of four main steps:
1. **Prediction**: Generates predictions for archaeological samples
2. **Uncertainty Analysis**: Evaluates the uncertainty of the predictions
3. **Provenance Determination**: Determines consensus provenance by site
4. **Visualization**: Creates entropy visualizations for result interpretation

### **Local SHAP Analysis**

The local Shap analyses require a specific process of training binary classification models for each source. The result is a plot of the influence of the features for each of the three sources considered, which is presented in section 3 of the article.


To run Local SHAP analysis:
```bash
python3 run_shap_analysis.py --model "models/final_model.pkl" --data "DATA/processed/final_input_data.xlsx"
```

## Output Structure

### Training Pipeline Outputs

Each experiment run creates a timestamped directory (`experiment_YYYYMMDD_HHMMSS`) containing:

- `data/processed/`: Processed training and validation datasets
- `models/`: Trained and calibrated model files
  - `final_model.pkl`: Main model
  - `tuned_model.pkl`: Tuned version of the main model
  - `rf_Destilled.pkl`: Model trained on destilled features
  - `rf_French.pkl`: Model trained on French-specific features
  - `rf_Quiruelas.pkl`: Model trained on Quiruelas-specific features
  - `rf_VdH.pkl`: Model trained on VdH-specific features
- `plots/`: Confusion matrices and other visualizations
- `tables/`: Performance metrics and evaluation results
  - Classification reports
  - Model evaluation scores
  - Predictions on validation data
- `logs/`: Detailed execution logs
- `experiment_info.txt`: Experiment configuration and summary including:
  - Input file information
  - Training parameters
  - Model configuration
  - Top features by importance (if `--full` flag was used)
  - Performance metrics for each specific model

SHAP analysis outputs are stored in `resultados_shap/`:
- `plots/`: SHAP visualizations for each provenance source
- `tables/`: Feature importance values
- `logs/`: Analysis logs

### Real-World Analysis Outputs

Real-world prediction results are stored in the specified output directory and include:

- `archaeological_predictions_YYYYMMDD.xlsx`: Raw predictions for each sample
- `uncertainty_analysis_YYYYMMDD.xlsx`: Uncertainty metrics for each prediction
- `provenance_analysis_YYYYMMDD.xlsx`: Determined provenance by site with confidence levels
- `site_entropy_distribution_YYYYMMDD.png`: Visualization of entropy distribution
- `site_statistics_YYYYMMDD.xlsx`: Statistical summary of provenance by site
- `archaeological_pipeline_YYYYMMDD.log`: Detailed pipeline execution log

## Models

VORTEX includes several specialized models:

1. **Main models:**
   - `final_model.pkl`: Main model trained on all features
   - `tuned_model.pkl`: Optimized version of the main model
   -  `rf_Destilled.pkl`: Destilled model using only the most important features: 
     - Ca, S, K, Ti, V, Cr, Cu, Zn, As, Se, Sr, Mo, Ba, Ta

1. **Region-specific models:**
   - `rf_French.pkl`: Specialized for French sites using:
     - Al, Si, P, S, Cl, K, Ca, Ti, V, Cr, Fe, Co, Cu, Zn, As, Se, Rb, Sr, Zr, Mo, Ba, Ta
   - `rf_Quiruelas.pkl`: Specialized for Quiruelas samples using:
     - Al, Si, P, K, Ca, Ti, V, Cr, Mn, Fe, Co, Ga, Zn, As, Rb, Sr, Zr
   - `rf_VdH.pkl`: Specialized for VdH region using:
     - Al, Si, P, S, Cl, K, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, As, Rb, Sr, Zr

Each specialized model is trained on a specific subset of features that have shown to be most discriminative for particular geological sources.

## Dependencies

Main dependencies include:
- pycaret==3.3.2
- pandas>=1.5.0
- scikit-learn>=1.0.2
- imbalanced-learn>=0.10.1
- matplotlib>=3.5.0
- shap>=0.41.0

For a complete list, see `requirements.txt`.

=======
## Troubleshooting

### Common Issues

1. **Dependency errors:**
   - Ensure you have installed all dependencies: `pip install -r requirements.txt`
   - For Windows users with compilation errors, install Microsoft C++ Build Tools
   - Try using conda environment if pip installation fails: `conda env create -f environment.yml`

2. **File path issues:**
   - Windows: Use either backslashes (`\`) or forward slashes (`/`) consistently
   - Be careful with spaces in file paths; enclose paths in quotes
   - Use absolute paths if relative paths aren't working

3. **Memory errors:**
   - For large datasets, ensure your system has enough RAM
   - If using random forest on large data, adjust n_jobs parameter in the code to limit parallelization

4. **Model loading issues:**
   - Ensure the model file exists in the specified directory
   - Model files should be compatible with your scikit-learn version
   - If loading models between different OS platforms, be aware of potential compatibility issues

### Windows-Specific Issues

1. **Path length limitations:**
   - Windows has a 260-character path length limit
   - Use shorter paths or enable long paths in Windows 10/11:
     - Run `regedit` and navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
     - Set `LongPathsEnabled` to `1`

2. **Permission issues:**
   - Run Command Prompt or PowerShell as Administrator for writing to system directories
   - Check Windows Defender or antivirus if files are being blocked

3. **Anaconda/Conda environment:**
   ```cmd
   conda create -n vortex python=3.8
   conda activate vortex
   pip3 install -r requirements.txt
   ```

4. **Jupyter notebook integration on Windows:**
   
   - If using Jupyter notebooks:

     
   ```cmd
     conda install jupyter
     python3 -m ipykernel install --user --name=vortex --display-name="Python (VORTEX)"
   ```

   - Navigate to the `notebooks` directory and run: `jupyter notebook`

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Daniel SG - daniel-sanchez-gomez@edu.ulisboa.pt

ZENODO Link: [https://zenodo.org/records/15000069](https://zenodo.org/records/15000069)

