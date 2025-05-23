
[![GitHub License](https://img.shields.io/github/license/Daniel-SanchezG/VORTEX_FINAL)](https://github.com/Daniel-SanchezG/VORTEX_FINAL/blob/main/LICENSE)



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15000069.svg)](https://doi.org/10.5281/zenodo.15000069)




**VORTEX** (Variscite ORigin Technology X-ray based) is a modular tool designed to source archaeological variscites samples using machine learning. The system makes predictions about the provenance of archaeological artifacts, evaluates the uncertainty of these predictions, determines provenance consensus by site, and visualizes the results.


Authors:

**Daniel Sanchez-Gomez , José Ángel Garrido-Cordero, José María Martínez-Blanes, Rodrigo Villalobos García, Manuel Edo i Benaigues, Ana Catarina Sousa, María Dolores Zambrana Vega, Rosa Barroso Bermejo, Primitiva Bueno Ramírez, Carlos P. Odriozola**


![image](https://github.com/Daniel-SanchezG/VORTEX_FINAL/blob/main/development_notebooks/20250313Model_schema-P%C3%A1gina-3.drawio.png)

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


## Project Structure

```
VORTEX/
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
│       ├── __init__.py
│       ├── archaeological_predictor.py
│       ├── provenance_determination.py
│       ├── run_predictions.py
│       ├── run_predictions_with_uncertainty.py
│       ├── uncertainty_analysis.py
│       └── visualization.py
│
├── models/                # Saved trained models
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
│── Dockerfile                #  Docker configuration
├── .dockerignore             #  Docker ignore file
│
├── real_world_results/       # Results from real-world analyses
├── main.py                   # Main training pipeline script
├── real_world.py             # Real-world prediction script
├── run_shap_analysis.py      # SHAP analysis script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Installation and Setup

There are two ways to run VORTEX:

1. **Using Docker (Recommended for reproducibility)**: A simple approach that works on any system without worrying about dependencies
2. **Direct installation**: Using Python virtual environments

### Choose Your Setup Method:

- **If you choose Docker**: You can avoid the Python installation, virtual environment setup and dependency installation steps. Just follow the Docker Instructions section and run the pipeline.
  
- **If you prefer direct installation**: Skip Docker instructions and junp to the complete installation process below.

---

## 1. Docker Instructions (Recommended for Reproducibility)

### Installing Docker

#### Windows
1. Download Docker Desktop for Windows from [docker.com](https://www.docker.com/products/docker-desktop)
2. Run the installer and follow the instructions

#### macOS
1. Download Docker Desktop for Mac from [docker.com](https://www.docker.com/products/docker-desktop)
2. Drag and drop Docker.app to your Applications folder
3. Open Docker from your Applications folder
4. When you see the Docker whale in the menu bar, the installation is complete

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
sudo docker run hello-world
```
---

### Installing Git (optional)

Git is useful for cloning the repository directly from the command line. Here's how to install it:

#### Windows

1. Download the official Git installer from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer with default options (you can customize if needed)
3. Verify the installation by opening Command Prompt or PowerShell and typing:
    
    ```cmd
    git --version
    ```

#### Linux/macOS

**For Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install git
```

**For macOS (using Homebrew):**

```bash
brew install git
```
---

### Installing and running VORTEX with Docker:

- You can download the ZIP file from [DOI 10.5281/zenodo.15000068](https://zenodo.org/records/15162972)  or  from the project's Github repository (look for the green button that says ‘Code’)  and select ‘Download ZIP’.

- Extract it to a folder of your choice and navigate to the project folder.

Alternatively you can clone the GitHub repository (Git must be installed):

#### Windows

1. Clone the repository and navigate to the project folder

Using Git Bash or Command Prompt:

```bash
git clone https://github.com/Daniel-SanchezG/VORTEX_FINAL.git
cd VORTEX_FINAL
```

#### Linux/macOS


```bash
git clone https://github.com/Daniel-SanchezG/VORTEX_FINAL.git
cd VORTEX_FINAL
```
---
Once in the project folder

#### Build the Docker image:

```bash
docker build -t vortex . 
```

### Run the training pipeline:

The`--full` flag activates the feature importance analysis through RFECV, which identifies and ranks the most important features. The process can be computationally intensive depending on the available resources, so it is disabled by default.

Windows

```cmd
#Basic usage
docker run -v "%cd%\outputs:/app/outputs" -v "%cd%\DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs" 

#Full process
docker run -v "%cd%\outputs:/app/outputs" -v "%cd%\DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs" --full
```

or if you are using PowerShell:

```powershell
docker run -v "${PWD}\outputs:/app/outputs" -v "${PWD}\DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs" --full
```


Linux/macOS
```bash
# Basic usage
docker run -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs"

# Full analysis
docker run -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs" --full
```


### Run real-world analysis:

Real-world analysis uses the trained model to predict the geological source of n=571 artefacts from 15 archaeological sites. It is a proof of concept with real data whose result is presented in section 3.3 of the article.

**Windows**

Command Prompt (CMD)

```cmd
docker run -v "%cd%\real_world_results:/app/real_world_results" -v "%cd%\DATA:/app/DATA" -v "%cd%\models:/app/models" vortex python3 real_world.py --data "DATA/real_world/real_world_data.xlsx" --models "models" --output "real_world_results"
```
PowerShell

```powershell
docker run -v "${PWD}\real_world_results:/app/real_world_results" -v "${PWD}\DATA:/app/DATA" -v "${PWD}\models:/app/models" vortex python3 real_world.py --data "DATA/real_world/real_world_data.xlsx" --models "models" --output "real_world_results"
```
**Linux/macOS**

```bash
docker run -v "$(pwd)/real_world_results:/app/real_world_results" -v "$(pwd)/DATA:/app/DATA" -v "$(pwd)/models:/app/models" vortex python3 real_world.py --data "DATA/real_world/real_world_data.xlsx" --models "models" --output "real_world_results"
```
### Find the results

Once the programme has finished (this could take several minutes), the results will be available:

- Open the VORTEX folder
- Find a new folder called "outputs"
- Inside you will find another folder with date and time (for example "experiment_AAAAMMDD_HHMMSS")

This folder contains the trianing results organised into:

- **plots**: Graphs and visualisations (figures 3 and 4 in the paper) 

- **tables**: Data in Excel tables (presented as a summary table 1 in the paper)

- **models**: Trained models

- **logs**: Detailed records of the process

For Real-world analysis:
- Find a new folder called "real_world_results"
- Inside you will find all the tables and visualisations generated (especially table 3 and figure 6 presented in section 3.3 of the paper).

`The results are the same as those presented in the paper`





## 2. Direct installation

### Prerequisites

Before installing VORTEX, you need to have Python (version 3.10), pip, and virtualenv installed on your system.


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

---

## Installation

You can download the ZIP file from [DOI 10.5281/zenodo.15000068](https://zenodo.org/records/15162972) and extract it to a folder of your choice.

Alternatively you can clone the GitHub repository:

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
    --output-dir "outputs" \
    --min-class-size 10 \
    --validation-split 0.2 \
    --full
```

**Windows:**
```cmd
python3 main.py ^
    --input "DATA\raw\input_data.xlsx" ^
    --output-dir "outputs" ^
    --min-class-size 10 ^
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

