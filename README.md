
[![GitHub License](https://img.shields.io/github/license/Daniel-SanchezG/VORTEX_FINAL)](https://github.com/Daniel-SanchezG/VORTEX_FINAL/blob/main/LICENSE)



[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15000069.svg)](https://doi.org/10.5281/zenodo.15000069)




## **VORTEX** (Variscite ORigin Technology X-ray based) is a framework designed to source archaeological variscites . The machine learning-based model makes predictions about the provenance of archaeological artifacts, address uncertainty using Feature Importance and Shapley values, determines provenance by site through majority voting  and visualises the results.

The framework is based in the following study and data:

- **Paper (preprint)**: [Read the paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5214878)
- **Data**: [Dataset](https://www.sciencedirect.com/science/article/pii/S2352340925006857) (Datapaper) 

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
├── shap_results/          # SHAP analysis results
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
2. **Direct installation**: Using Python virtual environments. follow [this Documentation](https://github.com/Daniel-SanchezG/VORTEX_FINAL/blob/main/DOCUMENTATION.md). 


---

## 1. Docker Instructions (Recommended)

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

### Installing and running VORTEX:

- You can download the ZIP file from [DOI 10.5281/zenodo.15000068](https://zenodo.org/records/15162972)  or  from the project's Github [repository](https://github.com/Daniel-SanchezG/VORTEX_FINAL) (look for the green button that says ‘Code’)  and select ‘Download ZIP’.

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

When running the training, the`--full` flag activates feature importance analysis through Recursive Feature Elimination (RFECV), which identifies and ranks the most important features and displays them in two plots. The process can be computationally intensive depending on the available resources.

#### Windows

```cmd
#Basic usage
docker run -v "%cd%\outputs:/app/outputs" -v "%cd%\DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs"
```

```cmd
#Full analysis
docker run -v "%cd%\outputs:/app/outputs" -v "%cd%\DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs" --full
```

or if you are using PowerShell:

```powershell

#Basic usage
docker run -v "${PWD}\outputs:/app/outputs" -v "${PWD}\DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs" 
```

```powershell
# Full analysis
docker run -v "${PWD}\outputs:/app/outputs" -v "${PWD}\DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs" --full
```


#### Linux/macOS

```bash
# Basic usage
docker run -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs"
```

```bash
# Full analysis
docker run -v "$(pwd)/outputs:/app/outputs" -v "$(pwd)/DATA:/app/DATA" vortex python3 main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs" --full
```

---

### Run real-world analysis:

The real-world analysis uses the trained model in the previous step to predict the geological origin of n=571 artefacts from 15 archaeological sites. This is a proof of concept of the framework with real-world data presented in the article.

The analysis  consists of four main steps:
1. **Prediction**: Generates predictions for archaeological samples
2. **Uncertainty Analysis**: Evaluates the uncertainty of the predictions
3. **Provenance Determination**: Determines provenance per site by majority vote
4. **Visualization**: Creates plot for uncertainty interpretation


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

---
### **Local SHAP Analysis**

Local Shap analyses visualise the influence of each characteristic within each of the green phosphate deposits considered in the study. These are executed by an independent module called`run_shap_analysis.py` . To run the analysis, simply copy and paste according to your system:

**Windows**

Command Prompt (CMD)
```bash
docker run -v "%cd%\shap_results:/app/shap_results" -v "%cd%\DATA:/app/DATA" vortex python3 run_shap_analysis.py --input "DATA/processed/training_data.xlsx" --sheets aliste can_tintorer encinasola 
```

PowerShell
```bash
 docker run -v "${PWD}\shap_results:/app/shap_results" -v "${PWD}\DATA:/app/DATA" vortex python3 run_shap_analysis.py --input "DATA/processed/training_data.xlsx" --sheets aliste can_tintorer encinasola
```

**Linux/macOS**


```bash
docker run -v "$(pwd)/shap_results:/app/shap_results" -v "$(pwd)/DATA:/app/DATA" vortex python3 run_shap_analysis.py --input "DATA/processed/training_data.xlsx" --sheets aliste can_tintorer encinasola
```

---
### **Find the results**

Once the programme has finished (this could take several minutes), the results will be available:

- Open the project folder
- Find a new folder called "outputs"
- Inside you will find a new directory with date and time ( "experiment_AAAAMMDD_HHMMSS")

This directory contains the model training results organised into:

- **plots**: Graphs and visualisations 
- **tables**: Data in Excel tables 
- **models**: Trained models
- **logs**: Detailed records of the experiment

**Real-world outputs are by default, located in a new folder called "real_world_results". It contains:**

- `archaeological_predictions_YYYYMMDD.xlsx`: Raw predictions for each sample
- `uncertainty_analysis_YYYYMMDD.xlsx`: Uncertainty metrics for each prediction
- `provenance_analysis_YYYYMMDD.xlsx`: Provenance per site with confidence levels
- `site_entropy_distribution_YYYYMMDD.png`: Visualization of uncertainty assessment
- `site_statistics_YYYYMMDD.xlsx`: Statistical summary of provenance by site
- `archaeological_pipeline_YYYYMMDD.log`: Detailed pipeline execution log


**Local SHAP Analysis results are located in a new folder called "shap_results" in the root of the project**

` When running real_world_analysis results should be the same as those presented in the paper`

---
### R Analysis for Summed Probability Distribution (SPD)

### Requirements

- R and RStudio installed
- RCarbon package

Go to the SPD folder and run the SPD script using RStudio. Before running the script, you will need to install the RCarbon package.

```bash
install.packages("rcarbon")
```

Once executed, you will find the *image SPD_sites_final.png* in the same folder.

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

