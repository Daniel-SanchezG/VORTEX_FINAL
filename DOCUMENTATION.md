
## Direct installation

### Prerequisites

Before installing VORTEX, you need to have Python (version 3.10), pip, and virtualenv installed on your system.

---
### Installing Python

#### Windows

1. Download the latest Python installer from [python.org](https://www.python.org/downloads/)    
2. Run the installer and make sure to check "Add Python to PATH" during installation
3. Verify the installation by opening Command Prompt and typing:
4. 
```cmd
  python --version  
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

---

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

---

### Installing virtualenv

virtualenv is a tool to create isolated Python environments.

#### Windows

```cmd
pip install virtualenv
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

##  VORTEX Installation

You can download the ZIP file from [DOI 10.5281/zenodo.15000068](https://zenodo.org/records/15162972) or from the project's Github [repository](https://github.com/Daniel-SanchezG/VORTEX_FINAL) (look for the green button that says 'Code') and select 'Download ZIP'.

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
python -m venv venv
```

```cmd
# Activate virtual environment
venv\Scripts\activate
```
 
3. Install dependencies: 

```cmd
# Install remaining dependencies
pip install -r requirements.txt
```

5. Windows-specific considerations:

    - If you encounter issues with packages that require C++ compilation (like scikit-learn or SHAP):
        
        - download and install Microsoft Visual C++ Redistributables: 
			- [Visual C++ 2015-2022 x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
			- [Visual C++ 2015-2022 x86](https://aka.ms/vs/17/release/vc_redist.x86.exe)

		**Important:** Restart your computer after installing these redistributables
		
        - Install Microsoft C++ Build Tools from the [Microsoft website](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
        - Or use pre-compiled wheels: `pip install --only-binary :all: -r requirements.txt`
    - For path issues, use backslashes or raw strings when specifying paths in commands:
        

```cmd
   python main.py --input "DATA\raw\input_data.xlsx"   
```

Or simply use forward slashes, which generally work in Windows as well:

```cmd
python main.py --input "DATA/raw/input_data.xlsx"
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

**Linux/macOS:**

```bash
python3 main.py --input "DATA/raw/input_data.xlsx"
```

**Windows:**

```cmd
python main.py --input "DATA/raw/input_data.xlsx"
```

#### Available Arguments:

- `--input`: Path to input file (Excel or CSV) (**Required**)
- `--output-dir`: Directory where results will be saved (**Default: "outputs"**)
- `--min-class-size`: Minimum class size to keep (**Default: 10**)
- `--validation-split`: Fraction of data for validation (**Default: 0.1**)
- `--full`: Run full analysis including feature importance analysis (**Flag**)

#### Example with All Parameters:

The`--full` flag activates the feature importance analysis through RFECV, which identifies and ranks the most important features. The process can be computationally intensive depending on the available resources, so it is disabled by default.

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
python main.py --input "DATA/raw/input_data.xlsx" --output-dir "outputs" --min-class-size 10 --validation-split 0.2 --full
```


---

### **Real-World Analysis**

The real-world analysis uses the trained model to predict the geological origin of n=571 artefacts from 15 archaeological sites. This is a proof of concept of the framework with real-world data presented in the article.

The analysis  consists of four main steps:

1. **Prediction**: Generates predictions for archaeological samples
2. **Uncertainty Analysis**: Evaluates the uncertainty of the predictions
3. **Provenance Determination**: Determines provenance per site by majority vote
4. **Visualization**: Creates plot for uncertainty interpretation


**Linux/macOS:**

```bash
python3 real_world.py --data "DATA/real_world/real_world_data.xlsx" --models "models" --output "real_world_results"
```

**Windows:**

```cmd
python real_world.py --data "DATA/real_world/real_world_data.xlsx" --models "models" --output "real_world_results"
```

#### Available Arguments:

- `--data`: Path to the Excel file with archaeological data (**Required**)
- `--models`: Directory containing trained models (**Required**)
- `--output`: Directory to save results (**Required**)
- `--threshold`: Confidence threshold for analysis (**Default: 0.7**)
- `--no-uncertainty`: Skip uncertainty analysis (**Flag**)
- `--no-provenance`: Skip provenance determination (**Flag**)
- `--no-visualization`: Skip visualization generation (**Flag**)

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
python real_world.py --data "DATA/real_world/real_world_data.xlsx" --models "models" --output "real_world_results" --threshold 0.7
```


---

### **Local SHAP Analysis**

Local Shap analyses visualise the influence of each characteristic within each of the green phosphate deposits considered in the study. These are executed by an independent module called run_shap_analysis.py. To run the analysis:

**Linux/macOS:**

```bash
python3 run_shap_analysis.py --input "DATA/processed/training_data.xlsx" --sheets aliste can_tintorer encinasola
```

**Windows (Command Prompt or PowerShell):**

```cmd
python run_shap_analysis.py --input "DATA/processed/training_data.xlsx" --sheets aliste can_tintorer encinasola
```

---

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

SHAP analysis outputs are stored in `shap_results/`:

- `plots/`: SHAP visualizations for each provenance source
- `tables/`: Feature importance values
- `logs/`: Analysis logs

### Real-World Analysis Outputs

Real-world prediction results are stored in the real_world_results directory and include:

- `archaeological_predictions_YYYYMMDD.xlsx`: Raw predictions for each sample
- `uncertainty_analysis_YYYYMMDD.xlsx`: Uncertainty metrics for each prediction
- `provenance_analysis_YYYYMMDD.xlsx`: Determined provenance by site with confidence levels
- `site_entropy_distribution_YYYYMMDD.png`: Visualization of uncertainty assessment
- `site_statistics_YYYYMMDD.xlsx`: Statistical summary of provenance by site
- `archaeological_pipeline_YYYYMMDD.log`: Detailed pipeline execution log

## Models

VORTEX includes several specialized models:

1. **Main models:**
    
    - `final_model.pkl`: Main model trained on all features
    - `rf_Destilled.pkl`: Destilled model using only the most important features:
	    - Ca, S, K, Ti, V, Cr, Cu, Zn, As, Se, Sr, Mo, Ba, Ta

2. **Region-specific models:**
    
    - `rf_French.pkl`: Specialized for French sites using:
        - Al, Si, P, S, Cl, K, Ca, Ti, V, Cr, Fe, Co, Cu, Zn, As, Se, Rb, Sr, Zr, Mo, Ba, Ta
    - `rf_Quiruelas.pkl`: Specialized for Quiruelas samples using:
        - Al, Si, P, K, Ca, Ti, V, Cr, Mn, Fe, Co, Ga, Zn, As, Rb, Sr, Zr
    - `rf_VdH.pkl`: Specialized for Valle de las Higueras site using:
        - Al, Si, P, S, Cl, K, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, As, Rb, Sr, Zr

**Each specialized model is trained on a specific subset of features that have shown to be most discriminative for particular geological sources**.

## Dependencies

Main dependencies include:

- pycaret==3.3.2
- pandas>=1.5.0
- scikit-learn>=1.0.2
- imbalanced-learn>=0.10.1
- matplotlib>=3.5.0
- shap>=0.44.0
- llvmlite==0.41.1 (Windows)
- numba==0.58.1 (Windows)

For a complete list, see `requirements.txt` (Linux/macOS) or `requirements_windows.txt` (Windows).

## Troubleshooting

### Common Issues

1. **Dependency errors:**
    
    - Ensure you have installed all dependencies: `pip install -r requirements.txt`
    - For Windows users with compilation errors, install Microsoft C++ Build Tools

2. **File path issues:**
    
    - Windows: Use either backslashes (`\`) or forward slashes (`/`) consistently
    - Be careful with spaces in file paths; enclose paths in quotes
    - Use absolute paths if relative paths aren't working

### **Windows-Specific Issues**

1. **DLL Initialization Failed / llvmlite errors:**
    
    **Symptoms:** Error message containing `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed` when importing SHAP or numba.
    
    **Solution:**
    
    - Install Visual C++ Redistributables (both x64 and x86 versions) from the links provided in the installation section
    - Restart your computer after installation
    - Reinstall the problematic packages with specific versions:
    
```cmd
pip uninstall llvmlite numba shap -y pip install llvmlite==0.41.1 pip install numba==0.58.1 pip install shap==0.44.0
```

2. **Permission issues:**
    
    - Run Command Prompt or PowerShell as Administrator for writing to system directories
    - Check Windows Defender or antivirus if files are being blocked

3. **Anaconda/Conda environment (Alternative method):**
    
    If you continue having issues with pip installation, try using conda:
    
    ```cmd
    conda create -n vortex python=3.10
    conda activate vortex
    conda install -c conda-forge numba shap
    pip install -r requirements.txt
    ```
    
4. **Jupyter notebook integration on Windows:**
    
    If using Jupyter notebooks:
    
    ```cmd
    conda install jupyter
    python -m ipykernel install --user --name=vortex --display-name="Python (VORTEX)"
    ```
    
    Navigate to the `notebooks` directory and run: `jupyter notebook`