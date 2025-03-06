# VORTEX Archaeological Analysis System

## Description

The VORTEX Archaeological Analysis System is a modular tool designed to analyze archaeological data using machine learning. The system makes predictions about the provenance of archaeological artifacts, evaluates the uncertainty of these predictions, determines provenance consensus by site, and visualizes the results.

## System Structure

The system consists of four main modules, each responsible for a specific phase of the analysis:

1. **Archaeological Prediction** (`archaeological_predictor.py`): Loads data from different archaeological sites, applies pre-trained machine learning models, and generates predictions about artifact provenance.

2. **Uncertainty Analysis** (`uncertainty_analysis.py`): Evaluates the reliability of predictions, calculates uncertainty metrics (entropy), and marks predictions as "uncertain" when they do not reach a predefined confidence threshold.

3. **Provenance Determination** (`provenance_determination.py`): Analyzes high-confidence predictions to determine provenance consensus by site, calculating statistics such as homogeneity and proportion of uncertain samples.

4. **Visualization** (`visualization.py`): Generates graphical representations of probability distribution and entropy by site, along with detailed statistics.

All of this is integrated in a main script (`run_predictions_with_uncertainty_and_provenance.py`) that executes the complete pipeline.

## Requirements

- Python 3.8+
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- pycaret

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/vortex-archaeological-analysis.git
cd vortex-archaeological-analysis

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
/VORTEX_FINAL/PRODUCTION/
│
├── DATA/
│   └── real_world/
│       └── real_world_data.xlsx  # Archaeological data by site
│
├── models/
│   ├── final_model               # General model
│   ├── 20250227_VdHSpecific      # Specific model for V_Higueras
│   ├── 20250227_QuiruelasSpecific # Specific model for Quiruelas
│   └── 20250227_FrenchSpecific   # Specific model for French sites
│
├── results/                      # Directory for results
│
└── src/
    └── real_world/
        ├── archaeological_predictor.py
        ├── uncertainty_analysis.py
        ├── provenance_determination.py
        ├── visualization.py
        └── run_predictions_with_uncertainty_and_provenance.py
```

## Usage

### Complete Execution

To run the complete pipeline from the command line:

```bash
python run_predictions_with_uncertainty_and_provenance.py
```

### Available Options

```bash
python run_predictions_with_uncertainty_and_provenance.py --help
```

Parameters:
- `--data`: Path to Excel file with archaeological data
- `--models`: Directory containing trained models
- `--output`: Directory to save results
- `--threshold`: Confidence threshold (default: 0.7)
- `--no-uncertainty`: Skip uncertainty analysis
- `--no-provenance`: Skip provenance determination
- `--no-visualization`: Skip visualization generation

### Example with Custom Parameters

```bash
python run_predictions_with_uncertainty_and_provenance.py --threshold 0.8 --output /custom/path/results/
```

## Results

The system generates several result files:

1. **Predictions** (`archaeological_predictions_YYYYMMDD.xlsx`): Raw predictions for each sample with probability scores.

2. **Uncertainty Analysis** (`uncertainty_analysis_YYYYMMDD.xlsx`): Predictions with uncertainty metrics, including:
   - Probabilities by class (CT, PCM, PDLC)
   - Original predictions
   - Confidence level
   - Predictions with uncertainty threshold
   - Entropy (in bits)

3. **Provenance Determination** (`provenance_analysis_YYYYMMDD.xlsx`): Site-level analysis, including:
   - Sample count by category
   - Uncertainty percentage
   - Samples used to determine provenance
   - Median entropy
   - Consensus provenance
   - Consensus homogeneity

4. **Visualization** (`site_entropy_distribution_YYYYMMDD.png`): Graph showing:
   - Median probability distribution by site
   - Median entropy by site

5. **Detailed Statistics** (`site_statistics_YYYYMMDD.xlsx`): Additional metrics by site.

## Use Cases

The system can be used for:

1. **Provenance Analysis**: Determine the most likely origin of archaeological artifacts.
2. **Uncertainty Evaluation**: Identify samples and sites with low-confidence predictions.
3. **Homogeneity Analysis**: Evaluate the consistency of provenances at each site.
4. **Pattern Visualization**: Visually identify sites with high uncertainty or particular probability distributions.

## Additional Features

- **ID Handling**: The system rigorously preserves the original sample IDs to ensure traceability.
- **Model Mapping**: Automatically selects the most appropriate model for each archaeological site.
- **Robust Analysis**: Uses median statistics to resist outliers.
- **Configurability**: Allows adjustment of confidence thresholds and other options according to specific needs.

## Contributing

If you wish to contribute to this project:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin new-feature`)
5. Create a Pull Request

## License

This project is licensed under [LICENSE] - see the LICENSE file for details.

## Contact

[Your Name] - [your.email@example.com]

Project Link: [https://github.com/your-username/vortex-archaeological-analysis](https://github.com/your-username/vortex-archaeological-analysis)
