## XRF Data Analysis Pipeline

This repository contains a complete pipeline for processing and analyzing XRF (X-Ray Fluorescence) data, including data preprocessing, model training, and evaluation.

## Features

- Automated data preprocessing pipeline
- Model training with Random Forest classifier
- Probability calibration for uncertainty estimation
- Comprehensive evaluation metrics and visualizations
- Timestamped experiments and logging
- Reproducible results with fixed random seeds

## Project Structure

```
project/
│
├── data/
│   ├── raw/                  # Raw input data
│   └── processed/            # Processed datasets
│
├── src/
│   ├── preprocessing/        # Data preprocessing modules
│   │   ├── __init__.py
│   │   └── data_processor.py
│   │
│   ├── training/            # Model training modules
│   │   ├── __init__.py
│   │   └── model_trainer.py
│   │
│   └── __init__.py
│
├── outputs/                 # Experiment outputs
│   └── experiment_YYYYMMDD_HHMMSS/
│       ├── data/           # Processed data
│       ├── models/         # Saved models
│       ├── plots/          # Generated visualizations
│       ├── tables/         # Metrics and results
│       └── logs/           # Experiment logs
│
├── main.py                 # Main script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline with default parameters:
```bash
python main.py --input "path/to/your/data.xlsx"
```

### Available Arguments:

- `--input`: Path to input file (Excel or CSV) [Required]
- `--output-dir`: Directory where results will be saved [Default: "outputs"]
- `--min-class-size`: Minimum class size to keep [Default: 10]
- `--validation-split`: Fraction of data for validation [Default: 0.1]

### Example with All Parameters:

```bash
python main.py \
    --input "data/raw/my_data.xlsx" \
    --output-dir "my_experiments" \
    --min-class-size 15 \
    --validation-split 0.2
```

## Output Structure

Each experiment run creates a timestamped directory containing:

- `data/processed/`: Processed training and validation datasets
- `models/`: Trained and calibrated model files
- `plots/`: Confusion matrices and other visualizations
- `tables/`: Performance metrics and evaluation results
- `logs/`: Detailed execution logs
- `experiment_info.txt`: Experiment configuration and summary

## Dependencies

Main dependencies include:
- pycaret==3.2.0
- pandas>=1.5.0
- scikit-learn>=1.0.2
- imbalanced-learn>=0.10.1
- matplotlib>=3.5.0

For a complete list, see `requirements.txt`.

## License

[Your License]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/your-username/your-repository](https://github.com/your-username/your-repository)