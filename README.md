# Patient Readmission Prediction

A comprehensive machine learning project for predicting patient readmission risk within 30 days of discharge using structured clinical data.

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS A RESEARCH DEMONSTRATION TOOL ONLY**

- **NOT FOR CLINICAL USE**: This tool is designed for research and educational purposes only
- **NOT MEDICAL ADVICE**: Never use this tool for actual clinical decision-making
- **NO DIAGNOSTIC VALUE**: The predictions are not validated for clinical use
- **CONSULT HEALTHCARE PROFESSIONALS**: Always seek professional medical advice for healthcare decisions

## Overview

This project implements a complete machine learning pipeline for patient readmission prediction, including:

- **Data Processing**: Synthetic EHR data generation and preprocessing
- **Multiple Models**: Traditional ML (XGBoost, LightGBM, Random Forest) and deep learning approaches
- **Clinical Metrics**: Comprehensive evaluation with clinically meaningful metrics
- **Explainability**: SHAP-based model interpretation
- **Interactive Demo**: Streamlit web application for model exploration

## Features

### Models Supported
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Traditional ML**: Random Forest, Logistic Regression, SVM
- **Deep Learning**: TabularNet, Attention-based networks

### Clinical Metrics
- **Discrimination**: AUC-ROC, AUC-PR, Sensitivity, Specificity
- **Calibration**: Brier Score, Expected Calibration Error
- **Clinical Utility**: PPV, NPV, Likelihood Ratios

### Explainability
- **SHAP Values**: Feature importance and individual predictions
- **Permutation Importance**: Alternative feature ranking
- **Visualization**: Summary plots, waterfall plots, feature importance

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Patient-Readmission-Prediction
cd Patient-Readmission-Prediction

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### Basic Usage

```bash
# Train a model with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/custom.yaml

# Train with your own data
python scripts/train.py --data-path path/to/your/data.csv
```

### Run Interactive Demo

```bash
# Start Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data processing utilities
│   ├── models/            # Model implementations
│   ├── metrics/           # Evaluation metrics
│   ├── train/             # Training utilities
│   └── utils/             # Core utilities
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks
├── assets/                # Generated plots and outputs
└── data/                  # Data directory
    ├── raw/              # Raw data files
    ├── processed/        # Processed data files
    └── external/         # External data sources
```

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
# Example configuration
model:
  type: "xgboost"
  xgboost:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 6

training:
  epochs: 100
  batch_size: 32
  patience: 10

evaluation:
  threshold: 0.5
  save_plots: true
```

## Data Schema

The synthetic dataset includes the following features:

### Demographics
- `age`: Patient age (18-100)
- `gender`: Male/Female
- `race`: Ethnicity categories

### Clinical Features
- `length_of_stay`: Hospital stay duration (days)
- `num_lab_procedures`: Number of lab tests
- `num_medications`: Number of medications
- `num_procedures`: Number of procedures
- `num_diagnoses`: Number of diagnoses

### Administrative
- `admission_type`: Emergency/Urgent/Elective/Newborn
- `discharge_disposition`: Home/SNF/Rehab/AMA/Expired
- `admission_source`: Physician/ER/Transfer/Other
- `diag_primary`: Primary diagnosis category

### Target
- `readmitted`: Binary readmission indicator (0/1)

## Model Performance

Example results on synthetic data:

| Model | AUC-ROC | Sensitivity | Specificity | F1-Score |
|-------|---------|-------------|-------------|----------|
| XGBoost | 0.742 | 0.681 | 0.703 | 0.692 |
| LightGBM | 0.738 | 0.675 | 0.698 | 0.686 |
| Random Forest | 0.721 | 0.658 | 0.684 | 0.671 |
| Logistic Regression | 0.698 | 0.642 | 0.654 | 0.648 |

## Evaluation Metrics

### Discrimination Metrics
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate

### Calibration Metrics
- **Brier Score**: Mean squared error of probabilities
- **ECE**: Expected Calibration Error

### Clinical Utility
- **PPV**: Positive Predictive Value
- **NPV**: Negative Predictive Value
- **LR+**: Positive Likelihood Ratio
- **LR-**: Negative Likelihood Ratio

## Explainability

The project provides comprehensive model interpretability:

### SHAP Analysis
- Feature importance ranking
- Individual prediction explanations
- Summary plots for global patterns

### Permutation Importance
- Alternative feature ranking method
- Model-agnostic approach

## Development

### Code Quality
- **Type Hints**: Full type annotation coverage
- **Documentation**: NumPy/Google-style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff static analysis
- **Testing**: Pytest unit tests

### Reproducibility
- **Deterministic Seeding**: Random seeds for all libraries
- **Device Fallback**: CUDA → MPS → CPU
- **Configuration Management**: OmegaConf for configs
- **Experiment Tracking**: Structured logging

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{patient_readmission_prediction,
  title={Patient Readmission Prediction: A Comprehensive ML Pipeline},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Patient-Readmission-Prediction}
}
```

## Acknowledgments

- Synthetic data generation inspired by healthcare ML research
- Model architectures based on tabular deep learning literature
- Clinical metrics following healthcare ML best practices
- SHAP explainability for model interpretability

---

**Remember**: This is a research demonstration tool. Always consult healthcare professionals for medical decisions.
# Patient-Readmission-Prediction
