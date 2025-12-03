# Customer Churn Prediction

A machine learning pipeline for predicting customer churn using ensemble methods. This project implements XGBoost, LightGBM, and Decision Tree classifiers with automated hyperparameter tuning, class imbalance handling via SMOTE, and a voting ensemble for robust predictions.

## Project Overview

This system trains multiple classification models to predict whether customers will churn based on their usage patterns, contract details, and service subscriptions. The final model is a soft-voting ensemble that combines predictions from three base classifiers.

### Current Performance

Based on the latest training run:

- **Accuracy**: 76.2%
- **Precision**: 53.7%
- **Recall**: 75.7%
- **F1-Score**: 62.8%
- **ROC-AUC**: 84.1%

## Features

- **Data Preprocessing**: Automated feature engineering including tenure grouping, average monthly spend calculation, and log transformations
- **Class Imbalance Handling**: SMOTE oversampling integrated into training pipelines
- **Multiple Models**: XGBoost, LightGBM, and Decision Tree with GridSearchCV optimization
- **Ensemble Method**: Soft-voting classifier with optimized threshold for controlling false positive rate
- **CI/CD Integration**: GitHub Actions workflow for automated model training and metrics reporting
- **Safe Model Persistence**: Models saved in skops format for secure serialization

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yaml              # CI/CD pipeline configuration
├── config/
│   └── best_params.json         # Saved hyperparameters for quick training
├── data/
│   └── dataset.csv              # Raw customer data
├── models/
│   ├── model.skops              # Voting ensemble (preferred model)
│   ├── model_xgb.skops          # XGBoost individual model
│   ├── model_lgbm.skops         # LightGBM individual model
│   └── model_dt.skops           # Decision Tree individual model
├── results/
│   ├── metrics.json             # Latest evaluation metrics
│   └── voting_threshold.json   # Optimal classification threshold
├── src/
│   ├── __init__.py
│   ├── load_data.py             # Data loading and preprocessing
│   ├── train.py                 # Model training and ensemble building
│   └── test.py                  # Model evaluation utilities
├── tests/
│   └── test_train.py            # Integration tests
├── main.py                       # CLI entry point
├── Makefile                      # Build automation
└── requirements.txt              # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the Makefile
make install
```

## Usage

### Command Line Interface

The project provides a unified CLI through `main.py`:

```bash
# Train all models (uses saved hyperparameters for speed)
python main.py train

# Train with full hyperparameter search (slower)
python main.py train --quick=false

# Evaluate a saved model
python main.py evaluate --model models/model.skops

# Run test suite
python main.py pytest
```

### Python API

```python
from src.load_data import DataProcessor
from src.train import ChurnTrainer
from src.test import ModelEvaluator

# Load and preprocess data
processor = DataProcessor('data/dataset.csv')
df = processor.preprocess()

# Train models
trainer = ChurnTrainer(df=df, random_state=42)
trainer.run_all(
    train_xgb=True,
    train_lgbm=True,
    train_dt=True,
    build_voting=True,
    quick=True  # Use saved params
)

# Evaluate
evaluator = ModelEvaluator('models/model.skops')
evaluator.load_model()
metrics = evaluator.evaluate(X_test, y_test)
```

## Training Pipeline

### Data Preprocessing

The `DataProcessor` class handles:

1. Loading raw CSV data
2. Converting `TotalCharges` to numeric format
3. Creating derived features:
   - `tenure_group`: Categorical bins for customer tenure
   - `avg_monthly_spend`: TotalCharges / tenure
   - `contract_value`: MonthlyCharges × tenure
   - `low_charge`: Binary flag for MonthlyCharges < 30
4. Log transformations on monetary features

### Model Training

The `ChurnTrainer` class implements:

1. **Train/test split**: 80/20 stratified split
2. **Preprocessing pipeline**: 
   - Numeric features: Imputation + StandardScaler
   - Categorical features: Imputation + OneHotEncoder
3. **SMOTE oversampling**: Applied during training to handle class imbalance
4. **GridSearchCV**: 5-fold stratified cross-validation with multiple scoring metrics
5. **Model persistence**: Individual models and voting ensemble saved in skops format

### Quick Mode

For faster iterations, the project supports quick mode training:

```python
trainer.run_all(quick=True)
```

This loads pre-optimized hyperparameters from `config/best_params.json` and skips grid search, fitting models directly with known-good parameters.

## Models

### Individual Classifiers

1. **XGBoost**: Gradient boosting with regularization
   - Hyperparameters: n_estimators, max_depth, learning_rate, subsample, etc.
   
2. **LightGBM**: Efficient gradient boosting
   - Hyperparameters: Similar to XGBoost with LightGBM-specific tuning
   
3. **Decision Tree**: Interpretable baseline
   - Hyperparameters: max_depth, min_samples_split, class_weight

### Voting Ensemble (Recommended)

The voting classifier combines all three models using soft voting (averaged probabilities). An optimal classification threshold is computed to maintain a maximum false positive rate of 10%.

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yaml`) automatically:

1. Runs on every push to `main`
2. Installs dependencies and CML (Continuous Machine Learning)
3. Generates a model metrics report
4. Posts results as a comment on commits/PRs

### Metrics Report Format

```
## CI Run Report
Accuracy: 0.7622427253
Precision: 0.5370018975
Recall: 0.756684492
F1-Score: 0.628190899
ROC-AUC: 0.841008551
*New model artifact created and tested.*
```

## Model Evaluation

The `ModelEvaluator` class provides:

- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix visualization
- Support for skops model format

```python
evaluator = ModelEvaluator('models/model.skops')
evaluator.load_model()
metrics = evaluator.evaluate(X_test, y_test, plot_cm=True)
print(metrics)
```

## Configuration

### Hyperparameters

Best hyperparameters are stored in `config/best_params.json` after each full training run. These can be edited manually or regenerated by running training without quick mode.

### Model Threshold

The voting classifier's optimal threshold is saved in `results/voting_threshold.json`. This threshold is tuned to control the false positive rate while maximizing recall.

## Testing

Run the integration test suite:

```bash
# Full test
python tests/test_train.py

# Quick sanity check
python tests/test_train.py --quick

# Via pytest
pytest tests/ -v
```

## Dependencies

Key libraries:

- **scikit-learn**: Core ML algorithms and preprocessing
- **xgboost**: Gradient boosting
- **lightgbm**: Efficient gradient boosting
- **imbalanced-learn**: SMOTE implementation
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **skops**: Safe model serialization
- **matplotlib/seaborn**: Visualization

See `requirements.txt` for complete dependency list.

## Limitations

- Models assume stationary data distribution; performance may degrade if customer behavior patterns change significantly
- SMOTE is applied during training only; production predictions use the natural class distribution
- Hyperparameter grids are manually defined and may not be exhaustive
- The optimal threshold is computed on test data; consider recalibrating on held-out validation data for production use

## Future Improvements

- [ ] Add feature importance analysis and SHAP values
- [ ] Implement automated model retraining on new data
- [ ] Add prediction API endpoint (FastAPI/Flask)
- [ ] Expand hyperparameter search spaces
- [ ] Add time-based validation splits for temporal data
- [ ] Implement model drift detection
