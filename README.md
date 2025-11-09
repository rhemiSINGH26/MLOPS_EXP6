# ML Model CI/CD Pipeline

A simple CI/CD pipeline using GitHub Actions for training, testing, and deploying a machine learning model.

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions workflow
├── train_model.py              # Model training script
├── evaluate_model.py           # Model evaluation script
├── test_model.py               # Test suite
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Features

- **Simple ML Model**: Random Forest classifier trained on the Iris dataset
- **Automated Testing**: Unit tests using pytest
- **Model Training**: Automated training pipeline
- **Model Evaluation**: Performance metrics (accuracy, F1-score, precision, recall)
- **Artifact Storage**: Saves trained models and metrics as artifacts

## CI/CD Pipeline Steps

The GitHub Actions workflow performs the following steps:

1. **Install Dependencies**: Installs required Python packages
2. **Run Tests**: Executes unit tests using pytest
3. **Train Model**: Trains the Random Forest classifier
4. **Evaluate Model**: Generates performance metrics
5. **Save Artifacts**: Stores the trained model and metrics

## Local Setup

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Exp6
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Train the Model
```bash
python train_model.py
```

#### Evaluate the Model
```bash
python evaluate_model.py
```

#### Run Tests
```bash
pytest test_model.py -v
```

## GitHub Actions Workflow

The CI/CD pipeline is triggered on:
- Push to `main` branch
- Pull requests to `main` branch

### Viewing Artifacts

After the workflow runs successfully:
1. Go to the "Actions" tab in your GitHub repository
2. Click on the latest workflow run
3. Download artifacts:
   - `trained-model`: The trained model file (iris_model.pkl)
   - `model-metrics`: Performance metrics (metrics.json)

## Model Performance

The model is evaluated on the following metrics:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1 score
- **Precision**: Weighted precision
- **Recall**: Weighted recall

## Output Files

- `models/iris_model.pkl`: Trained model
- `models/test_data.pkl`: Test dataset
- `metrics/metrics.json`: Performance metrics

## Dependencies

- scikit-learn: Machine learning library
- pytest: Testing framework
- joblib: Model serialization

## License

MIT License
