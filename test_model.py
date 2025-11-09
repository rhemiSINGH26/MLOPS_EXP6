"""
Test suite for the ML model
"""
import pytest
from train_model import train_model
import os
import joblib

def test_train_model():
    """Test that the model trains successfully"""
    model = train_model()
    assert model is not None
    assert hasattr(model, 'predict')

def test_model_file_exists():
    """Test that the model file is created"""
    # Train the model first
    train_model()
    assert os.path.exists('models/iris_model.pkl')

def test_model_predictions():
    """Test that the model can make predictions"""
    model = train_model()
    # Simple test data (4 features for iris dataset)
    test_input = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(test_input)
    assert prediction is not None
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]  # Valid iris classes
