"""
Test suite for the ML model
"""
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

def test_dataset_loads():
    """Test that the Iris dataset loads successfully"""
    iris = load_iris()
    assert iris is not None
    assert iris.data.shape[0] > 0
    assert iris.data.shape[1] == 4

def test_model_creation():
    """Test that a Random Forest model can be created"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')

def test_model_training():
    """Test that the model can be trained"""
    iris = load_iris()
    X, y = iris.data[:100], iris.target[:100]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Test prediction
    test_input = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(test_input)
    assert prediction is not None
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]
