"""
Simple ML Model Training Script
Trains a classifier on the Iris dataset
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model():
    """Train a simple Random Forest classifier on the Iris dataset"""
    print("Loading dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = 'models/iris_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save test data for evaluation
    test_data_path = 'models/test_data.pkl'
    joblib.dump((X_test, y_test), test_data_path)
    print(f"Test data saved to {test_data_path}")
    
    return model

if __name__ == "__main__":
    train_model()
    print("Training completed successfully!")
