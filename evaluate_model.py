"""
Model Evaluation Script
Evaluates the trained model and generates performance metrics
"""
import joblib
import json
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def evaluate_model():
    """Evaluate the trained model and save metrics"""
    print("Loading model and test data...")
    
    # Load the trained model
    model = joblib.load('models/iris_model.pkl')
    
    # Load test data
    X_test, y_test = joblib.load('models/test_data.pkl')
    
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    print("Calculating metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    # Print metrics
    print("\n=== Model Performance Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("=" * 33)
    
    # Print detailed classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))
    
    # Save metrics to JSON file
    os.makedirs('metrics', exist_ok=True)
    metrics_path = 'metrics/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    evaluate_model()
    print("\nEvaluation completed successfully!")
