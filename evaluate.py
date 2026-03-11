#!/usr/bin/env python3
"""
Evaluation script for Hybrid Quantum-Classical Model
"""

import numpy as np
import tensorflow as tf
import os

from utils.data_loader import DataLoader
from utils.metrics import MetricsCalculator
from models.hybrid_qnn import HybridQNN
from plots.evaluation_plots import EvaluationPlotter
from utils.config import MODELS_DIR

def evaluate():
    """Main evaluation pipeline"""
  
    print("HYBRID QUANTUM-CLASSICAL MODEL EVALUATION")
    
    # Load data
    print("\nLoading test data...")
    data = DataLoader()
    data.load_and_preprocess().split_data().scale_features()
    
    # Load model
    print("\nLoading trained model...")
    model_path = os.path.join(MODELS_DIR, 'final_model')
    
    if os.path.exists(model_path):
        loaded_model = tf.keras.models.load_model(model_path)
        # Wrap in HybridQNN class for compatibility
        model = HybridQNN(input_dim=data.X_train.shape[1])
        model.model = loaded_model
    else:
        print(f"Model not found at {model_path}. Training a new model...")
        from train import train
        model, _, _ = train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred_prob = model.model.predict(data.X_test).flatten()
    y_test_pred = (y_test_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    metrics = MetricsCalculator.calculate_all_metrics(
        data.y_test, y_test_pred, y_test_pred_prob
    )
    
    # Calculate confidence intervals
    ci = MetricsCalculator.calculate_confidence_intervals(data.y_test, y_test_pred_prob)
    
    print("\n.... Test Set Results ....")
    print(f"{'Metric':20s} {'Value':10s} {'95% CI':20s}")
  
    print(f"{'Accuracy':20s} {metrics['accuracy']:.4f}")
    print(f"{'Precision':20s} {metrics['precision']:.4f}")
    print(f"{'Recall':20s} {metrics['recall']:.4f}")
    print(f"{'Specificity':20s} {metrics['specificity']:.4f}")
    print(f"{'F1-score':20s} {metrics['f1_score']:.4f}")
    print(f"{'Balanced Accuracy':20s} {metrics['balanced_accuracy']:.4f}")
    print(f"{'ROC AUC':20s} {metrics['roc_auc']:.4f} [{ci['roc_ci'][0]:.4f}-{ci['roc_ci'][1]:.4f}]")
    print(f"{'PR AUC':20s} {metrics['pr_auc']:.4f} [{ci['pr_ci'][0]:.4f}-{ci['pr_ci'][1]:.4f}]")
    print(f"{'MCC':20s} {metrics['mcc']:.4f}")
    print(f"{'Cohens Kappa':20s} {metrics['kappa']:.4f}")
    print(f"{'Brier Score':20s} {metrics['brier_score']:.4f}")
    print(f"{'Youden Index':20s} {metrics['youden_index']:.4f}")
    
    # Generate evaluation plots
    print("\nGenerating evaluation plots...")
    plotter = EvaluationPlotter(data.y_test, y_test_pred, y_test_pred_prob)
    
    plotter.plot_confusion_matrix()
    plotter.plot_roc_curve(ci_bounds=[ci['roc_ci'][0], ci['roc_ci'][1]])
    plotter.plot_pr_curve()
    plotter.plot_calibration_curve()
    plotter.plot_metrics_radar({k: v for k, v in metrics.items()  if k in ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']})
    plotter.plot_prediction_distribution()
    
    print("Evaluation completed! All plots saved to plots or directory")
    
    return metrics

if __name__ == "__main__":
    metrics = evaluate()