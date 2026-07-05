#!/usr/bin/env python3
"""
Evaluation script for Classical Baseline Model
"""

import os
import numpy as np
import tensorflow as tf
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)
from utils.data_loader import DataLoader
from utils.metrics import MetricsCalculator
from plots.classical_evaluation_plots import EvaluationPlotter
from models.classical_baseline import ClassicalBaseline
from utils.config import MODELS_DIR


def evaluate():
    """Main evaluation pipeline"""

    print("CLASSICAL BASELINE MODEL EVALUATION")

    # Load data
    print("\nLoading test data...")

    data = DataLoader()
    data.load_and_preprocess()

    # Load trained model
    print("\nLoading trained model...")

    model_path = os.path.join(
        MODELS_DIR, 'classical_final_model.h5'
    )

    if os.path.exists(model_path):

        loaded_model = tf.keras.models.load_model(model_path)

        model = ClassicalBaseline(
            input_dim=data.X_train.shape[1]
        )

        model.model = loaded_model

    else:

        print("Saved model not found.")
        print("Training a new Classical model...")

        from baseline_train import train

        model, _, _ = train()

    # Prediction
    print("\nEvaluating on Test Set...")

    y_test_pred_prob = model.model.predict(
        data.X_test,
        verbose=0
    ).flatten()

    y_test_pred = (
        y_test_pred_prob > 0.5
    ).astype(int)

    # Metrics
    metrics = MetricsCalculator.calculate_all_metrics(
        data.y_test,
        y_test_pred,
        y_test_pred_prob
    )

    ci = MetricsCalculator.calculate_confidence_intervals(
        data.y_test,
        y_test_pred_prob
    )

    print("\n========== TEST RESULTS ==========\n")

    print(f"{'Accuracy':20s}: {metrics['accuracy']:.4f}")
    print(f"{'Precision':20s}: {metrics['precision']:.4f}")
    print(f"{'Recall':20s}: {metrics['recall']:.4f}")
    print(f"{'Specificity':20s}: {metrics['specificity']:.4f}")
    print(f"{'F1 Score':20s}: {metrics['f1_score']:.4f}")
    print(f"{'Balanced Accuracy':20s}: {metrics['balanced_accuracy']:.4f}")

    print(
        f"{'ROC AUC':20s}: "
        f"{metrics['roc_auc']:.4f} "
        f"[{ci['roc_ci'][0]:.4f}-{ci['roc_ci'][1]:.4f}]"
    )

    print(
        f"{'PR AUC':20s}: "
        f"{metrics['pr_auc']:.4f} "
        f"[{ci['pr_ci'][0]:.4f}-{ci['pr_ci'][1]:.4f}]"
    )

    print(f"{'MCC':20s}: {metrics['mcc']:.4f}")
    print(f"{'Cohen Kappa':20s}: {metrics['kappa']:.4f}")
    print(f"{'Brier Score':20s}: {metrics['brier_score']:.4f}")
    print(f"{'Youden Index':20s}: {metrics['youden_index']:.4f}")

    # Generate plots
    print("\nGenerating Evaluation Plots...")

    plotter = EvaluationPlotter(
        data.y_test,
        y_test_pred,
        y_test_pred_prob
    )

    plotter.plot_confusion_matrix()

    plotter.plot_roc_curve(
        ci_bounds=[
            ci["roc_ci"][0],
            ci["roc_ci"][1]
        ]
    )

    plotter.plot_pr_curve()

    plotter.plot_calibration_curve()

    plotter.plot_metrics_radar({

        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Specificity": metrics["specificity"],
        "F1 Score": metrics["f1_score"],
        "ROC AUC": metrics["roc_auc"]

    })

    plotter.plot_prediction_distribution()

    print("\nEvaluation Completed Successfully.")
    print("All plots saved in plots directory.")

    return metrics


if __name__ == "__main__":

    metrics = evaluate()