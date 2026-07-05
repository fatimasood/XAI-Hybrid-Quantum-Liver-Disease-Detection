#!/usr/bin/env python3
"""
Training script for Classical Baseline Model
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.metrics import MetricsCalculator
from models.classical_baseline import ClassicalBaseline
from plots.training_plots import TrainingPlotter
from utils.config import (
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    MODELS_DIR,
    N_QUBITS,
    N_QLAYERS,
    CLASSICAL_LAYERS
)


def train():
    """Main training pipeline"""

    print("CLASSICAL BASELINE MODEL TRAINING")

    try:
        # Load and prepare data
        print("\nLoading and preprocessing data...")
        data = DataLoader()
        data.load_and_preprocess()
        data.perform_eda()
        data.get_data_summary()

        # Cross-validation
        print("\nPerforming cross-validation...")
        kf = StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=42
        )

        cv_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            kf.split(data.X_train, data.y_train)
        ):

            print(f"\nFold {fold+1}/{kf.n_splits}")

            X_tr = data.X_train[train_idx]
            X_val = data.X_train[val_idx]

            y_tr = data.y_train[train_idx]
            y_val = data.y_train[val_idx]

            print("Building Classical Baseline...")
            model = ClassicalBaseline(input_dim=X_tr.shape[1])

            print(f"Training Fold {fold+1}...")

            history = model.model.fit(
                X_tr,
                y_tr,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(X_val, y_val),
                callbacks=model.get_callbacks(fold+1),
                verbose=0
            )

            y_val_pred_prob = model.model.predict(
                X_val,
                verbose=0
            ).flatten()

            y_val_pred = (y_val_pred_prob > 0.5).astype(int)

            fold_metrics = MetricsCalculator.calculate_all_metrics(
                y_val,
                y_val_pred,
                y_val_pred_prob
            )

            cv_metrics.append(fold_metrics)

            print(f"Accuracy : {fold_metrics['accuracy']:.4f}")
            print(f"ROC AUC  : {fold_metrics['roc_auc']:.4f}")
            print(f"F1 Score : {fold_metrics['f1_score']:.4f}")

        print("\nCross Validation Results")

        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "specificity"
        ]:

            values = [m[metric] for m in cv_metrics]

            print(
                f"{metric:15s}: "
                f"{np.mean(values):.4f} ± {np.std(values):.4f}"
            )

        print("\nTraining Final Classical Model...")

        final_model = ClassicalBaseline(
            input_dim=data.X_train.shape[1]
        )

        class_weight_dict = data.get_class_weights()

        history = final_model.model.fit(
            data.X_train,
            data.y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            callbacks=final_model.get_callbacks(),
            class_weight=class_weight_dict,
            verbose=1
        )

        # Save model
        final_model.save(os.path.join(MODELS_DIR, 'classical_final_model'))
        print(f"\nwithout quantization Model saved to {MODELS_DIR}")

        print("\nModel Saved Successfully!")

        print("\nGenerating Training Plots...")

        plotter = TrainingPlotter(history)

        plotter.plot_loss_accuracy()
        plotter.plot_metrics()
        plotter.create_training_dashboard()

        print("\nModel Summary")

        final_model.summary()

        print("\nTraining Completed Successfully")
        print(f"Classical Layers : {CLASSICAL_LAYERS}")

        return final_model, history, data

    except Exception as e:

        print(f"\nError : {e}")

        print("\nFull Traceback")

        traceback.print_exc()

        return None, None, None


if __name__ == "__main__":

    model, history, data = train()

    if model is not None:
        print("\nTraining Successful!")