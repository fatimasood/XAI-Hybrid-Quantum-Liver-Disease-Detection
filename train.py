#!/usr/bin/env python3
"""
Training script for Hybrid Quantum-Classical Model
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import sys
import traceback

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.metrics import MetricsCalculator
from models.hybrid_qnn import HybridQNN
from plots.training_plots import TrainingPlotter
from utils.config import (
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, MODELS_DIR,
    N_QUBITS, N_QLAYERS, CLASSICAL_LAYERS
)

def train():
    """Main training pipeline"""
    
    print("HYBRID QUANTUM-CLASSICAL MODEL TRAINING")
    
    try:
        # Load and prepare data
        print("\nLoading and preprocessing data...")
        data = DataLoader()
        data.load_and_preprocess().split_data().scale_features()
        data.perform_eda()
        data.get_data_summary()
        
        # Cross-validation
        print("\nPerforming cross-validation...")
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced to 3 folds for faster testing
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data.X_train, data.y_train)):
            print(f"\nFold {fold+1}/{kf.n_splits}")
            
            X_tr, X_val = data.X_train[train_idx], data.X_train[val_idx]
            y_tr, y_val = data.y_train[train_idx], data.y_train[val_idx]
            
            # Create and train model
            print(f"  Building model with {N_QUBITS} qubits...")
            model = HybridQNN(input_dim=X_tr.shape[1])
            model.build_model().compile_model()
            
            print(f"  Training fold {fold+1}...")
            history = model.model.fit(
                X_tr, y_tr,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(X_val, y_val),
                callbacks=model.get_callbacks(fold+1),
                verbose=0
            )
            
            # Evaluate
            y_val_pred_prob = model.model.predict(X_val, verbose=0)
            y_val_pred = (y_val_pred_prob > 0.5).astype(int)
            
            metrics = MetricsCalculator.calculate_all_metrics(y_val, y_val_pred, y_val_pred_prob)
            cv_metrics.append(metrics)
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  F1-score: {metrics['f1_score']:.4f}")
        
        # Print CV results
        print("\nCross-validation results (mean ± std):")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'specificity']:
            values = [m[metric] for m in cv_metrics]
            print(f"  {metric:15s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # Train final model
        print("\nTraining final model on full training set...")
        final_model = HybridQNN(input_dim=data.X_train.shape[1])
        final_model.build_model().compile_model()
        
        class_weight_dict = data.get_class_weights()
        
        history = final_model.model.fit(
            data.X_train, data.y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            callbacks=final_model.get_callbacks(),
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Save model
        final_model.save(os.path.join(MODELS_DIR, 'final_model'))
        print(f"\nModel saved to {MODELS_DIR}")
        
        # Plot training history
        print("\nGenerating training plots...")
        plotter = TrainingPlotter(history)
        plotter.plot_loss_accuracy()
        plotter.plot_metrics()
        plotter.create_training_dashboard()
        
        # Model summary
        print("\nModel Architecture:")
        final_model.summary()
        
        print(f"Training completed! Model saved with {N_QUBITS} qubits and {N_QLAYERS} layers")
        print(f"Classical layers: {CLASSICAL_LAYERS}")

        
        return final_model, history, data
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, history, data = train()
    
    if model is not None:
        print("\nTraining successful!")
        