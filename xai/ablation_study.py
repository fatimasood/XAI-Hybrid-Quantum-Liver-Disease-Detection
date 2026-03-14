import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.config import XAI_DIR, FEATURE_NAMES

class FeatureAblator:
    """Validates XAI by removing top vs bottom features"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.results = {}

    def run_study(self, ranked_features):
        """
        ranked_features: list of feature names sorted by importance
        """
        X_test = self.data.X_test
        y_test = self.data.y_test
        
        # Baseline (All features)
        y_pred = (self.model.predict(X_test).flatten() > 0.5).astype(int)
        baseline_acc = accuracy_score(y_test, y_pred)
        
        self.results['Baseline'] = baseline_acc
        print(f"Baseline Accuracy: {baseline_acc:.4f}")

        # Remove Top 3 Features (Top-Down)
        top_3 = [f[0] for f in ranked_features[:3]]
        X_no_top = X_test.copy()
        for f_name in top_3:
            idx = FEATURE_NAMES.index(f_name)
            X_no_top[:, idx] = 0 # "Ablate" by setting to zero
            
        y_pred_no_top = (self.model.predict(X_no_top).flatten() > 0.5).astype(int)
        acc_no_top = accuracy_score(y_test, y_pred_no_top)
        self.results['Without Top 3'] = acc_no_top
        print(f"Accuracy without {top_3}: {acc_no_top:.4f}")

        # Remove Bottom 3 Features (Bottom-Up)
        bottom_3 = [f[0] for f in ranked_features[-3:]]
        X_no_bottom = X_test.copy()
        for f_name in bottom_3:
            idx = FEATURE_NAMES.index(f_name)
            X_no_bottom[:, idx] = 0
            
        y_pred_no_bottom = (self.model.predict(X_no_bottom).flatten() > 0.5).astype(int)
        acc_no_bottom = accuracy_score(y_test, y_pred_no_bottom)
        self.results['Without Bottom 3'] = acc_no_bottom
        print(f"Accuracy without {bottom_3}: {acc_no_bottom:.4f}")

        self.plot_results()
        return self.results

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        names = list(self.results.keys())
        values = list(self.results.values())
        
        colors = ['gray', 'red', 'green']
        plt.bar(names, values, color=colors, alpha=0.7)
        plt.ylim(0, 1.0)
        plt.ylabel('Accuracy')
        plt.title('Feature Ablation Study: XAI Validation', fontsize=14, fontweight='bold')
        
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
            
        plt.savefig(os.path.join(XAI_DIR, 'ablation_study.png'))
        plt.show()