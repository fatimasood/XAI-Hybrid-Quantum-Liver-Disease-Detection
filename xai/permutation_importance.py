import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from utils.config import XAI_DIR, FEATURE_NAMES

class PermutationImportance:
    """Permutation feature importance for Hybrid QNN"""
    
    def __init__(self, model, X, y, feature_names=FEATURE_NAMES, metric='accuracy'):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.metric = metric
        self.importances_ = None
        self.importances_std_ = None
        
    def calculate(self, n_repeats=5, random_state=42):
        """
        Calculate permutation importance. 
        Note: Reduced default n_repeats to 5 because Quantum simulation is slow.
        """
        np.random.seed(random_state)
        
        # 1. Calculate baseline score
        y_pred_prob = self.model.predict(self.X).flatten() # Added .flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        if self.metric == 'accuracy':
            baseline_score = accuracy_score(self.y, y_pred)
        elif self.metric == 'roc_auc':
            baseline_score = roc_auc_score(self.y, y_pred_prob)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
            
        print(f"Baseline {self.metric}: {baseline_score:.4f}")
        
        # 2. Calculate importance for each feature
        n_features = self.X.shape[1]
        importances = np.zeros((n_features, n_repeats))
        
        # Use a copy of X to avoid modifying original data
        X_curr = self.X.copy()
        
        for i in range(n_features):
            print(f"Analyzing feature {i+1}/{n_features}: {self.feature_names[i]}")
            for r in tqdm(range(n_repeats), desc=f"  Permutations", leave=False):
                X_permuted = X_curr.copy()
                
                # Shuffle only the current feature column
                np.random.shuffle(X_permuted[:, i])
                
                # Get new predictions with corrupted feature
                y_p_prob = self.model.predict(X_permuted).flatten()
                y_p = (y_p_prob > 0.5).astype(int)
                
                if self.metric == 'accuracy':
                    score = accuracy_score(self.y, y_p)
                else:
                    score = roc_auc_score(self.y, y_p_prob)
                
                # Importance is how much the score DROPS when feature is ruined
                importances[i, r] = baseline_score - score
        
        self.importances_ = np.mean(importances, axis=1)
        self.importances_std_ = np.std(importances, axis=1)
        
        return self.importances_, self.importances_std_
    
    def plot_importance(self, top_n=None):
        """Plot permutation importance horizontally"""
        if self.importances_ is None:
            raise ValueError("Please calculate importances first using .calculate()")
        
        sorted_idx = np.argsort(self.importances_)
        
        if top_n is not None:
            sorted_idx = sorted_idx[-top_n:] # Take top N
        
        names = [self.feature_names[i] for i in sorted_idx]
        values = self.importances_[sorted_idx]
        errors = self.importances_std_[sorted_idx]
        
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(names))
        
        # Using a nice color palette
        bars = plt.barh(y_pos, values, xerr=errors, color='skyblue', capsize=5)
        
        plt.yticks(y_pos, names)
        plt.axvline(x=0, color='black', lw=1, ls='--')
        plt.xlabel(f'Decrease in {self.metric} (Importance)')
        plt.title(f'Permutation Importance: {self.metric.upper()}', fontsize=14, fontweight='bold')
        
        # Label bars with values
        for i, (val, err) in enumerate(zip(values, errors)):
            plt.text(val + err + 0.002, i, f'{val:.4f}±{err:.3f}', va='center', fontsize=9)
            
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, f'permutation_importance_{self.metric}.png'), dpi=300)
        plt.show()
        
    def get_ranked_features(self):
        """Returns feature names ranked by importance"""
        if self.importances_ is None:
            return []
        
        ranked = sorted(
            zip(self.feature_names, self.importances_, self.importances_std_),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked