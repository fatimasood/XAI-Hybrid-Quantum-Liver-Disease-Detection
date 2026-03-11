import shap
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.config import XAI_DIR, FEATURE_NAMES, CLASS_NAMES

class SHAPAnalyzer:
    """SHAP-based model explanations with multi-output and list handling"""
    
    def __init__(self, model, X_train, X_test, feature_names=FEATURE_NAMES):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def _get_relevant_shap_values(self):
        """Helper to extract the correct array from SHAP list or array"""
        if isinstance(self.shap_values, list):
            # For binary models, index 1 usually represents the 'Positive' class 
            # (Disease), but index 0 works too as they are mirrors.
            return self.shap_values[0]
        return self.shap_values

    def explain(self, n_samples=100):
        """Generate SHAP explanations using KernelExplainer"""
        # Create background dataset
        indices = np.random.choice(self.X_train.shape[0], n_samples, replace=False)
        background = self.X_train[indices]
        
        # Create explainer
        self.explainer = shap.KernelExplainer(self.model.predict, background)
        
        # Calculate SHAP values for the first n_samples of test set
        self.shap_values = self.explainer.shap_values(self.X_test[:n_samples])
        
        return self.shap_values
    
    def plot_summary(self):
        """Create SHAP summary plot"""
        vals = self._get_relevant_shap_values()
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(vals, self.X_test[:len(vals)], 
                         feature_names=self.feature_names, show=False)
        plt.title('SHAP Feature Impact Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_importance(self):
        """Create SHAP feature importance bar plot"""
        vals = self._get_relevant_shap_values()
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(vals, self.X_test[:len(vals)], 
                         feature_names=self.feature_names, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, 'shap_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_waterfall(self, instance_idx=0):
        """Create waterfall plot for a specific instance"""
        # Extract specific SHAP values for the instance
        if isinstance(self.shap_values, list):
            current_shap_values = self.shap_values[0][instance_idx]
            current_expected_value = self.explainer.expected_value[0]
        else:
            current_shap_values = self.shap_values[instance_idx]
            current_expected_value = self.explainer.expected_value

        # Ensure expected_value is a scalar
        if isinstance(current_expected_value, (np.ndarray, list)):
            current_expected_value = current_expected_value[0]

        plt.figure(figsize=(12, 8))
        
        # Build Explanation object required by the new waterfall API
        exp = shap.Explanation(
            values=current_shap_values,
            base_values=current_expected_value,
            data=self.X_test[instance_idx],
            feature_names=self.feature_names
        )
        
        shap.plots.waterfall(exp, show=False)
        
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, f'shap_waterfall_{instance_idx}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_dependence(self, feature_idx):
        """Create SHAP dependence plot for a feature"""
        vals = self._get_relevant_shap_values()
        
        plt.figure(figsize=(10, 6))
        # Use the legacy dependence_plot interface but with corrected array
        shap.dependence_plot(feature_idx, vals, self.X_test[:len(vals)],
                            feature_names=self.feature_names, show=False)
        
        feature_name = self.feature_names[feature_idx] if self.feature_names else f"Feature {feature_idx}"
        plt.title(f'SHAP Dependence Plot - {feature_name}', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, f'shap_dependence_{feature_idx}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_feature_importance(self):
        """Get ranked feature importance from SHAP"""
        vals = self._get_relevant_shap_values()
        
        # Calculate mean absolute SHAP values
        importance = np.abs(vals).mean(axis=0)
        
        # Handle division by zero if all importance values are zero
        total_importance = importance.sum()
        if total_importance > 0:
            importance_norm = importance / total_importance
        else:
            importance_norm = importance
        
        # Create ranked list
        ranked_features = sorted(
            zip(self.feature_names, importance_norm),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked_features