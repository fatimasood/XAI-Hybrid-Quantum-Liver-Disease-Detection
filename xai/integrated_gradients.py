import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from utils.config import XAI_DIR, FEATURE_NAMES

class IntegratedGradients:
    """Integrated Gradients for feature attribution in Hybrid QNN"""
    
    def __init__(self, model, feature_names=FEATURE_NAMES):
        self.model = model
        self.feature_names = feature_names
        
    def explain(self, X, baseline=None, steps=50):
        """
        Compute integrated gradients for a single instance.
        Formula: (x - x') * integral[grad(x' + alpha(x - x'))]
        """
        # Ensure input is a float32 tensor
        X = tf.cast(X, tf.float32)
        
        if baseline is None:
            baseline = tf.zeros_like(X)
        else:
            baseline = tf.cast(baseline, tf.float32)
        
        # 1. Generate interpolated inputs (the path from baseline to X)
        alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)
        
        # We use tf.newaxis to allow broadcasting: (steps, 1) * (1, features)
        alphas_x = alphas[:, tf.newaxis] 
        delta = X - baseline # (1, features)
        
        # Generate all points along the path
        # Resulting shape: (steps + 1, num_features)
        scaled_inputs = baseline + alphas_x * delta
        
        # 2. Compute Gradients
        with tf.GradientTape() as tape:
            tape.watch(scaled_inputs)
            predictions = self.model(scaled_inputs)
            
        # Get gradients of the output with respect to the interpolated inputs
        grads = tape.gradient(predictions, scaled_inputs)
        
        # 3. Approximate the Integral
        # We take the average of the gradients (trapezoidal rule approximation)
        avg_grads = tf.reduce_mean(grads[:-1], axis=0)
        
        # 4. Calculate Final Attribution: (X - baseline) * avg_grads
        # We use .numpy() to return to standard format for plotting
        integrated_gradients = (delta * avg_grads).numpy()
        
        return integrated_gradients
    
    def explain_batch(self, X, n_samples=100):
        """Compute integrated gradients for a batch of samples"""
        explanations = []
        limit = min(n_samples, len(X))
        
        print(f"Processing {limit} samples for Integrated Gradients...")
        for i in range(limit):
            try:
                # Process one sample: X[i:i+1] keeps the 2D shape (1, features)
                ig = self.explain(X[i:i+1])
                explanations.append(ig.flatten())
                if (i + 1) % 10 == 0:
                    print(f"  Done {i + 1}/{limit} samples...")
            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                explanations.append(np.zeros(len(self.feature_names)))
        
        return np.array(explanations)
    
    def plot_feature_importance(self, X, y_true=None, n_samples=100):
        """Plot feature importance based on integrated gradients"""
        explanations = self.explain_batch(X, n_samples)
        
        # Calculate mean absolute attribution for overall importance
        importance = np.abs(explanations).mean(axis=0)
        
        # Avoid division by zero
        total = importance.sum()
        importance_norm = importance / total if total > 0 else importance
        
        # Sort features by importance
        sorted_idx = np.argsort(importance_norm)[::-1]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(importance_norm)), importance_norm[sorted_idx], 
                      color='#3498db', alpha=0.8)
        
        plt.xticks(range(len(importance_norm)), 
                   [self.feature_names[i] for i in sorted_idx], 
                   rotation=45, ha='right')
        
        plt.xlabel('Features')
        plt.ylabel('Normalized Importance')
        plt.title('Feature Importance (Integrated Gradients)', fontsize=14, fontweight='bold')
        
        # Add values on top of bars
        for bar, val in zip(bars, importance_norm[sorted_idx]):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, 'ig_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return sorted_idx, importance_norm
    
    def plot_instance_explanation(self, X, instance_idx=0):
        """Plot explanation for a single instance (Positive/Negative impact)"""
        explanation = self.explain(X[instance_idx:instance_idx+1]).flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Bar plot of individual attributions
        # Green for positive contribution, Red for negative
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in explanation]
        ax1.bar(range(len(explanation)), explanation, color=colors)
        ax1.set_xticks(range(len(explanation)))
        ax1.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Attribution Value')
        ax1.set_title(f'Feature Attributions: Instance {instance_idx}', fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # 2. Cumulative Effect Plot
        cumsum = np.cumsum(explanation)
        ax2.plot(range(len(explanation)+1), [0] + list(cumsum), 'bo-', linewidth=2, markersize=8)
        ax2.fill_between(range(len(explanation)+1), 0, [0] + list(cumsum), alpha=0.2, color='blue')
        ax2.set_xticks(range(len(explanation)+1))
        ax2.set_xticklabels(['Baseline'] + self.feature_names, rotation=45, ha='right')
        ax2.set_xlabel('Features (Cumulative)')
        ax2.set_ylabel('Cumulative Probability Shift')
        ax2.set_title('Total Effect on Prediction', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, f'ig_instance_{instance_idx}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        return explanation