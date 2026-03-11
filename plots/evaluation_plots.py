import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import os
from utils.config import PLOTS_DIR

class EvaluationPlotter:
    """Plot evaluation results and metrics"""
    
    def __init__(self, y_true, y_pred, y_prob, class_names=['No Disease', 'Disease']):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.class_names = class_names
        
    def plot_confusion_matrix(self):
        """Plot confusion matrix with absolute and normalized views"""
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Absolute values
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Normalized values
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax2,
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Proportion'})
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_roc_curve(self, ci_bounds=None):
        """Plot ROC curve with optional confidence intervals"""
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        
        if ci_bounds:
            plt.fill_between(fpr, ci_bounds[0], ci_bounds[1], 
                           alpha=0.2, color='darkorange', label='95% CI')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def plot_pr_curve(self):
        """Plot Precision-Recall curve"""
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='green', lw=2, label='PR curve')
        plt.axhline(y=self.y_true.mean(), color='navy', lw=2, linestyle='--', 
                   label=f'Baseline ({self.y_true.mean():.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'pr_curve.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def plot_prediction_distribution(self):
        """Plot distribution of predicted probabilities"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(self.y_prob[self.y_true==0], bins=30, alpha=0.7, 
                label='Class 0', color='#2ecc71')
        ax1.hist(self.y_prob[self.y_true==1], bins=30, alpha=0.7, 
                label='Class 1', color='#e74c3c')
        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Prediction Distribution by True Class', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data = [self.y_prob[self.y_true==0], self.y_prob[self.y_true==1]]
        bp = ax2.boxplot(data, labels=['Class 0', 'Class 1'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        ax2.set_ylabel('Predicted Probability')
        ax2.set_title('Prediction Distribution Box Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'prediction_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_calibration_curve(self, n_bins=10):
        """Plot calibration curve"""
        
        # Bin predictions
        bin_counts, bin_edges = np.histogram(self.y_prob, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate fraction of positives in each bin
        fraction_positives = []
        for i in range(n_bins):
            mask = (self.y_prob >= bin_edges[i]) & (self.y_prob < bin_edges[i+1])
            if np.sum(mask) > 0:
                fraction_positives.append(np.mean(self.y_true[mask]))
            else:
                fraction_positives.append(0)
        
        plt.figure(figsize=(10, 8))
        
        # Perfect calibration line
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        
        # Model calibration
        plt.plot(bin_centers, fraction_positives, 'o-', lw=2, 
                color='#9b59b6', label='Model')
        
        # Histogram of predictions
        plt.hist(self.y_prob, bins=n_bins, density=True, alpha=0.3, 
                color='gray', label='Prediction Density')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'calibration_curve.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def plot_metrics_radar(self, metrics_dict):
        """Plot radar chart of metrics"""
        
        # Prepare data
        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        # Number of variables
        N = len(categories)
        
        # Create angles
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]
        angles += angles[:1]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'metrics_radar.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig