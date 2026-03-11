import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils.config import PLOTS_DIR

class TrainingPlotter:
    """Plot training history and curves"""
    
    def __init__(self, history):
        self.history = history.history
        
    def plot_loss_accuracy(self):
        """Plot training and validation loss & accuracy"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['loss'], label='Training', lw=2, color='#3498db')
        axes[0].plot(self.history['val_loss'], label='Validation', lw=2, color='#e74c3c')
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['accuracy'], label='Training', lw=2, color='#3498db')
        axes[1].plot(self.history['val_accuracy'], label='Validation', lw=2, color='#e74c3c')
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_metrics(self):
        """Plot additional metrics"""
        
        metrics = ['auc', 'precision', 'recall']
        available_metrics = [m for m in metrics if m in self.history]
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = ['#9b59b6', '#f39c12', '#1abc9c']
        
        for ax, metric, color in zip(axes, available_metrics, colors):
            ax.plot(self.history[metric], label=f'Training {metric}', lw=2, color=color)
            ax.plot(self.history[f'val_{metric}'], label=f'Validation {metric}', 
                   lw=2, color=color, linestyle='--')
            ax.set_title(f'{metric.upper()} over Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'additional_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_learning_rate(self):
        """Plot learning rate if available"""
        
        if 'lr' not in self.history:
            return None
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['lr'], lw=2, color='#e67e22')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'learning_rate.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def plot_gradient_norm(self):
        """Plot gradient norm (if tracked)"""
        
        if 'gradient_norm' not in self.history:
            return None
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['gradient_norm'], lw=2, color='#16a085')
        plt.title('Gradient Norm', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'gradient_norm.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def create_training_dashboard(self):
        """Create comprehensive training dashboard"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history['loss'], 'b-', label='Train', lw=2)
        ax1.plot(self.history['val_loss'], 'r-', label='Val', lw=2)
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.history['accuracy'], 'b-', label='Train', lw=2)
        ax2.plot(self.history['val_accuracy'], 'r-', label='Val', lw=2)
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # AUC
        if 'auc' in self.history:
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.plot(self.history['auc'], 'b-', label='Train', lw=2)
            ax3.plot(self.history['val_auc'], 'r-', label='Val', lw=2)
            ax3.set_title('AUC')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('AUC')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in self.history:
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.plot(self.history['precision'], 'b-', label='Train', lw=2)
            ax4.plot(self.history['val_precision'], 'r-', label='Val', lw=2)
            ax4.set_title('Precision')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Precision')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in self.history:
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.plot(self.history['recall'], 'b-', label='Train', lw=2)
            ax5.plot(self.history['val_recall'], 'r-', label='Val', lw=2)
            ax5.set_title('Recall')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Recall')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # F1-score (if available)
        if 'f1_score' in self.history:
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.plot(self.history['f1_score'], 'b-', label='Train', lw=2)
            ax6.plot(self.history['val_f1_score'], 'r-', label='Val', lw=2)
            ax6.set_title('F1-Score')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('F1-Score')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Learning rate
        if 'lr' in self.history:
            ax7 = fig.add_subplot(gs[2, :])
            ax7.plot(self.history['lr'], 'g-', lw=2)
            ax7.set_title('Learning Rate')
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('Learning Rate')
            ax7.set_yscale('log')
            ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Training Dashboard', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(PLOTS_DIR, 'training_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig