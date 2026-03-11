import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss, roc_auc_score,
    roc_curve, precision_recall_curve
)

class MetricsCalculator:
    """Comprehensive metrics calculation for model evaluation"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_prob):
        """Calculate comprehensive set of evaluation metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred),
            'brier_score': brier_score_loss(y_true, y_prob),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob)
        }
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Derived metrics
        metrics['youden_index'] = metrics['recall'] + metrics['specificity'] - 1
        metrics['diagnostic_odds_ratio'] = (tp * tn) / (fp * fn) if (fp * fn) > 0 else np.inf
        
        return metrics
    
    @staticmethod
    def calculate_confidence_intervals(y_true, y_prob, n_bootstrap=1000, ci=95):
        """Calculate bootstrap confidence intervals for AUC metrics"""
        
        np.random.seed(42)
        roc_aucs = []
        pr_aucs = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[indices])) == 2:
                roc_aucs.append(roc_auc_score(y_true[indices], y_prob[indices]))
                pr_aucs.append(average_precision_score(y_true[indices], y_prob[indices]))
        
        alpha = (100 - ci) / 2
        roc_ci = np.percentile(roc_aucs, [alpha, 100 - alpha])
        pr_ci = np.percentile(pr_aucs, [alpha, 100 - alpha])
        
        return {
            'roc_auc': np.mean(roc_aucs),
            'roc_ci': roc_ci,
            'pr_auc': np.mean(pr_aucs),
            'pr_ci': pr_ci
        }
    
    @staticmethod
    def get_roc_curve_data(y_true, y_prob):
        """Get ROC curve data"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return fpr, tpr, thresholds
    
    @staticmethod
    def get_pr_curve_data(y_true, y_prob):
        """Get Precision-Recall curve data"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        return precision, recall, thresholds