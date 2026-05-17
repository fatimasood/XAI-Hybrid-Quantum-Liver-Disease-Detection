import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xai.shap_analysis import SHAPAnalyzer
from utils.config import FEATURE_NAMES

class XAIFeatureExtractor:
    def __init__(self, model, X_train, X_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = FEATURE_NAMES
        self.shap_analyzer = None
        self.shap_values = None

    def prepare_shap(self, n_background=50):
        """Run KernelSHAP once for all test samples."""
        self.shap_analyzer = SHAPAnalyzer(
            model=self.model,
            X_train=self.X_train,
            X_test=self.X_test,
            feature_names=self.feature_names
        )
        self.shap_values = self.shap_analyzer.explain(n_samples=n_background)

    def get_top_features(self, sample_idx, top_k=3):
        """Return top-k features with their SHAP values for a given test sample."""
        if self.shap_values is None:
            self.prepare_shap()
        vals = self.shap_analyzer._get_relevant_shap_values()
        sample_shap = vals[sample_idx]
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:top_k]
        top_features = []
        for idx in sorted_idx:
            top_features.append({
                "feature": self.feature_names[idx],
                "shap_impact": sample_shap[idx]
            })
        return top_features

    def get_shap_dict(self, sample_idx):
        """Return all features with SHAP values as dict."""
        top = self.get_top_features(sample_idx, top_k=len(self.feature_names))
        return {f['feature']: f['shap_impact'] for f in top}

    @staticmethod
    def get_ablation_dict(top_features):
        """Convert SHAP magnitudes to pseudo-ablation impact (0–0.1 scale)."""
        max_shap = max(abs(f['shap_impact']) for f in top_features) if top_features else 1
        ablation = {}
        for f in top_features:
            ablation[f['feature']] = (abs(f['shap_impact']) / max_shap) * 0.1 if max_shap != 0 else 0.0
        return ablation