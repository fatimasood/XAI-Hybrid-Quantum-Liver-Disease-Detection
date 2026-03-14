#!/usr/bin/env python3
"""
XAI explainability script for Hybrid Quantum-Classical Model
"""

import numpy as np
import tensorflow as tf
import os

from utils.data_loader import DataLoader
from models.hybrid_qnn import HybridQNN
from xai.shap_analysis import SHAPAnalyzer
from xai.integrated_gradients import IntegratedGradients
from xai.permutation_importance import PermutationImportance
from utils.config import MODELS_DIR, FEATURE_NAMES
from xai.ablation_study import FeatureAblator

def explain():
    """Main explainability pipeline"""
    
    print("MODEL EXPLAINABILITY ANALYSIS")
    
    # Load data and model
    print("\nLoading data and model...")
    data = DataLoader()
    data.load_and_preprocess()
    
    model_path = os.path.join(MODELS_DIR, 'final_model')
    
    if os.path.exists(model_path):
        loaded_model = tf.keras.models.load_model(model_path)
    else:
        print(f"Model not found at {model_path}. Training a new model...")
        from train import train
        model, _, _ = train()
        loaded_model = model.model
    
    # SHAP Analysis
    print("\nPerforming SHAP analysis...")
    shap_analyzer = SHAPAnalyzer(
        model=loaded_model,
        X_train=data.X_train,
        X_test=data.X_test,
        feature_names=FEATURE_NAMES
    )
    
    shap_values = shap_analyzer.explain(n_samples=50)
    shap_analyzer.plot_summary()
    shap_analyzer.plot_importance()
    shap_analyzer.plot_waterfall(instance_idx=0)
    shap_analyzer.plot_dependence(feature_idx=0)  # Age
    
    # Get feature importance ranking
    ranked_features = shap_analyzer.get_feature_importance()
    print("\nTop 5 features (SHAP):")
    for i, (feature, importance) in enumerate(ranked_features[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    #Integrated Gradients
    print("\nComputing Integrated Gradients...")
    ig = IntegratedGradients(model=loaded_model, feature_names=FEATURE_NAMES)
    
    ig.plot_feature_importance(data.X_test, y_true=data.y_test, n_samples=100)
    ig.plot_instance_explanation(data.X_test, instance_idx=0)
    ig.plot_instance_explanation(data.X_test, instance_idx=1)
    
    # Permutation Importance
    print("\nComputing Permutation Importance...")
    perm_importance = PermutationImportance(
        model=loaded_model,
        X=data.X_test,
        y=data.y_test,
        feature_names=FEATURE_NAMES,
        metric='accuracy'
    )
    
    importances, stds = perm_importance.calculate(n_repeats=10)
    perm_importance.plot_importance(top_n=10)
    
    # Generate comparison report
    print("\nGenerating comparison report...")
    
    # Get permutation importance ranking
    ranked_perm = perm_importance.get_ranked_features()


    print("\nRunning Feature Ablation Study for Validation...")
    ablator = FeatureAblator(model=loaded_model, data=data)
    # Use SHAP or Permutation rankings to test
    ablation_results = ablator.run_study(ranked_features)
    
    print("\n...... Feature Importance Comparison .....")
    print("\n{:<20s} {:<20s} {:<20s}".format("SHAP Rank", "Feature", "Permutation Rank"))
    
    for i in range(len(FEATURE_NAMES)):
        shap_feature = ranked_features[i][0]
        shap_imp = ranked_features[i][1]
        
        # Find rank in permutation
        perm_rank = next(j for j, (f, imp, std) in enumerate(ranked_perm) if f == shap_feature)
        
        print(f"{i+1:2d}. {shap_feature:<15s} ({shap_imp:.3f})  {perm_rank+1:2d}. {shap_feature:<15s}")
    
    # Save comparison to file
    with open(os.path.join('xai', 'results', 'importance_comparison.txt'), 'w') as f:
        f.write("Feature Importance Comparison\n")
        
        f.write("SHAP Importance:\n")
        for i, (feature, imp) in enumerate(ranked_features):
            f.write(f"{i+1}. {feature}: {imp:.4f}\n")
        
        f.write("\nPermutation Importance:\n")
        for i, (feature, imp, std) in enumerate(ranked_perm):
            f.write(f"{i+1}. {feature}: {imp:.4f} ± {std:.4f}\n")

    print("Explainability analysis completed! All results saved to xai/results/")
    
    return {
        'shap': ranked_features,
        'permutation': ranked_perm,
        'ig': ig
    }

if __name__ == "__main__":
    results = explain()