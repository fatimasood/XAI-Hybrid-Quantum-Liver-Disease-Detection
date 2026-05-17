import os, sys
import numpy as np
from dotenv import load_dotenv
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
load_dotenv(os.path.join(base_dir, ".env"))

from utils.data_loader import DataLoader
from utils.config import MODELS_DIR, FEATURE_NAMES
import tensorflow as tf
from llm.advisor import LLMHealthAdvisor, estimate_confidence_interval
from llm.xai_extractor import XAIFeatureExtractor

def main():
    print("Loading data and model...")
    data = DataLoader()
    data.load_and_preprocess()   # now X_test_original is available
    
    model_path = os.path.join(MODELS_DIR, 'final_model')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Train model first. Not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print("Preparing SHAP explainer (takes ~30s for KernelSHAP)...")
    xai_ext = XAIFeatureExtractor(model, data.X_train, data.X_test)
    xai_ext.prepare_shap(n_background=50)
    
    advisor = LLMHealthAdvisor()
    
    artifact_file = os.path.join(base_dir, "xai_thesis_artifacts.md")
    with open(artifact_file, "w", encoding="utf-8") as f:
        f.write("# Explainable Clinical Reports (Real SHAP + QNN)\n")
        f.write(f"Model: {advisor.model_name}\n\n---\n\n")
    
    n_samples = min(3, len(data.X_test_original))
    for i in range(n_samples):
        print(f"\n{'='*60}\nSample {i}")
        # Original scale features for LLM
        row = data.X_test_original.iloc[i]
        features = row.to_dict()
        
        # Scaled input for prediction
        X_scaled = data.X_test[i:i+1]
        prob = model.predict(X_scaled, verbose=0).flatten()[0]
        ci_low, ci_high = estimate_confidence_interval(model, X_scaled, n_iter=30, noise_std=0.05)
        
        # Real SHAP
        top_features = xai_ext.get_top_features(i, top_k=3)
        shap_dict = xai_ext.get_shap_dict(i)            # all features
        ablation_dict = xai_ext.get_ablation_dict(top_features)
        
        print("Top SHAP:")
        for f in top_features:
            print(f"  {f['feature']}: {f['shap_impact']:+.4f}")
        print(f"Probability: {prob:.3f}, CI: [{ci_low:.3f}, {ci_high:.3f}]")
        
        report = advisor.get_recommendations(
            features=features,
            prob=prob,
            shap_values=shap_dict,
            ablation_impact=ablation_dict,
            ci_lower=ci_low,
            ci_upper=ci_high
        )
        print("\nLLM Report:\n" + report)
        
        with open(artifact_file, "a", encoding="utf-8") as f:
            f.write(f"## Sample {i} (Prob={prob:.3f})\n")
            f.write("SHAP Vectors:\n")
            for k, v in shap_dict.items():
                f.write(f"- {k}: {v:+.4f}\n")
            f.write("\nAblation:\n")
            for k, v in ablation_dict.items():
                f.write(f"- {k}: {v:+.4f}\n")
            f.write(f"\n{report}\n\n---\n\n")
    
    print(f"\nArtifacts saved to {artifact_file}")

if __name__ == "__main__":
    main()