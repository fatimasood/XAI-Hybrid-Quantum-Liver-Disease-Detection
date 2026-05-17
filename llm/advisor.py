import os
import numpy as np
from typing import Dict, Optional
from huggingface_hub import InferenceClient

class LLMHealthAdvisor:
    REFERENCE_RANGES = {
        'Age':      'Adult context',
        'Gender':   '0 = Male, 1 = Female',
        'TB':       '0.1–1.2 mg/dL (Total Bilirubin)',
        'DB':       '0.0–0.3 mg/dL (Direct Bilirubin)',
        'Alkphos':  '40–129 U/L (Alkaline Phosphatase)',
        'Sgpt':     '10–40 U/L (ALT)',
        'Sgot':     '10–40 U/L (AST)',
        'TP':       '6.6–8.7 g/dL (Total Protein)',
        'ALB':      '3.5–5.0 g/dL (Albumin)',
        'A/G':      '1.1–2.5 (Albumin/Globulin Ratio)',
    }

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", api_token=None):
        self.model_name = model_name
        self.client = InferenceClient(
            model=self.model_name,
            token=api_token or os.getenv("HF_TOKEN")
        )

    def _analyze_clinical_anomalies(self, features):
        anomalies = []
        if features.get('TB', 0) > 1.2: anomalies.append(f"Elevated Total Bilirubin ({features['TB']} mg/dL)")
        if features.get('DB', 0) > 0.3: anomalies.append(f"Elevated Direct Bilirubin ({features['DB']} mg/dL)")
        if features.get('Alkphos', 0) > 129: anomalies.append(f"Elevated Alkaline Phosphatase ({features['Alkphos']} U/L)")
        if features.get('Sgpt', 0) > 40: anomalies.append(f"Elevated SGPT/ALT ({features['Sgpt']} U/L)")
        if features.get('Sgot', 0) > 40: anomalies.append(f"Elevated SGOT/AST ({features['Sgot']} U/L)")
        if features.get('Sgot', 0) > 0 and features.get('Sgpt', 0) > 0:
            ratio = features['Sgot'] / features['Sgpt']
            if ratio > 1.5 and features['Sgot'] > 40:
                anomalies.append(f"High AST/ALT Ratio ({ratio:.2f})")
        return "\n".join([f"  * CRITICAL ANOMALY: {a}" for a in anomalies]) if anomalies else "No extreme baseline boundary violations."

    def _build_prompt(self, features, prob, shap_values, ablation_impact, ci_lower, ci_upper):
        risk = "low" if prob < 0.3 else ("moderate" if prob < 0.7 else "high")
        feat_str = "\n".join([f"  - {k}: {v} (Normal: {self.REFERENCE_RANGES.get(k, 'N/A')})" for k, v in features.items()])
        clinical = self._analyze_clinical_anomalies(features)
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        shap_str = "\n".join([f"  * SHAP -> {k}: {v:+.4f}" for k, v in sorted_shap])
        ablation_str = "\n".join([f"  * Ablation -> Removing [{k}] drops accuracy by: {v:+.4f}" for k, v in ablation_impact.items()])
        ci_str = f"Quantum CI (95%): [{ci_lower:.3f} – {ci_upper:.3f}].\n" if ci_lower is not None else ""

        return f"""[SYSTEM DATA INGESTION PIPELINE]
Patient Quantitative Metrics:
{feat_str}
Hybrid Quantum-Classical Model Analysis:
- Target Probability: {prob:.3f}
- Classification: {risk.upper()} RISK
- {ci_str}
Explainable AI (XAI) Mathematical Metrics:
{shap_str}
Ablation Study Feature Sensitivities:
{ablation_str}
Baseline Clinical Anomalies:
{clinical}

[INSTRUCTION]: Map the SHAP attributions and Ablation drops directly to the health strategy. State mathematically why the model prioritized specific parameters."""

    def get_recommendations(self, features, prob, shap_values, ablation_impact,
                            ci_lower=None, ci_upper=None, max_new_tokens=1500):
        user_msg = self._build_prompt(features, prob, shap_values, ablation_impact, ci_lower, ci_upper)
        system_instruction = (
            "You are an expert Explainable AI (XAI) Decision Support System for Quantum Medical Frameworks. "
    "Your objective is to provide professional, non-diagnostic lifestyle and clinical strategies based on mathematical parameters. "
    "\n\nCRITICAL CONSTRAINTS FOR EXAMINER VALIDATION:\n"
    "1. You MUST start with the exact header: '**XAI Quantum Attribution Ingestion Review**'. Under this, explicitly identify the features "
    "with the highest absolute SHAP values and the highest Ablation performance drops.\n"
    "2. MATHEMATICAL CORRECTNESS RULE: A positive (+) SHAP value means that feature contributed to INCREASING the liver disease risk score. "
    "A negative (-) SHAP value means that feature contributed to DECREASING the risk score (protecting the patient or lowering model confidence). "
    "You must explain this distinction correctly based on the signs provided. Do not invert this logic.\n"
    "3. CLINICAL CONTEXT RULE: Only call a parameter a 'critical anomaly' or 'abnormal' if it genuinely falls outside its provided normal range. "
    "If a value is inside the normal range (e.g., Total Protein 7.9 within 6.6-8.7), explicitly treat it as stable/normal.\n"
    "4. Do NOT dump raw SHAP or Ablation dictionaries inside the lifestyle bullet points. Keep the analysis restricted to the first section.\n"
    "5. Structure the rest of the report using these exact markdown sections cleanly without redundancy:\n"
    "   - **Targeted Dietary & Hydration Interventions**\n"
    "   - **Metabolic Tracking & Physical Load Adjustments**\n"
    "   - **Recommended Diagnostic Monitoring Protocols**\n"
    "6. Keep responses highly concise and structured to prevent token truncation midway. Maintain a non-diagnostic posture.\n"
    "7. You MUST terminate your complete response with exactly this literal string and nothing else after it:\n"
    "Disclaimer: This is AI generated information for educational purposes only. Always consult a qualified healthcare provider."
        )

        try:
            # Use chat_completion (available for most recent models on HF)
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=max_new_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to text_generation with chat template
            try:
                prompt = f"<|system|>\n{system_instruction}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>"
                output = self.client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=0.1, return_full_text=False)
                return output.strip()
            except Exception as e2:
                return f"LLM API error: {e2}"


def estimate_confidence_interval(model, X_sample, n_iter=30, noise_std=0.05):
    preds = []
    for _ in range(n_iter):
        X_perturbed = X_sample + np.random.normal(0, noise_std, X_sample.shape)
        preds.append(model.predict(X_perturbed, verbose=0).flatten()[0])
    preds = np.array(preds)
    return np.percentile(preds, 2.5), np.percentile(preds, 97.5)