# llm/advisor.py
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
        return ", ".join(anomalies) if anomalies else "None"

    def _build_prompt(self, features, prob, shap_values, ablation_impact, ci_lower, ci_upper):
        risk = "low" if prob < 0.3 else ("moderate" if prob < 0.7 else "high")
        clinical_outliers = self._analyze_clinical_anomalies(features)
        
        # PYTHON DETECTOR FOR MATHEMATICAL DIRECTIONS (Strict Guardrails)
        pathological_drivers = [f"{k} ({v:+.4f})" for k, v in shap_values.items() if v > 0]
        protective_factors = [f"{k} ({v:+.4f})" for k, v in shap_values.items() if v <= 0]
        
        patho_str = ", ".join(pathological_drivers[:2]) if pathological_drivers else "None detected"
        prot_str = ", ".join(protective_factors[:2]) if protective_factors else "None detected"
        
        # Identify if any critical outlier has an inverted SHAP sign
        anomalies_log = []
        if features.get('Alkphos', 0) > 129 and shap_values.get('Alkphos', 0) < 0:
            anomalies_log.append("WARNING: Elevated Alkphos is acting as a protective factor (Negative SHAP). This indicates an inverse mathematical feature relationship.")
        if features.get('TB', 0) > 1.2 and shap_values.get('TB', 0) < 0:
            anomalies_log.append("WARNING: Elevated Total Bilirubin is acting as a protective factor (Negative SHAP).")
        if features.get('DB', 0) > 0.3 and shap_values.get('DB', 0) < 0:
            anomalies_log.append("WARNING: Elevated Direct Bilirubin is acting as a protective factor (Negative SHAP).")
            
        anomalies_str = "\n".join([f"  * {item}" for item in anomalies_log]) if anomalies_log else "  * None. Feature directions align cleanly with standard expectations."

        ablation_str = "\n".join([f"  * Drop [{k}] layer reduces architecture accuracy by: {v:.4f}" for k, v in ablation_impact.items() if v > 0.02])
        ci_str = f"95% CI Zone: [{ci_lower:.3f} – {ci_upper:.3f}]"

        return f"""[DATA LOG]
Patient Metrics: {list(features.items())}
Model Risk Probability: {prob:.3f} ({risk.upper()} RISK) | {ci_str}

[SHAP VERIFICATION]
* Calculated Risk Drivers (+ SHAP): {patho_str}
* Calculated Protective Factors (- SHAP): {prot_str}
* Mathematical Inversion Check:
{anomalies_str}

[GLOBAL WEIGHTS]
{ablation_str}

[CLINICAL OBSERVATION]
Active Outliers: {clinical_outliers}"""

    def get_recommendations(self, features, prob, shap_values, ablation_impact,
                            ci_lower=None, ci_upper=None, max_new_tokens=600):
        user_msg = self._build_prompt(features, prob, shap_values, ablation_impact, ci_lower, ci_upper)
        
        # CLINICALLY SAFE SYSTEM PROMPT
        system_instruction = (
            "You are a precise, human-style Medical Informatics decision support system.\n"
            "CRITICAL PROTOCOLS:\n"
            "1. Output must be perfectly concise, professional, direct, and completely free of conversational fluff.\n"
            "2. Under 'Mathematical Sensitivity', map the exact risk drivers and protective factors provided in the user log. If a 'WARNING' is listed under the Mathematical Inversion Check, you MUST explicitly name it as a model architectural artifact.\n"
            "3. DO NOT order random dietary restrictions (e.g., do not advise restricting protein or fat arbitrarily as this can cause sarcopenia or mask diagnostic patterns). Advise maintaining balanced nutrition.\n"
            "4. DO NOT attribute liver enzyme anomalies to 'dehydration'.\n"
            "5. If Total Bilirubin, Direct Bilirubin, or Alkphos are elevated, always prioritize immediate Right Upper Quadrant (RUQ) abdominal ultrasound tracking over delayed imaging.\n\n"
            "Strictly follow this layout and structure:\n\n"
            "**XAI Quantum Attribution Ingestion Review**\n"
            "- Mathematical Sensitivity: [Identify features increasing/decreasing risk exactly as computed in the logs. Explicitly note any structural artifacts/warnings if present].\n"
            "- Global Architectural Weights: [State which top biomarkers trigger high ablation drops, validating that the hybrid quantum model relies on true medical biomarkers over demographic noise].\n\n"
            "**Targeted Dietary & Hydration Interventions**\n"
            "- [First highly concise bullet point focusing on maintaining a balanced, nutrient-dense diet to avoid nutritional deficits]\n"
            "- [Second short, precise bullet point regarding standard hydration to support metabolic baseline]\n\n"
            "**Metabolic Tracking & Physical Load Adjustments**\n"
            "- [First brief bullet point regarding maintaining stable daily baseline activities]\n"
            "- [Second short bullet point stating no active physical load limits or exercise restrictions are indicated unless symptomatic]\n\n"
            "**Recommended Diagnostic Monitoring Protocols**\n"
            "- [First brief tracking protocol, e.g., immediate Right Upper Quadrant (RUQ) abdominal ultrasound if biliary markers (TB, DB, Alkphos) are elevated]\n"
            "- [Second short tracking point, e.g., repeat comprehensive hepatic panel testing to monitor acute trends]\n\n"
            "You MUST terminate your complete response with exactly this literal string and nothing else after it:\n"
            "Disclaimer: This is AI generated information for educational purposes only. Always consult a qualified healthcare provider."
        )

        try:
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
            try:
                prompt = f"<|system|>\n{system_instruction}</s>\n<|user|>\n{user_msg}</s>\n<|assistant|>"
                output = self.client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=0.1, return_full_text=False)
                return output.strip()
            except Exception as e2:
                return f"LLM Execution Error: {e2}"

def estimate_confidence_interval(model, X_sample, n_iter=30, noise_std=0.05):
    preds = []
    for _ in range(n_iter):
        X_perturbed = X_sample + np.random.normal(0, noise_std, X_sample.shape)
        preds.append(model.predict(X_perturbed, verbose=0).flatten()[0])
    preds = np.array(preds)
    return np.percentile(preds, 2.5), np.percentile(preds, 97.5)
