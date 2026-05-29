# Explainable Clinical Reports (Real SHAP + QNN)
Model: Qwen/Qwen2.5-7B-Instruct

---

## Sample 0 (Prob=0.414)
SHAP Vectors:
- ALB: -0.1774
- TP: +0.0904
- Alkphos: -0.0467
- Gender: -0.0274
- DB: +0.0139
- A/G: -0.0099
- Sgpt: -0.0088
- Age: +0.0085
- Sgot: -0.0055
- TB: +0.0053

Ablation:
- ALB: +0.1000
- TP: +0.0509
- Alkphos: +0.0263

**XAI Quantum Attribution Ingestion Review**
- Mathematical Sensitivity: TP (+0.0904) increases risk, while DB (+0.0139) and Alkphos (-0.0467) decrease risk. The model incorrectly identifies Alkphos as a protective factor, which is a model architectural artifact.
- Global Architectural Weights: TP and ALB are the top biomarkers triggering high ablation drops, validating the model's reliance on true medical biomarkers.

**Targeted Dietary & Hydration Interventions**
- Maintain a balanced, nutrient-dense diet to avoid nutritional deficits.
- Ensure standard hydration to support metabolic baseline.

**Metabolic Tracking & Physical Load Adjustments**
- Maintain stable daily baseline activities.
- No active physical load limits or exercise restrictions are indicated unless symptomatic.

**Recommended Diagnostic Monitoring Protocols**
- Immediate Right Upper Quadrant (RUQ) abdominal ultrasound if biliary markers (TB, DB, Alkphos) are elevated.
- Repeat comprehensive hepatic panel testing to monitor acute trends.

Disclaimer: This is AI generated information for educational purposes only. Always consult a qualified healthcare provider.

---

## Sample 1 (Prob=0.856)
SHAP Vectors:
- DB: +0.0764
- Age: +0.0544
- ALB: +0.0523
- TP: +0.0340
- TB: +0.0271
- Sgot: +0.0254
- Gender: +0.0105
- A/G: +0.0083
- Alkphos: -0.0047
- Sgpt: +0.0010

Ablation:
- DB: +0.1000
- Age: +0.0712
- ALB: +0.0684

**XAI Quantum Attribution Ingestion Review**
- Mathematical Sensitivity: Elevated Direct Bilirubin (DB) (+0.0764) and Age (+0.0544) increase risk, while Elevated Alkaline Phosphatase (Alkphos) (-0.0047) is a protective factor. The model architectural artifact WARNING indicates an inverse relationship for Alkphos.
- Global Architectural Weights: The top biomarkers triggering high ablation drops are Direct Bilirubin (DB) and Age, validating the model's reliance on true medical biomarkers.

**Targeted Dietary & Hydration Interventions**
- Maintain a balanced, nutrient-dense diet to avoid nutritional deficits.
- Ensure standard hydration to support metabolic baseline.

**Metabolic Tracking & Physical Load Adjustments**
- Maintain stable daily baseline activities.
- No active physical load limits or exercise restrictions are indicated unless symptomatic.

**Recommended Diagnostic Monitoring Protocols**
- Immediate Right Upper Quadrant (RUQ) abdominal ultrasound if Total Bilirubin (TB), Direct Bilirubin (DB), or Alkaline Phosphatase (Alkphos) are elevated.
- Repeat comprehensive hepatic panel testing to monitor acute trends.

Disclaimer: This is for educational purposes only. Always consult a qualified healthcare provider.

---

## Sample 2 (Prob=0.857)
SHAP Vectors:
- DB: +0.0856
- Alkphos: +0.0650
- TB: +0.0429
- ALB: +0.0400
- TP: +0.0297
- A/G: +0.0250
- Gender: -0.0055
- Age: -0.0041
- Sgpt: +0.0040
- Sgot: +0.0027

Ablation:
- DB: +0.1000
- Alkphos: +0.0759
- TB: +0.0502

**XAI Quantum Attribution Ingestion Review**
- Mathematical Sensitivity: DB (+0.0856), Alkphos (+0.0650) increase risk; Gender (-0.0055), Age (-0.0041) decrease risk. No structural artifacts/warnings noted.
- Global Architectural Weights: Alkphos (0.0759) and DB (0.1000) trigger high ablation drops, validating reliance on true medical biomarkers.

**Targeted Dietary & Hydration Interventions**
- Maintain a balanced, nutrient-dense diet to avoid nutritional deficits.
- Ensure standard hydration to support metabolic baseline.

**Metabolic Tracking & Physical Load Adjustments**
- Maintain stable daily baseline activities.
- No active physical load limits or exercise restrictions are indicated unless symptomatic.

**Recommended Diagnostic Monitoring Protocols**
- Immediate Right Upper Quadrant (RUQ) abdominal ultrasound if biliary markers (TB, DB, Alkphos) are elevated.
- Repeat comprehensive hepatic panel testing to monitor acute trends.

Disclaimer: This is for educational purposes only. Always consult a qualified healthcare provider.

---

