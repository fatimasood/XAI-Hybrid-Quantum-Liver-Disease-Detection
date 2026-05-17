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

- **ALB (Albumin)**: SHAP value = -0.1774 (negative, indicating a decrease in the risk score)
- **TP (Total Protein)**: SHAP value = +0.0904 (positive, indicating an increase in the risk score)
- **Alkphos (Alkaline Phosphatase)**: SHAP value = -0.0467 (negative, indicating a decrease in the risk score)
- **Ablation Performance Drops**:
  - Removing **ALB** drops accuracy by +0.1000
  - Removing **TP** drops accuracy by +0.0509
  - Removing **Alkphos** drops accuracy by +0.0263

**Targeted Dietary & Hydration Interventions**
- **Albumin (ALB)**: Increase protein intake to support liver function. Consider lean meats, dairy, and legumes.
- **Total Protein (TP)**: Maintain stable protein levels. Ensure balanced intake of proteins to support overall health.

**Metabolic Tracking & Physical Load Adjustments**
- **Alkaline Phosphatase (Alkphos)**: Monitor and manage any factors contributing to elevated levels, such as bone health or liver function. Consider reducing physical load if necessary.

**Recommended Diagnostic Monitoring Protocols**
- Regular liver function tests to monitor bilirubin, alkaline phosphatase, and other relevant parameters.
- Consult a healthcare provider for further evaluation of elevated bilirubin and alkaline phosphatase levels.

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

- **DB (Direct Bilirubin)**: +0.0764 (Highest SHAP value)
- **Age**: +0.0544 (Second highest SHAP value)
- **ALB (Albumin)**: +0.0523 (Third highest SHAP value)
- **TP (Total Protein)**: +0.0340 (Fourth highest SHAP value)
- **Sgot (SGOT/AST)**: +0.0254 (Fifth highest SHAP value)

- **Ablation -> Removing [DB]** drops accuracy by: +0.1000 (Highest Ablation performance drop)
- **Ablation -> Removing [Age]** drops accuracy by: +0.0712 (Second highest Ablation performance drop)

**Targeted Dietary & Hydration Interventions**
- Increase hydration to support liver function and reduce bilirubin levels.
- Focus on a diet low in fat and high in fiber to manage elevated alkaline phosphatase and bilirubin.
- Consider protein-rich foods to support albumin levels, but monitor total protein intake to avoid overloading the liver.

**Metabolic Tracking & Physical Load Adjustments**
- Regularly monitor liver enzymes and bilirubin levels.
- Adjust physical activity levels to avoid overexertion, especially if SGOT/AST levels are high.
- Monitor and manage weight to reduce metabolic strain on the liver.

**Recommended Diagnostic Monitoring Protocols**
- Schedule regular liver function tests, including bilirubin, alkaline phosphatase, and liver enzyme levels.
- Consider imaging studies such as ultrasound or MRI to assess liver structure and function.
- Monitor albumin and total protein levels to track nutritional status and liver function.

Disclaimer: This is AI generated information for educational purposes only. Always consult a qualified healthcare provider.

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

- **DB (Direct Bilirubin)**: +0.0856 (Highest SHAP value)
- **Alkphos (Alkaline Phosphatase)**: +0.0650 (Second highest SHAP value)
- **TB (Total Bilirubin)**: +0.0429 (Third highest SHAP value)
- **Ablation Sensitivity**: Removing **DB** drops accuracy by +0.1000 (Highest Ablation drop)
- **Ablation Sensitivity**: Removing **Alkphos** drops accuracy by +0.0759 (Second highest Ablation drop)
- **Ablation Sensitivity**: Removing **TB** drops accuracy by +0.0502 (Third highest Ablation drop)

**Targeted Dietary & Hydration Interventions**
- Focus on reducing bilirubin levels through a diet low in red meat and high in vegetables and fruits.
- Increase hydration to support liver function and reduce bilirubin levels.

**Metabolic Tracking & Physical Load Adjustments**
- Monitor and manage physical activity to avoid overloading the liver.
- Regularly track liver enzymes and bilirubin levels to adjust activity levels as needed.

**Recommended Diagnostic Monitoring Protocols**
- Schedule regular liver function tests, including bilirubin, alkaline phosphatase, and liver enzyme levels.
- Consider imaging studies such as ultrasound or MRI to assess liver structure and function.
- Monitor protein levels and albumin/globulin ratio to ensure nutritional support and liver health.

Disclaimer: This is AI generated information for educational purposes only. Always consult a qualified healthcare provider.

---

