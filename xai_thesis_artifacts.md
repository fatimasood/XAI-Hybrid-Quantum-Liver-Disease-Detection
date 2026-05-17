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

### XAI Quantum Attribution Ingestion Review

The hybrid quantum-classical model has classified the patient as having a **MODERATE RISK** with a target probability of 0.414, within the 95% confidence interval of [0.347 – 0.476]. The SHAP (SHapley Additive exPlanations) values and ablation study results provide insights into which parameters are most influential in the model's decision-making process.

### Targeted Dietary & Hydration Interventions

**SHAP Attribution:**
- **ALB (Albumin):** -0.1774
- **TP (Total Protein):** +0.0904
- **Alkphos (Alkaline Phosphatase):** -0.0467
- **Gender:** -0.0274
- **DB (Direct Bilirubin):** +0.0139
- **A/G (Albumin/Globulin Ratio):** -0.0099
- **Sgpt (Serum Glutamic-Pyruvic Transaminase):** -0.0088
- **Age:** +0.0085
- **Sgot (Serum Glutamic-Oxaloacetic Transaminase):** -0.0055
- **TB (Total Bilirubin):** +0.0053

**Mathematical Prioritization:**
The model prioritizes **Albumin (ALB)** and **Total Protein (TP)** due to their significant SHAP values. The negative SHAP value for ALB (-0.1774) indicates that a decrease in albumin levels is associated with a higher risk, while the positive SHAP value for TP (+0.0904) suggests that an increase in total protein levels is beneficial. The model also considers **Alkaline Phosphatase (Alkphos)** and **Direct Bilirubin (DB)**, with negative and positive SHAP values, respectively, indicating their influence on the risk assessment.

**Dietary & Hydration Recommendations:**
- **Increase Protein Intake:** Consume more high-quality proteins such as lean meats, fish, eggs, and dairy products to support liver function and overall health.
- **Hydration:** Maintain adequate hydration by drinking at least 8-10 glasses of water daily to support liver detoxification.
- **Limit Alcohol and Avoid Harmful Substances:** Reduce alcohol consumption and avoid exposure to harmful substances that can further stress the liver.

### Metabolic Tracking & Physical Load Adjustments

**SHAP Attribution:**
- **Age:** +0.0085
- **Sgot (Serum Glutamic-Oxaloacetic Transaminase):** -0.0055
- **TB (Total Bilirubin):** +0.0053

**Mathematical Prioritization:**
The model considers **Age** as a minor factor, with a positive SHAP value, indicating that age is a slight risk factor. **Total Bilirubin (TB)** has a positive SHAP value (+0.0053), suggesting that elevated levels of bilirubin are associated with a higher risk. **SGOT (Serum Glutamic-Oxaloacetic Transaminase)** has a negative SHAP value (-0.0055), indicating that lower levels of SGOT are beneficial.

**Metabolic Tracking & Physical Load Adjustments:**
- **Regular Monitoring:** Regularly track liver function tests, including bilirubin levels, to monitor any changes.
- **Physical Activity:** Engage in moderate physical activity, such as walking or light exercise, to support overall health without overloading the liver.
- **Avoid Overexertion:** Avoid strenuous physical activities that could increase liver stress.

### Recommended Diagnostic Monitoring Protocols

**SHAP Attribution:**
- **ALB (Albumin):** -0.1774
- **TP (Total Protein):** +0.0904
- **Alkphos (Alkaline Phosphatase):** -0.0467
- **DB (Direct Bilirubin):** +0.0139
- **A/G (Albumin/Globulin Ratio):** -0.0099
- **Sgpt (Serum Glutamic-Pyruvic Transaminase):** -0.0088
- **Age:** +0.0085
- **Sgot (Serum Glutamic-Oxaloacetic Transaminase):** -0.0055
- **TB (Total Bilirubin):** +0.0053

**Mathematical Prioritization:**
The model prioritizes **Albumin (ALB)**, **Total Protein (TP)**, **Alkaline Phosphatase (Alkphos)**, and **Direct Bilirubin (DB)** due to their significant SHAP values. The negative SHAP values for ALB and Alkphos, and the positive SHAP values for TP and DB, indicate the importance of these parameters in the risk assessment.

**Diagnostic Monitoring Protocols:**
- **Regular Liver Function Tests:** Schedule regular blood tests to monitor liver enzymes, bilirubin levels, and albumin levels.
- **Ultrasound or Imaging:** Consider periodic liver imaging to assess liver structure and detect any abnormalities.
- **Consultation with a Specialist:** Regular follow-ups with a hepatologist or a gastroenterologist to monitor liver health and adjust treatment as necessary.

### Disclaimer: 
This is AI-generated information for educational purposes only. Always consult a qualified healthcare provider.

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

### XAI Quantum Attribution Ingestion Review

The hybrid quantum-classical model has identified a high-risk profile for the patient with a target probability of 0.856, indicating a significant risk level. The SHAP (SHapley Additive exPlanations) values and ablation study results provide insights into which parameters are most influential in this high-risk classification.

#### SHAP Attribution Analysis
- **DB (Direct Bilirubin): +0.0764** - The highest SHAP value indicates that direct bilirubin is the most influential factor in the high-risk classification.
- **Age: +0.0544** - Age is the second most influential factor.
- **ALB (Albumin): +0.0523** - Albumin is the third most influential factor.
- **TP (Total Protein): +0.0340** - Total protein is the fourth most influential factor.
- **TB (Total Bilirubin): +0.0271** - Total bilirubin is the fifth most influential factor.
- **SGOT (AST): +0.0254** - SGOT (AST) is the sixth most influential factor.
- **Gender: +0.0105** - Gender is the least influential factor among the top contributors.
- **A/G (Albumin/Globulin Ratio): +0.0083** - The albumin/globulin ratio is the least influential factor among the top contributors.
- **Alkphos (Alkaline Phosphatase): -0.0047** - Alkaline phosphatase has a negative influence.
- **Sgpt (ALT): +0.0010** - SGPT (ALT) has a minimal positive influence.

#### Ablation Study Feature Sensitivities
- **Removing DB (Direct Bilirubin) drops accuracy by: +0.1000** - This indicates that direct bilirubin is the most critical factor in the model's accuracy.
- **Removing Age drops accuracy by: +0.0712** - Age is the second most critical factor.
- **Removing ALB (Albumin) drops accuracy by: +0.0684** - Albumin is the third most critical factor.
- **Removing TB (Total Bilirubin) drops accuracy by: +0.0271** - Total bilirubin is the fourth most critical factor.
- **Removing SGOT (AST) drops accuracy by: +0.0254** - SGOT (AST) is the fifth most critical factor.

### Targeted Dietary & Hydration Interventions
- **Direct Bilirubin (DB):** The high direct bilirubin level suggests potential liver issues. Dietary interventions should focus on reducing fat intake and increasing fiber to support liver function. Hydration is crucial to help flush out toxins.
- **Total Bilirubin (TB):** The elevated total bilirubin level also indicates liver concerns. A diet rich in antioxidants and low in processed foods can help. Adequate hydration is essential.
- **Albumin (ALB):** Low albumin levels suggest malnutrition or liver dysfunction. A diet high in protein, particularly from lean meats, fish, and plant-based sources, is recommended. Ensure adequate hydration.

### Metabolic Tracking & Physical Load Adjustments
- **Alkaline Phosphatase (Alkphos):** Elevated levels may indicate liver or bone issues. Regular monitoring of liver function is advised. Physical activity should be moderate and tailored to the patient's current health status.
- **SGOT (AST):** Elevated SGOT levels suggest liver damage. Monitoring physical activity to avoid overexertion is important. Regular exercise can help improve liver function, but it should be balanced with rest.

### Recommended Diagnostic Monitoring Protocols
- **Liver Function Tests:** Regular monitoring of liver enzymes (ALT, AST, ALP, and bilirubin) is essential.
- **Ultrasound:** To assess liver structure and detect any abnormalities.
- **Complete Blood Count (CBC):** To monitor overall health and detect any signs of infection or anemia.
- **Nutritional Assessment:** Regular assessments to ensure adequate intake of essential nutrients, particularly protein and vitamins.

### Disclaimer: 
This is AI-generated information for educational purposes only. Always consult a qualified healthcare provider.

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

### XAI Quantum Attribution Ingestion Review

The hybrid quantum-classical model has identified a high risk with a target probability of 0.857, indicating a significant likelihood of a critical health condition. The SHAP (SHapley Additive exPlanations) values and ablation study results provide insights into which parameters are most influential in this high-risk classification.

#### SHAP Attribution Analysis
- **DB (Direct Bilirubin): +0.0856** - The highest SHAP value indicates that direct bilirubin is the most influential factor in the high-risk classification.
- **Alkphos (Alkaline Phosphatase): +0.0650** - Alkaline phosphatase is the second most influential factor.
- **TB (Total Bilirubin): +0.0429** - Total bilirubin is the third most influential factor.
- **ALB (Albumin): +0.0400** - Albumin is the fourth most influential factor.
- **TP (Total Protein): +0.0297** - Total protein is the fifth most influential factor.
- **A/G (Albumin/Globulin Ratio): +0.0250** - Albumin/globulin ratio is the sixth most influential factor.
- **Gender: -0.0055** - Gender has a slight negative influence.
- **Age: -0.0041** - Age has a slight negative influence.
- **Sgpt (SGPT/ALT): +0.0040** - SGPT/ALT has a slight positive influence.
- **Sgot (SGOT/AST): +0.0027** - SGOT/AST has a slight positive influence.

#### Ablation Study Feature Sensitivities
- **Removing DB (Direct Bilirubin): +0.1000** - The highest drop in accuracy indicates that direct bilirubin is the most critical parameter.
- **Removing Alkphos (Alkaline Phosphatase): +0.0759** - Alkaline phosphatase is the second most critical parameter.
- **Removing TB (Total Bilirubin): +0.0502** - Total bilirubin is the third most critical parameter.

### Targeted Dietary & Hydration Interventions
- **Direct Bilirubin (DB):** The high levels of direct bilirubin suggest potential liver issues. Dietary interventions should focus on reducing foods high in fat and cholesterol, which can exacerbate liver conditions. Increase intake of foods rich in antioxidants such as leafy greens, berries, and nuts.
- **Alkaline Phosphatase (Alkphos):** Elevated levels of alkaline phosphatase can indicate liver or bone issues. A diet rich in lean proteins, whole grains, and vegetables can help manage this. Avoid excessive intake of fatty foods and alcohol.
- **Total Bilirubin (TB):** High total bilirubin levels are a critical concern. Focus on a diet that is low in fat and high in fiber, including plenty of fruits, vegetables, and whole grains. Stay well-hydrated to aid in liver function.

### Metabolic Tracking & Physical Load Adjustments
- **Metabolic Tracking:** Regular monitoring of liver function tests is essential. Track changes in bilirubin, alkaline phosphatase, and other liver enzymes. Adjust physical activity levels based on metabolic tracking results. Avoid strenuous activities that may strain the liver.
- **Physical Load Adjustments:** Reduce physical load to avoid exacerbating liver conditions. Engage in moderate activities such as walking, swimming, or cycling. Avoid heavy lifting and high-impact exercises.

### Recommended Diagnostic Monitoring Protocols
- **Liver Function Tests:** Regularly monitor liver function tests, including bilirubin, alkaline phosphatase, SGPT, and SGOT levels.
- **Imaging Studies:** Consider imaging studies such as ultrasound or MRI to assess liver structure and function.
- **Genetic Testing:** If there is a family history of liver disease, consider genetic testing to identify any hereditary conditions.
- **Consultation with Specialists:** Regular consultations with hepatologists and other specialists are recommended to manage and monitor liver health.

### Disclaimer: 
This is AI-generated information for educational purposes only. Always consult a qualified healthcare provider.

---

