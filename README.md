
# 🧬 Explainable Quantum AI for Liver Disease Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![PennyLane](https://img.shields.io/badge/PennyLane-QuantumML-purple)
![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20Integrated%20Gradients-green)
![Status](https://img.shields.io/badge/Project-Research-blue)

> Bridging **Quantum Machine Learning** and **Clinical Transparency** using Explainable AI.

This repository presents a **Hybrid Quantum–Classical Neural Network (HQCNN)** designed for **liver disease detection** using the **Indian Liver Patient Dataset (ILPD)**.

Unlike traditional black-box models, this project integrates **Explainable AI (XAI)** techniques and validates them using a **Feature Ablation Study** to ensure the model truly relies on meaningful clinical biomarkers.

The objective is to demonstrate how **Quantum Machine Learning combined with Explainable AI** can produce **transparent and trustworthy healthcare prediction systems**.

---

# 🌟 Key Innovations

### Hybrid Quantum Architecture

A **PennyLane variational quantum circuit** is integrated with a classical neural network built using **TensorFlow/Keras**, creating a hybrid model capable of learning complex nonlinear relationships within clinical data.

### Multi-Method Explainability

This project applies multiple XAI methods to interpret model predictions:

* Kernel SHAP
* Integrated Gradients
* Permutation Feature Importance

Using several explanation methods ensures that feature importance results are **consistent and reliable**.

### Scientific Validation via Ablation Study

A **feature ablation pipeline** validates explainability results.

After identifying the most important clinical features using SHAP, these features are removed from the dataset and the model is retrained.

A noticeable performance drop confirms that the model is **actually learning from medically relevant variables** rather than spurious correlations.

### Clinically Relevant Insights

The model identifies **Direct Bilirubin (DB)** and **Alkaline Phosphotase (Alkphos)** as dominant predictors of liver disease, which aligns with known clinical indicators.

---

# 📊 Model Performance

| Metric                 | Result     |
| ---------------------- | ---------- |
| Baseline Accuracy      | **80.14%** |
| Recall (Disease Class) | **75%**    |
| AUC Score              | **~0.85**  |

The training process demonstrates **stable convergence** with minimal overfitting, indicated by similar training and validation curves.

---

# 🔬 XAI Validation (Ablation Study)

Explainability results are validated using **feature removal experiments**.

| Experiment                  | Accuracy |
| --------------------------- | -------- |
| Baseline Model              | **80%**  |
| After Removing Top Features | **~72%** |

A **7–8% decrease in accuracy** confirms that the removed features were genuinely important to the model’s predictions.

---

# 📊 Visual Results

## SHAP Feature Importance


## Ablation Study Results

---

# 🛠 Tech Stack

* Python
* TensorFlow / Keras
* PennyLane (Quantum Machine Learning)
* SHAP
* Scikit-Learn
* NumPy / Pandas
* Matplotlib / Seaborn

---

# 📂 Project Structure

```
XAI-Quantum-Liver-Disease-Detection/

data/
│   Liver Patient Dataset (ILPD)

models/
│   Hybrid Quantum-Classical Model

xai/
│   SHAP analysis
│   Integrated Gradients
│   Permutation Importance

validation/
│   Feature Ablation Study

plots/
│   Generated visualizations

results/
│   Metrics and experiment outputs

train.py
evaluate.py
explainability.py
```

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/fatimasood/XAI-Quantum-Liver-Disease-Detection.git
cd XAI-Quantum-Liver-Disease-Detection
```

Create virtual environment

```
python -m venv qml_env
```

Activate environment

Linux / Mac

```
source qml_env/bin/activate
```

Windows

```
qml_env\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Train the hybrid quantum model

```
python train.py
```

Evaluate the trained model

```
python evaluate.py
```

Generate explainability analysis

```
python explainability.py
```

---

# 🎯 Research Contribution

This project demonstrates how **Explainable AI can be integrated with Quantum Machine Learning** to develop **transparent healthcare prediction systems**.

The inclusion of a **feature ablation validation pipeline** ensures that model explanations are **scientifically validated rather than purely visual interpretations**.

---

LinkedIn: *https://www.linkedin.com/in/fatimamasoodfm/*

---

# ⚠️ Disclaimer

This project is intended for **research and educational purposes only** and should **not be used as a clinical diagnostic system** without medical validation.
