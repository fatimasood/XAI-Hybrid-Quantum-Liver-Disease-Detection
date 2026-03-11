# 🧬 Explainable Quantum AI for Liver Disease Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![PennyLane](https://img.shields.io/badge/PennyLane-QuantumML-purple)
![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20Integrated%20Gradients-green)
![Status](https://img.shields.io/badge/Project-Research-blue)

Hybrid **Quantum–Classical Neural Network (HQNN)** for liver disease prediction with **Explainable AI (XAI)** techniques.

This project integrates **Quantum Machine Learning** with **Deep Learning** to build an interpretable model for healthcare diagnosis. The model combines **PennyLane quantum circuits** with classical neural networks and applies **SHAP** and **Integrated Gradients** to explain predictions.

Dataset used: **Indian Liver Patient Dataset (ILPD)**.

---

# 🚀 Key Features

- Hybrid Quantum–Classical Neural Network
- Explainable AI integration
- Medical dataset classification
- Model evaluation and performance metrics
- Feature importance visualization using SHAP

---

# 🛠 Tech Stack

- Python
- TensorFlow / Keras
- PennyLane (Quantum Machine Learning)
- SHAP
- Scikit-Learn
- NumPy / Pandas
- Matplotlib / Seaborn

---

# 📂 Project Structure

```
project/
│
├── data/              # Dataset
├── models/            # Hybrid quantum model
├── xai/               # Explainability methods
├── plots/             # Generated visualizations
│
├── train.py           # Model training
├── evaluate.py        # Model evaluation
└── explainability.py  # XAI analysis
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/fatimasood/XAI-Quantum-Liver-Disease-Detection.git
cd XAI-Quantum-Liver-Disease-Detection
```

Create virtual environment

```bash
python -m venv qml_env
```

Activate environment

Linux / Mac

```bash
source qml_env/bin/activate
```

Windows

```bash
qml_env\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Train the model

```bash
python train.py
```

Evaluate the model

```bash
python evaluate.py
```

Run explainability analysis

```bash
python explainability.py
```

---

# 📊 Model Visualization

<img width="4112" height="1769" alt="Model Architecture" src="https://github.com/user-attachments/assets/4fefe54d-7574-4163-af63-9797c7d10779" />

## 🔍 SHAP Feature Summary

<img width="2254" height="1618" alt="image" src="https://github.com/user-attachments/assets/fb4f2e11-42a0-4967-ba56-db389c8de3bc" />

(All images and results are in their respective folders.....)
---

# 🙏 Acknowledgment

This project is inspired by the research of Dr. Laura María Donaire and aims to bridge the gap between Quantum Computing and Healthcare Transparency.

---


 
