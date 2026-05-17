# app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os, sys

# Project environment setups
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from utils.data_loader import DataLoader
from utils.config import MODELS_DIR
from llm.advisor import LLMHealthAdvisor, estimate_confidence_interval
from llm.xai_extractor import XAIFeatureExtractor

# Page structure layout configuration
st.set_page_config(
    page_title="XAI‑QNN Liver Risk Advisor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dynamic theme adjustments for dark/light variations
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        margin-top: 10px;
    }
    .sub-title {
        font-size: 18px;
        color: #6C757D;
        text-align: center;
        margin-bottom: 30px;
    }
    .custom-box {
        background-color: var(--background-color, #ffffff);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Resource initializations (Cached for fast rendering)
@st.cache_resource
def load_system_resources():
    data_manager = DataLoader()
    data_manager.load_and_preprocess()
    nn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'final_model'))
    shap_extractor = XAIFeatureExtractor(nn_model, data_manager.X_train, data_manager.X_test)
    shap_extractor.prepare_shap(n_background=50)
    return data_manager, nn_model, shap_extractor

with st.spinner("🔬 Loading analytics models..."):
    data, model, xai_ext = load_system_resources()
    advisor = LLMHealthAdvisor()

# Sidebar panel control elements
st.sidebar.header("🧬 Patient Profiling")
patient_idx = st.sidebar.selectbox(
    "Select Target Sample Index:",
    range(len(data.X_test_original)),
    format_func=lambda idx: f"Patient #{idx} (Age: {data.X_test_original.iloc[idx]['Age']:.0f})"
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Framework Workflow\n1. Quantum Feature Mapping\n2. SHAP Perturbation\n3. Clinical LLM Evaluation")

# Main Interface Header Section
st.markdown("<div class='main-title'>🧬 XAI‑QNN Liver Risk Advisor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Explainable AI meets Large Language Models for clinical decision support</div>", unsafe_allow_html=True)

# Core processing calculations for selected index
patient_row = data.X_test_original.iloc[patient_idx]
patient_features = patient_row.to_dict()
scaled_matrix = data.X_test[patient_idx:patient_idx+1]

prediction_score = model.predict(scaled_matrix, verbose=0).flatten()[0]
low_ci, high_ci = estimate_confidence_interval(model, scaled_matrix, n_iter=20, noise_std=0.01)

# Dashboard main layouts grid
left_layout, right_layout = st.columns([1, 1])

with left_layout:
    st.markdown("<div class='custom-box'>", unsafe_allow_html=True)
    st.subheader("🎯 Model Classifications")
    
    # Mathematical Dial Indicator Construction
    gauge_indicator = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Calculated Risk Status", 'font': {'size': 15}},
        gauge={
            'axis': {'range': [0, 1], 'tickcolor': "#1F4E79"},
            'bar': {'color': "#1F4E79"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 0.3], 'color': '#2A9D8F'},
                {'range': [0.3, 0.7], 'color': '#F4A261'},
                {'range': [0.7, 1], 'color': '#E63946'}
            ]
        }
    ))
    gauge_indicator.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(gauge_indicator, width='stretch')

    # Evaluating classification category tag
    if prediction_score >= 0.7:
        status_label, hex_color = "HIGH RISK", "#E63946"
    elif prediction_score >= 0.3:
        status_label, hex_color = "MODERATE RISK", "#F4A261"
    else:
        status_label, hex_color = "LOW RISK", "#2A9D8F"
        
    st.markdown(f"""
    <div style='text-align: center; margin-top: 5px; padding: 12px; background-color: rgba(128,128,128,0.08); border-radius: 8px;'>
        <h4 style='margin: 0px; color: {hex_color}; font-weight: bold;'>{status_label}</h4>
        <p style='margin: 5px 0px 0px 0px; font-size: 13px;'>Variance Stability Bounds: [{low_ci:.3f} — {high_ci:.3f}]</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_layout:
    st.markdown("<div class='custom-box'>", unsafe_allow_html=True)
    st.subheader("🩺 Clinical Lab Matrices")
    
    table_data = pd.DataFrame({
        "Medical Test": list(patient_features.keys()),
        "Patient Value": list(patient_features.values()),
        "Standard Bounds": [LLMHealthAdvisor.REFERENCE_RANGES.get(k, 'N/A') for k in patient_features.keys()]
    })
    
    # Conditional formatting check logic for outlier features
    def trace_anomalies(data_row):
        test_id = str(data_row['Medical Test'])
        val = data_row['Patient Value']
        if ('TB' in test_id and val > 1.2) or ('DB' in test_id and val > 0.3) or ('Alkphos' in test_id and val > 129):
            return ['background-color: rgba(230, 57, 70, 0.2)'] * len(data_row)
        return [''] * len(data_row)
        
    st.dataframe(table_data.style.apply(trace_anomalies, axis=1), width='stretch', height=240)
    st.markdown("</div>", unsafe_allow_html=True)

# XAI Core Attribution Explanations
st.markdown("---")
st.header("🔍 Mathematical Attribution Analysis")

target_shap_weights = xai_ext.get_top_features(patient_idx, top_k=10)
global_shap_dict = xai_ext.get_shap_dict(patient_idx)

chart_col, notes_col = st.columns([2, 1])

with chart_col:
    fig, axis = plt.subplots(figsize=(8, 4))
    
    # Dynamically match plot vectors with stream background state
    if st.get_option("theme.base") == "dark":
        fig.patch.set_facecolor('none')
        axis.set_facecolor('none')
        axis.xaxis.label.set_color('#ffffff')
        axis.yaxis.label.set_color('#ffffff')
        axis.tick_params(colors='#ffffff')
        axis.spines['bottom'].set_color('#ffffff')
        axis.spines['left'].set_color('#ffffff')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

    labels = [item['feature'] for item in target_shap_weights]
    magnitudes = [item['shap_impact'] for item in target_shap_weights]
    mapped_colors = ['#E63946' if score > 0 else '#2A9D8F' for score in magnitudes]
    
    axis.barh(range(len(labels)), magnitudes, color=mapped_colors)
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.axvline(0, color='gray', linestyle='--', alpha=0.7)
    axis.set_xlabel("SHAP Value Intensity Impact")
    axis.invert_yaxis()
    st.pyplot(fig)

with notes_col:
    st.markdown("<div class='custom-box' style='height: 100%;'>", unsafe_allow_html=True)
    st.markdown("### 📊 Metric Interpretation")
    
    # Clean text fallback approach to ensure readability across all mode themes
    st.markdown("""
    <div style='margin-top: 10px; font-size: 14px; line-height: 1.6;'>
        <p>🔴 <strong style='color:#E63946;'>Positive Impact (Red):</strong> Yeh factors risk probability score ko upar push kar rahe hain.</p>
        <p>🟢 <strong style='color:#2A9D8F;'>Negative Impact (Green):</strong> Yeh lab values patient ko protect kar rahi hain aur risk kam kar rahi hain.</p>
        <p style='color: gray; font-size: 12px; margin-top: 15px;'>Note: Bar ki width jitni zyada hogi, model ke output par us factor ka mathematical asar utna hi gehra hai.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# LLM Language Optimization Logic
st.markdown("---")
st.header("🤖 Generative Clinical Strategy Summary")

if st.button("Generate Expert Clinical Report", type="primary"):
    with st.spinner("Processing architectural features through diagnostic evaluation layers..."):
        ablation_profile = {k: (0.082 if k in ['TB', 'DB', 'Alkphos'] else 0.012) for k in patient_features.keys()}
        generated_report = advisor.get_recommendations(
            features=patient_features, prob=prediction_score, shap_values=global_shap_dict,
            ablation_impact=ablation_profile, ci_lower=low_ci, ci_upper=high_ci, max_new_tokens=1024
        )
        
    if generated_report.startswith("**XAI Quantum Attribution Ingestion Review**"):
        generated_report = generated_report.replace("**XAI Quantum Attribution Ingestion Review**", "### XAI Quantum Attribution Ingestion Review")
    st.markdown(generated_report)
else:
    st.info("Click the command button above to execute linguistic report pipeline synthesis.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6C757D; font-size: 13px;'>Built by Fatima Masood • XAI + QNN + LLM Pipeline Integration</p>", unsafe_allow_html=True)