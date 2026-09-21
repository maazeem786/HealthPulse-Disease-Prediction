"""Streamlit dashboard for HealthPulse disease risk assessment.

A user-friendly interface for predicting disease risk from symptoms.
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.prediction import DiseasePredictor
from src.data_preprocessing import DataPreprocessor

st.set_page_config(
    page_title="HealthPulse - Disease Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
    }
    .moderate-risk {
        background-color: #fff3e0;
        border-left: 4px solid #f57c00;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 4px solid #388e3c;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #999;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


def initialize_predictor():
    """Initialize the disease predictor."""
    if 'predictor' not in st.session_state:
        with st.spinner("Initializing HealthPulse model..."):
            predictor = DiseasePredictor()
            predictor.initialize()
            st.session_state.predictor = predictor
    return st.session_state.predictor


def get_symptom_categories():
    """Get organized symptom categories for the UI."""
    return {
        "General Symptoms": [
            "fatigue", "weight_gain", "weight_loss", "mild_fever", "high_fever",
            "chills", "shivering", "restlessness", "lethargy", "debility"
        ],
        "Pain Symptoms": [
            "headache", "joint_pain", "muscle_pain", "back_pain", "chest_pain",
            "stomach_pain", "knee_pain", "hip_joint_pain", "belly_pain", "abdominal_pain",
            "pain_behind_the_eyes"
        ],
        "Respiratory Symptoms": [
            "cough", "breathlessness", "shortness_of_breath", "wheezing",
            "sputum", "phlegm", "throat_irritation", "sore_throat"
        ],
        "Gastrointestinal Symptoms": [
            "vomiting", "nausea", "diarrhea", "constipation", "indigestion",
            "loss_of_appetite", "bloating", "cramping", "acidity"
        ],
        "Skin Symptoms": [
            "skin_rash", "itching", "blister", "yellowish_skin", "ulcer",
            "nodal_skin_eruptions", "red_sore_around_nose", "yellow_crust_ooze"
        ],
        "Cardiovascular Symptoms": [
            "chest_pain", "palpitations", "fast_heart_rate", "swollen_legs",
            "swollen_extremeties", "sweating", "breathlessness"
        ],
        "Neurological Symptoms": [
            "dizziness", "loss_of_balance", "confusion", "seizures",
            "numbness", "tingling", "slurred_speech", "loss_of_smell", "coma"
        ],
        "Metabolic Symptoms": [
            "polyuria", "polydipsia", "polyphagia", "sudden_weight_loss",
            "increased_hunger", "irregular_sugar_level"
        ],
        "Musculoskeletal": [
            "muscle_wasting", "muscle_weakness", "stiff_neck", "swelling_joints",
            "painful_walking", "joint_pain", "muscle_pain"
        ]
    }


def main():
    """Main application entry point."""
    st.markdown('<h1 class="main-header">🏥 HealthPulse</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Disease Risk Assessment from Symptoms</p>',
        unsafe_allow_html=True
    )

    predictor = initialize_predictor()
    symptom_categories = get_symptom_categories()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 👤 Patient Information")

        age = st.slider("Age", 1, 100, 40)
        gender = st.radio("Gender", ["male", "female"], horizontal=True)

        bmi = st.number_input(
            "BMI (Body Mass Index)",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.5
        )

        bp_options = {"Normal": "normal", "Elevated": "elevated", "High": "high"}
        blood_pressure = st.selectbox(
            "Blood Pressure",
            options=list(bp_options.keys()),
            index=0
        )
        blood_pressure_value = bp_options[blood_pressure]

        cholesterol = st.number_input(
            "Cholesterol Level (mg/dL)",
            min_value=100,
            max_value=400,
            value=200,
            step=5
        )

        family_history_input = st.multiselect(
            "Family History of Diseases",
            options=DataPreprocessor.DISEASES,
            default=[]
        )

    with col2:
        st.markdown("### 🩺 Symptoms")

        selected_symptoms = []

        for category, symptoms in symptom_categories.items():
            with st.expander(f"{category}", expanded=False):
                for symptom in symptoms:
                    if st.checkbox(
                        symptom.replace('_', ' ').title(),
                        key=f"symptom_{symptom}",
                        value=False
                    ):
                        selected_symptoms.append(symptom)

        st.markdown(f"**{len(selected_symptoms)} symptoms selected**")

    if st.button("🔍 Analyze Disease Risk", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("Please select at least one symptom for analysis.")
        else:
            patient_data = {
                'symptoms': selected_symptoms,
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'blood_pressure': blood_pressure_value,
                'cholesterol': cholesterol,
                'family_history': family_history_input
            }

            with st.spinner("Analyzing symptoms and calculating risk scores..."):
                try:
                    result = predictor.predict_risk(patient_data)

                    st.markdown("---")
                    st.markdown("### 📊 Risk Assessment Results")

                    col_pred1, col_pred2, col_pred3 = st.columns(3)

                    with col_pred1:
                        top_disease = result['predictions'][0]
                        st.metric(
                            "Primary Risk",
                            top_disease['disease'].replace('_', ' '),
                            f"{top_disease['probability']:.1%}"
                        )

                    with col_pred2:
                        severity = result['severity_score']
                        severity_label = "High" if severity >= 60 else "Moderate" if severity >= 30 else "Low"
                        st.metric("Severity Score", f"{severity:.1f}", severity_label)

                    with col_pred3:
                        second_disease = result['predictions'][1]
                        st.metric(
                            "Secondary Risk",
                            second_disease['disease'].replace('_', ' '),
                            f"{second_disease['probability']:.1%}"
                        )

                    st.markdown("### 🏆 All Disease Risks")

                    for pred in result['predictions']:
                        disease_name = pred['disease'].replace('_', ' ')
                        prob = pred['probability']
                        risk = pred['risk_level']

                        risk_class = "high-risk" if risk == "High" else "moderate-risk" if risk == "Moderate" else "low-risk"

                        col_left, col_right = st.columns([4, 1])

                        with col_left:
                            progress_bar = st.progress(prob)
                            st.markdown(f"{disease_name} ({prob:.1%})")

                        with col_right:
                            color = "🔴" if risk == "High" else "🟡" if risk == "Moderate" else "🟢"
                            st.markdown(f"{color} {risk}")

                    st.markdown("### ⚠️ Top Contributing Risk Factors")

                    if result['top_risk_factors']:
                        for i, factor in enumerate(result['top_risk_factors'], 1):
                            factor_name = factor['factor'].replace('_', ' ').title()
                            st.markdown(f"{i}. **{factor_name}** - Importance: {factor['importance']:.3f}")
                    else:
                        st.info("No significant risk factors identified based on symptoms.")

                    st.markdown("### 💡 Recommendations")

                    for i, rec in enumerate(result['recommendations'], 1):
                        st.markdown(f"{i}. {rec}")

                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    st.info("Please try again with different symptoms.")

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer</strong><br>
        HealthPulse is designed as a decision-support tool and should NOT replace professional medical advice. 
        Always consult qualified healthcare providers for diagnosis and treatment. 
        The predictions provided are based on statistical patterns and should be used alongside clinical judgment.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()