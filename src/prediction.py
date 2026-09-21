"""Prediction pipeline for HealthPulse disease prediction.

Provides high-level API for disease risk prediction on new patients.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor
from src.model import DiseaseRiskModel

logger = logging.getLogger(__name__)


class DiseasePredictor:
    """High-level API for disease risk prediction."""

    SYMPTOM_LIST = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'shivering', 'chills',
        'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
        'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue',
        'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
        'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level',
        'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
        'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
        'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
        'diarrhea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
        'swollen_legs', 'swollen_blood_vessels', 'increased_hunger', 'drying_and_tingling_lips',
        'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
        'stiff_neck', 'swelling_joints', 'swelling_stomach', 'botulism',
        'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'belly_pain',
        'white_scratches_on_throat', 'abnormal_menstruation', 'watering_from_eye',
        'increased_appetite', 'polyuria', 'sudden_weight_loss', 'family_history',
        'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
        'receiving_blood_transfusion', 'coma', 'blood_in_sputum', 'palpitations',
        'inflammatory_nails', 'yellowish_skin', 'fast_heart_rate', 'thyroid',
        'brittle_nails', 'swollen_extremeties', 'diverted_septum', 'snoring',
        'shortness_of_breath', 'chest_pain', 'loss_of_balance', 'unresponsiveness',
        'ulcer', 'extra_marital_contacts', 'internal_itching', 'muscle_pain',
        'digital_fridge', 'debility', 'polyphagia', 'gene_mutation', 'stiff_neck',
        'loss_of_smell', 'nasal_congestion', 'congestion', 'loss_of_taste',
        'coma', 'red_flags', 'scurring', 'foul_smell_ofurine', 'spotting',
        'diabetes', 'obesity', 'cushing', 'fracture', 'swallowing_difficulty',
        'hoarse_voice', 'nodosities', 'history_of_alcohol_consumption', 'fluid_overload',
        'painful_walking', 'phlegm', 'throat_irritation', 'rectal_bleeding',
        'bloating', 'cramping', 'hemorrhoids', 'abdominal_pain', 'swallowing_difficulty'
    ]

    DISEASE_LIST = [
        'Diabetes', 'Heart_Disease', 'Rheumatoid_Arthritis', 'Hypothyroidism',
        'Hypertension', 'COPD', 'Asthma', 'Migraine'
    ]

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize the disease predictor.

        Args:
            model_path: Path to a pre-trained model. If None, a new model will be trained.
        """
        self.model_path = model_path
        self.model: Optional[DiseaseRiskModel] = None
        self.preprocessor: DataPreprocessor = DataPreprocessor()
        self.feature_extractor: FeatureExtractor = FeatureExtractor()
        self._is_initialized = False
        self.feature_names: List[str] = []

    def initialize(self, data_path: Optional[str] = None) -> 'DiseasePredictor':
        """Initialize the prediction pipeline by training or loading a model.

        Args:
            data_path: Path to training data CSV.

        Returns:
            Self for method chaining.
        """
        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Loading model from {self.model_path}")
            self.model = DiseaseRiskModel()
            self.model.load(self.model_path)
            self.feature_names = list(self.model.feature_importance_.keys()) if self.model.feature_importance_ else []
        else:
            logger.info("Training new model...")
            self._train_model(data_path)

        self._is_initialized = True
        return self

    def _train_model(self, data_path: Optional[str] = None) -> None:
        """Train a new model on the provided data.

        Args:
            data_path: Path to training data CSV.
        """
        df = self.preprocessor.load_data(data_path)
        df_clean = self.preprocessor.clean_data(df)
        X, y = self.preprocessor.encode_target(df_clean)

        all_features = list(X.columns)
        self.feature_extractor = FeatureExtractor()
        X_transformed = self.feature_extractor.transform(X)
        self.feature_names = X_transformed.columns.tolist()

        X_scaled = self.preprocessor.scale_features(X_transformed, fit=True)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X_scaled, y)

        self.model = DiseaseRiskModel()
        self.model.fit(X_train, y_train, feature_names=self.feature_names)

        if self.model_path:
            self.model.save(self.model_path)

    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict disease risk for a new patient.

        Args:
            patient_data: Dictionary containing patient information with keys:
                - symptoms: List of symptom strings
                - age: Patient age (int)
                - gender: 'male' or 'female'
                - bmi: Body mass index (float)
                - blood_pressure: 'normal', 'elevated', or 'high'
                - cholesterol: Cholesterol level (float)
                - family_history: List of family disease history

        Returns:
            Dictionary with prediction results including:
                - predictions: List of (disease, probability) tuples
                - top_risk_factors: List of contributing risk factors
                - severity_score: Overall symptom severity
                - recommendations: List of suggested next steps
        """
        if not self._is_initialized:
            self.initialize()

        patient_df = self._prepare_patient_data(patient_data)
        patient_transformed = self.feature_extractor.transform(patient_df)
        
        for col in self.feature_names:
            if col not in patient_transformed.columns:
                patient_transformed[col] = 0
        
        patient_transformed = patient_transformed[self.feature_names]
        patient_scaled = self.preprocessor.scaler.transform(patient_transformed)

        proba = self.model.predict_proba(patient_scaled)[0]
        predictions = self._format_predictions(proba)

        severity_score = self.feature_extractor.calculate_severity_score(patient_transformed).iloc[0]

        top_risk_factors = self._identify_risk_factors(patient_df, proba)

        recommendations = self._generate_recommendations(predictions, severity_score)

        return {
            'predictions': predictions,
            'top_risk_factors': top_risk_factors,
            'severity_score': float(severity_score),
            'recommendations': recommendations
        }

    def _prepare_patient_data(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert patient data to DataFrame format.

        Args:
            patient_data: Raw patient data dictionary.

        Returns:
            DataFrame ready for feature extraction.
        """
        row = {symptom: 0 for symptom in self.SYMPTOM_LIST}

        for symptom in patient_data.get('symptoms', []):
            normalized = symptom.lower().replace(' ', '_')
            if normalized in row:
                row[normalized] = 1

        row['age'] = patient_data.get('age', 40)
        row['gender'] = 0 if patient_data.get('gender') == 'male' else 1
        row['bmi'] = patient_data.get('bmi', 25.0)
        row['blood_pressure'] = {'normal': 0, 'elevated': 1, 'high': 2}.get(
            patient_data.get('blood_pressure', 'normal'), 0
        )
        row['cholesterol'] = patient_data.get('cholesterol', 200)
        row['family_history'] = 1 if patient_data.get('family_history') else 0

        return pd.DataFrame([row])

    def _format_predictions(self, proba: np.ndarray) -> List[Dict[str, Any]]:
        """Format prediction probabilities.

        Args:
            proba: Probability array from model.

        Returns:
            List of dictionaries with disease and probability.
        """
        disease_mapping = self.preprocessor.get_disease_mapping()
        
        predictions = []
        for idx, prob in enumerate(proba):
            disease_name = disease_mapping.get(idx, f"Disease_{idx}")
            predictions.append({
                'disease': disease_name,
                'probability': float(prob),
                'risk_level': self._get_risk_level(prob)
            })

        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return predictions

    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level from probability.

        Args:
            probability: Probability value.

        Returns:
            Risk level string.
        """
        if probability >= 0.7:
            return 'High'
        elif probability >= 0.4:
            return 'Moderate'
        elif probability >= 0.2:
            return 'Low'
        else:
            return 'Minimal'

    def _identify_risk_factors(
        self,
        patient_df: pd.DataFrame,
        proba: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Identify key risk factors for the patient.

        Args:
            patient_df: Patient data DataFrame.
            proba: Prediction probabilities.

        Returns:
            List of risk factor dictionaries.
        """
        if not self.model or not self.model.feature_importance_:
            return []

        risk_factors = []
        top_disease_idx = np.argmax(proba)

        for feature, importance in sorted(
            self.model.feature_importance_.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:
            if feature in patient_df.columns:
                value = patient_df[feature].iloc[0]
                if value == 1 or (isinstance(value, (int, float)) and value != 0):
                    risk_factors.append({
                        'factor': feature,
                        'importance': float(importance),
                        'value': int(value) if isinstance(value, (int, float)) else value
                    })

        return risk_factors[:5]

    def _generate_recommendations(
        self,
        predictions: List[Dict[str, Any]],
        severity_score: float
    ) -> List[str]:
        """Generate recommendations based on predictions.

        Args:
            predictions: Sorted list of disease predictions.
            severity_score: Overall symptom severity.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        top_prediction = predictions[0]
        if top_prediction['probability'] >= 0.6:
            recommendations.append(
                f"High priority: Consult a physician for potential {top_prediction['disease']} assessment"
            )

        if severity_score >= 60:
            recommendations.append(
                "Urgent: Your symptom severity is elevated. Seek medical attention within 24-48 hours"
            )
        elif severity_score >= 30:
            recommendations.append(
                "Schedule an appointment with your primary care provider within the next week"
            )

        if top_prediction['probability'] >= 0.4:
            recommendations.append(
                f"Consider targeted tests for {top_prediction['disease']}: "
                f"Discuss with your doctor appropriate diagnostic procedures"
            )

        recommendations.append(
            "Maintain a symptom diary to track any changes or worsening of conditions"
        )

        return recommendations

    @property
    def is_ready(self) -> bool:
        """Check if predictor is initialized and ready for predictions."""
        return self._is_initialized and self.model is not None