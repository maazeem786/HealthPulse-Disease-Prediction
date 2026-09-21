"""Feature extraction module for HealthPulse disease prediction.

Handles feature engineering and extraction from patient data.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts and engineers features from patient symptoms and medical history."""

    SYMPTOM_CATEGORIES = {
        'general': ['fatigue', 'fever', 'mild_fever', 'high_fever', 'weight_loss', 'weight_gain'],
        'pain': ['headache', 'joint_pain', 'muscle_pain', 'back_pain', 'chest_pain',
                 'stomach_pain', 'knee_pain', 'hip_joint_pain', 'belly_pain', 'abdominal_pain'],
        'respiratory': ['cough', 'breathlessness', 'shortness_of_breath', 'wheezing',
                        'sputum', 'phlegm', 'throat_irritation', 'sore_throat'],
        'dermatological': ['skin_rash', 'itching', 'blister', 'red_sore_around_nose',
                          'yellow_crust_ooze', 'ulcer', 'scurring'],
        'gastrointestinal': ['vomiting', 'nausea', 'diarrhea', 'constipation', 'indigestion',
                             'loss_of_appetite', 'bloating', 'cramping'],
        'neurological': ['dizziness', 'loss_of_balance', 'confusion', 'seizures',
                         'numbness', 'tingling', 'slurred_speech', 'loss_of_smell'],
        'cardiovascular': ['chest_pain', 'palpitations', 'fast_heart_rate', 'swollen_legs',
                          'swollen_extremeties', 'shortness_of_breath'],
        'metabolic': ['polyuria', 'polydipsia', 'polyphagia', 'sudden_weight_loss'],
    }

    RISK_FACTORS = {
        'age': {'young': (0, 35), 'middle': (35, 55), 'senior': (55, 120)},
        'bmi': {'underweight': (0, 18.5), 'normal': (18.5, 25), 'overweight': (25, 30), 'obese': (30, 100)},
    }

    def __init__(self) -> None:
        """Initialize the feature extractor."""
        self.feature_names: List[str] = []
        self.category_names: List[str] = list(self.SYMPTOM_CATEGORIES.keys())

    def extract_symptom_counts(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate symptom count per category.

        Args:
            X: Feature DataFrame with symptom columns.

        Returns:
            DataFrame with additional category count features.
        """
        X_enhanced = X.copy()

        for category, symptoms in self.SYMPTOM_CATEGORIES.items():
            available_symptoms = [s for s in symptoms if s in X.columns]
            if available_symptoms:
                X_enhanced[f'{category}_count'] = X[available_symptoms].sum(axis=1)

        return X_enhanced

    def extract_risk_factors(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract and encode risk factor features.

        Args:
            X: Feature DataFrame with demographic columns.

        Returns:
            DataFrame with encoded risk factor features.
        """
        X_enhanced = X.copy()

        if 'age' in X.columns:
            age_bins = [0, 35, 55, 120]
            age_labels = ['young', 'middle', 'senior']
            X_enhanced['age_group'] = pd.cut(X['age'], bins=age_bins, labels=age_labels)
            X_enhanced['age_group'] = X_enhanced['age_group'].map(
                {'young': 0, 'middle': 1, 'senior': 2}
            ).fillna(1).astype(int)

        if 'bmi' in X.columns:
            bmi_bins = [0, 18.5, 25, 30, 100]
            bmi_labels = ['underweight', 'normal', 'overweight', 'obese']
            X_enhanced['bmi_category'] = pd.cut(X['bmi'], bins=bmi_bins, labels=bmi_labels)
            X_enhanced['bmi_category'] = X_enhanced['bmi_category'].map(
                {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
            ).fillna(1).astype(int)

        if 'blood_pressure' in X.columns:
            X_enhanced['bp_risk'] = X['blood_pressure'].apply(
                lambda x: 2 if x == 2 else (1 if x == 1 else 0)
            )

        if 'cholesterol' in X.columns:
            X_enhanced['high_cholesterol'] = (X['cholesterol'] > 240).astype(int)

        if 'family_history' in X.columns:
            X_enhanced['family_history_binary'] = X['family_history'].astype(int)

        return X_enhanced

    def extract_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from symptom patterns.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame with temporal features.
        """
        X_enhanced = X.copy()

        general_symptoms = self.SYMPTOM_CATEGORIES['general']
        available_general = [s for s in general_symptoms if s in X.columns]
        if available_general:
            X_enhanced['general_symptom_count'] = X[available_general].sum(axis=1)

        pain_symptoms = self.SYMPTOM_CATEGORIES['pain']
        available_pain = [s for s in pain_symptoms if s in X.columns]
        if available_pain:
            X_enhanced['pain_symptom_count'] = X[available_pain].sum(axis=1)

        return X_enhanced

    def calculate_severity_score(self, X: pd.DataFrame) -> pd.Series:
        """Calculate overall symptom severity score.

        Args:
            X: Feature DataFrame with symptom columns.

        Returns:
            Series with severity scores.
        """
        symptom_columns = X.select_dtypes(include=[np.number]).columns
        symptom_data = X[symptom_columns]

        severity_weights = {
            'high_fever': 3,
            'chest_pain': 3,
            'breathlessness': 3,
            'confusion': 3,
            'seizures': 3,
            'mild_fever': 2,
            'headache': 2,
            'joint_pain': 2,
            'back_pain': 2,
            'fatigue': 1,
            'cough': 1,
            'nausea': 1,
        }

        scores = pd.Series(0, index=X.index)
        for col in symptom_columns:
            weight = severity_weights.get(col, 1)
            scores += X[col] * weight

        max_possible = sum(severity_weights.values())
        normalized_score = (scores / max_possible) * 100

        return normalized_score

    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between symptoms and demographics.

        Args:
            X: Feature DataFrame.

        Returns:
            DataFrame with interaction features.
        """
        X_enhanced = X.copy()

        if 'age' in X.columns and 'chest_pain' in X.columns:
            X_enhanced['age_chest_pain'] = X['age'] * X['chest_pain']

        if 'bmi' in X.columns and 'shortness_of_breath' in X.columns:
            X_enhanced['bmi_breathlessness'] = X['bmi'] * X['shortness_of_breath']

        if 'age' in X.columns and 'joint_pain' in X.columns:
            X_enhanced['age_joint_pain'] = X['age'] * X['joint_pain']

        return X_enhanced

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature extraction transformations.

        Args:
            X: Raw feature DataFrame.

        Returns:
            Transformed feature DataFrame.
        """
        X_transformed = X.copy()

        X_transformed = self.extract_symptom_counts(X_transformed)
        X_transformed = self.extract_risk_factors(X_transformed)
        X_transformed = self.extract_temporal_features(X_transformed)
        X_transformed = self.create_interaction_features(X_transformed)

        severity_score = self.calculate_severity_score(X_transformed)
        X_transformed['severity_score'] = severity_score

        self.feature_names = X_transformed.columns.tolist()
        logger.info(f"Extracted {len(self.feature_names)} features")

        return X_transformed

    def get_top_risk_factors(
        self,
        importance_dict: Dict[str, float],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top risk factors from model feature importances.

        Args:
            importance_dict: Dictionary mapping feature names to importance values.
            top_n: Number of top factors to return.

        Returns:
            List of dictionaries with risk factor information.
        """
        sorted_factors = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return [
            {'feature': feat, 'importance': imp, 'rank': i + 1}
            for i, (feat, imp) in enumerate(sorted_factors)
        ]