"""Data preprocessing module for HealthPulse disease prediction.

Handles data loading, cleaning, and preparation for model training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations for the disease prediction system."""

    SYMPTOMS = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'shivering', 'chills',
        'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
        'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue',
        'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
        'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level',
        'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
        'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
        'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
        'diahhrea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
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

    DISEASES = [
        'Diabetes', 'Heart_Disease', 'Rheumatoid_Arthritis', 'Hypothyroidism',
        'Hypertension', 'COPD', 'Asthma', 'Migraine'
    ]

    def __init__(self, data_path: Optional[str] = None) -> None:
        """Initialize the preprocessor with optional data path.

        Args:
            data_path: Path to the CSV file containing training data.
        """
        self.data_path = data_path
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self._is_fitted = False

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load data from CSV file or generate synthetic data.

        Args:
            file_path: Path to CSV file. If None, generates synthetic data.

        Returns:
            DataFrame containing the loaded data.
        """
        if file_path:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            df = pd.read_csv(path)
            logger.info(f"Loaded data from {file_path}: {df.shape[0]} records, {df.shape[1]} features")
            return df

        df = self._generate_synthetic_data(12000)
        logger.info(f"Generated synthetic data: {df.shape[0]} records, {df.shape[1]} features")
        return df

    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate realistic synthetic patient data for demonstration.

        Args:
            n_samples: Number of patient records to generate.

        Returns:
            DataFrame with synthetic patient data.
        """
        np.random.seed(42)

        data: Dict[str, Any] = {
            'patient_id': [f"P{i:05d}" for i in range(1, n_samples + 1)],
            'age': np.random.randint(18, 85, n_samples),
            'gender': np.random.choice(['male', 'female'], n_samples),
            'bmi': np.random.normal(26, 5, n_samples).clip(15, 50),
            'blood_pressure': np.random.choice(['normal', 'elevated', 'high'], n_samples, p=[0.5, 0.25, 0.25]),
            'cholesterol': np.random.normal(200, 40, n_samples).clip(100, 350),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        }

        for symptom in self.SYMPTOMS[:50]:
            prob = np.random.uniform(0.02, 0.25)
            data[symptom] = np.random.choice([0, 1], n_samples, p=[1 - prob, prob])

        disease_weights = {
            'Diabetes': 0.15,
            'Heart_Disease': 0.12,
            'Rheumatoid_Arthritis': 0.08,
            'Hypothyroidism': 0.10,
            'Hypertension': 0.20,
            'COPD': 0.08,
            'Asthma': 0.12,
            'Migraine': 0.15,
        }

        disease_probs = np.array(list(disease_weights.values()))
        disease_probs /= disease_probs.sum()

        data['disease'] = np.random.choice(self.DISEASES, n_samples, p=disease_probs)

        df = pd.DataFrame(data)
        df = self._add_disease_correlated_symptoms(df)

        return df

    def _add_disease_correlated_symptoms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add symptoms that are correlated with specific diseases.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with disease-correlated symptoms added.
        """
        symptom_disease_map = {
            'Diabetes': ['polyphagia', 'polyuria', 'sudden_weight_loss', 'fatigue'],
            'Heart_Disease': ['chest_pain', 'shortness_of_breath', 'palpitations', 'fatigue'],
            'Rheumatoid_Arthritis': ['joint_pain', 'stiff_neck', 'swelling_joints', 'muscle_pain'],
            'Hypothyroidism': ['weight_gain', 'cold_hands_and_feets', 'fatigue', 'constipation'],
            'Hypertension': ['headache', 'dizziness', 'blurred_vision', 'fatigue'],
            'COPD': ['cough', 'breathlessness', 'wheezing', 'sputum'],
            'Asthma': ['wheezing', 'shortness_of_breath', 'chest_tightness', 'cough'],
            'Migraine': ['headache', 'nausea', 'sensitivity_to_light', 'vomiting'],
        }

        for disease, symptoms in symptom_disease_map.items():
            mask = df['disease'] == disease
            for symptom in symptoms:
                if symptom in df.columns:
                    prob_change = np.random.uniform(0.15, 0.35)
                    change_indices = np.random.choice(
                        df[mask].index,
                        size=int(df[mask].sum() * prob_change),
                        replace=False
                    )
                    df.loc[change_indices, symptom] = 1

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the input data.

        Args:
            df: Raw DataFrame to clean.

        Returns:
            Cleaned DataFrame.
        """
        df_clean = df.copy()

        df_clean['bmi'] = df_clean['bmi'].fillna(df_clean['bmi'].median())
        df_clean['age'] = df_clean['age'].clip(0, 120)
        df_clean['gender'] = df_clean['gender'].map({'male': 0, 'female': 1})
        df_clean['blood_pressure'] = df_clean['blood_pressure'].map({
            'normal': 0, 'elevated': 1, 'high': 2
        })

        numeric_cols = ['age', 'bmi', 'cholesterol']
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        missing_threshold = 0.3
        cols_to_drop = []
        for col in df_clean.columns:
            if df_clean[col].isna().mean() > missing_threshold:
                cols_to_drop.append(col)

        df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')

        logger.info(f"Cleaned data: {df_clean.shape[0]} records, {df_clean.shape[1]} features")
        return df_clean

    def encode_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Encode the target variable.

        Args:
            df: DataFrame with target column.

        Returns:
            Tuple of (features DataFrame, encoded target array).
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()

        y = self.label_encoder.fit_transform(df['disease'])
        X = df.drop(columns=['disease', 'patient_id'], errors='ignore')

        self.feature_names = X.columns.tolist()
        logger.info(f"Target encoded: {len(self.label_encoder.classes_)} classes")

        return X, y

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale numerical features.

        Args:
            X: Feature DataFrame.
            fit: Whether to fit the scaler or transform only.

        Returns:
            Scaled feature array.
        """
        numerical_cols = ['age', 'bmi', 'cholesterol']

        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self._is_fitted = True
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fitted. Call scale_features with fit=True first.")
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets.

        Args:
            X: Feature array.
            y: Target array.
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        logger.info(f"Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test

    def get_disease_mapping(self) -> Dict[int, str]:
        """Get the disease label encoding mapping.

        Returns:
            Dictionary mapping encoded integers to disease names.
        """
        if self.label_encoder is None:
            raise RuntimeError("Label encoder not fitted. Run encode_target first.")
        return {i: name for i, name in enumerate(self.label_encoder.classes_)}

    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted."""
        return self._is_fitted