"""HealthPulse Disease Prediction Package.

A comprehensive disease risk prediction system using ensemble ML methods.
"""

from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor
from src.model import DiseaseRiskModel
from src.evaluation import ModelEvaluator
from src.prediction import DiseasePredictor

__version__ = "1.0.0"
__author__ = "Mohd Abdul Azeem"

__all__ = [
    "DataPreprocessor",
    "FeatureExtractor",
    "DiseaseRiskModel",
    "ModelEvaluator",
    "DiseasePredictor",
]