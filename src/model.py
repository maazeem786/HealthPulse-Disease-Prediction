"""Machine learning models for HealthPulse disease prediction.

Implements Random Forest and Gradient Boosting classifiers with ensemble methods.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class DiseaseRiskModel:
    """Ensemble model combining Random Forest and Gradient Boosting for disease risk prediction."""

    def __init__(self) -> None:
        """Initialize the disease risk model with default hyperparameters."""
        self.rf_model: Optional[RandomForestClassifier] = None
        self.gb_model: Optional[GradientBoostingClassifier] = None
        self.is_fitted = False
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.classes_: Optional[np.ndarray] = None

    def _initialize_models(self) -> None:
        """Initialize Random Forest and Gradient Boosting classifiers."""
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> 'DiseaseRiskModel':
        """Train both Random Forest and Gradient Boosting models.

        Args:
            X_train: Training features.
            y_train: Training labels.
            feature_names: Optional list of feature names for importance tracking.

        Returns:
            Self for method chaining.
        """
        self._initialize_models()

        logger.info("Training Random Forest model...")
        self.rf_model.fit(X_train, y_train)

        logger.info("Training Gradient Boosting model...")
        self.gb_model.fit(X_train, y_train)

        self._compute_feature_importance(feature_names)
        self.classes_ = self.rf_model.classes_
        self.is_fitted = True

        train_accuracy = self._evaluate_train_accuracy(X_train, y_train)
        logger.info(f"Training complete. RF accuracy: {train_accuracy['rf']:.4f}, GB accuracy: {train_accuracy['gb']:.4f}")

        return self

    def _evaluate_train_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate training accuracy for both models.

        Args:
            X: Features.
            y: Labels.

        Returns:
            Dictionary with accuracy scores.
        """
        rf_pred = self.rf_model.predict(X)
        gb_pred = self.gb_model.predict(X)

        return {
            'rf': accuracy_score(y, rf_pred),
            'gb': accuracy_score(y, gb_pred)
        }

    def _compute_feature_importance(self, feature_names: Optional[List[str]]) -> None:
        """Compute and normalize feature importances from both models.

        Args:
            feature_names: List of feature names.
        """
        if self.rf_model is None or self.gb_model is None:
            return

        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_

        combined_importance = (rf_importance + gb_importance) / 2

        if feature_names and len(feature_names) == len(combined_importance):
            self.feature_importance_ = dict(zip(feature_names, combined_importance))
        else:
            self.feature_importance_ = {
                f'feature_{i}': imp
                for i, imp in enumerate(combined_importance)
            }

        total = sum(self.feature_importance_.values())
        self.feature_importance_ = {
            k: v / total for k, v in self.feature_importance_.items()
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using ensemble averaging.

        Args:
            X: Feature array.

        Returns:
            Ensemble probability predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        rf_proba = self.rf_model.predict_proba(X)
        gb_proba = self.gb_model.predict_proba(X)

        ensemble_proba = 0.5 * rf_proba + 0.5 * gb_proba

        return ensemble_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using ensemble voting.

        Args:
            X: Feature array.

        Returns:
            Predicted class labels.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation on both models.

        Args:
            X: Feature array.
            y: Label array.
            cv: Number of folds.

        Returns:
            Dictionary with cross-validation metrics.
        """
        if self.rf_model is None or self.gb_model is None:
            raise RuntimeError("Models not initialized. Call fit() first.")

        rf_scores = cross_val_score(self.rf_model, X, y, cv=cv, scoring='accuracy')
        gb_scores = cross_val_score(self.gb_model, X, y, cv=cv, scoring='accuracy')

        return {
            'random_forest': {
                'mean': float(np.mean(rf_scores)),
                'std': float(np.std(rf_scores)),
                'scores': rf_scores.tolist()
            },
            'gradient_boosting': {
                'mean': float(np.mean(gb_scores)),
                'std': float(np.std(gb_scores)),
                'scores': gb_scores.tolist()
            }
        }

    def get_metrics(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics for both models.

        Args:
            X: Feature array.
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary with classification metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y_pred = self.predict(X)
        proba = self.predict_proba(X)

        rf_pred = self.rf_model.predict(X)
        gb_pred = self.gb_model.predict(X)

        metrics = {
            'ensemble': self._compute_metrics(y_true, y_pred, proba),
            'random_forest': self._compute_single_metrics(y_true, rf_pred),
            'gradient_boosting': self._compute_single_metrics(y_true, gb_pred)
        }

        return metrics

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics for ensemble predictions.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            proba: Probability predictions.

        Returns:
            Dictionary with metrics.
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def _compute_single_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics for a single model.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary with metrics.
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def get_classification_report(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> str:
        """Generate classification report.

        Args:
            X: Feature array.
            y_true: True labels.

        Returns:
            Classification report as string.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y_pred = self.predict(X)
        return classification_report(y_true, y_pred, zero_division=0)

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save the model.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        model_data = {
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'feature_importance': self.feature_importance_,
            'classes': self.classes_,
            'is_fitted': self.is_fitted
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> 'DiseaseRiskModel':
        """Load model from disk.

        Args:
            filepath: Path to the saved model.

        Returns:
            Self for method chaining.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.rf_model = model_data['rf_model']
        self.gb_model = model_data['gb_model']
        self.feature_importance_ = model_data['feature_importance']
        self.classes_ = model_data['classes']
        self.is_fitted = model_data['is_fitted']

        logger.info(f"Model loaded from {filepath}")
        return self