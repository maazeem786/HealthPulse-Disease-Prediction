"""Evaluation module for HealthPulse disease prediction.

Provides comprehensive model evaluation with visualizations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics and visualizations."""

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        """Initialize evaluator with class names.

        Args:
            class_names: List of class label names.
        """
        self.class_names = class_names or [
            'Diabetes', 'Heart_Disease', 'Rheumatoid_Arthritis', 'Hypothyroidism',
            'Hypertension', 'COPD', 'Asthma', 'Migraine'
        ]
        self.evaluation_results: Dict[str, Any] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Ensemble"
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.
            model_name: Name of the model being evaluated.

        Returns:
            Dictionary with evaluation results.
        """
        results = {
            'model_name': model_name,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(
                y_true, y_pred, target_names=self.class_names, output_dict=True, zero_division=0
            ),
            'accuracy': float((y_true == y_pred).mean()),
        }

        try:
            roc_data = self._compute_roc_curves(y_true, y_proba)
            results['roc_curves'] = roc_data
            results['roc_auc'] = self._compute_macro_roc_auc(y_true, y_proba)
        except Exception as e:
            logger.warning(f"Could not compute ROC curves: {e}")

        try:
            pr_data = self._compute_precision_recall_curves(y_true, y_proba)
            results['precision_recall'] = pr_data
        except Exception as e:
            logger.warning(f"Could not compute PR curves: {e}")

        self.evaluation_results[model_name] = results
        return results

    def _compute_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Compute ROC curves for each class.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.

        Returns:
            Dictionary with ROC curve data.
        """
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        roc_data = {}
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            roc_data[class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(roc_auc)
            }

        return roc_data

    def _compute_macro_roc_auc(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> float:
        """Compute macro-averaged ROC AUC.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.

        Returns:
            Macro-averaged ROC AUC score.
        """
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        auc_scores = []
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                auc_scores.append(auc(fpr, tpr))
            except Exception:
                auc_scores.append(0.5)

        return float(np.mean(auc_scores))

    def _compute_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Compute precision-recall curves for each class.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.

        Returns:
            Dictionary with PR curve data.
        """
        n_classes = len(self.class_names)
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        pr_data = {}
        for i, class_name in enumerate(self.class_names):
            precision, recall, thresholds = precision_recall_curve(
                y_true_bin[:, i], y_proba[:, i]
            )
            avg_precision = average_precision_score(y_true_bin[:, i], y_proba[:, i])

            pr_data[class_name] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist() if len(thresholds) > 0 else [],
                'average_precision': float(avg_precision)
            }

        return pr_data

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Plot confusion matrix heatmap.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path: Optional path to save the figure.
            figsize: Figure size.

        Returns:
            Matplotlib figure object.
        """
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix - Disease Prediction', fontsize=14, fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curves(
        self,
        roc_data: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Plot ROC curves for all classes.

        Args:
            roc_data: ROC curve data from evaluation.
            save_path: Optional path to save the figure.
            figsize: Figure size.

        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)

        for class_name, data in roc_data.items():
            fpr = data['fpr']
            tpr = data['tpr']
            roc_auc = data['auc']

            ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Multi-Class Disease Prediction', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 15,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """Plot top N feature importances.

        Args:
            feature_importance: Dictionary mapping feature names to importance.
            top_n: Number of top features to display.
            save_path: Optional path to save the figure.
            figsize: Figure size.

        Returns:
            Matplotlib figure object.
        """
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        features, importances = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(range(len(features)), importances, color='steelblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Risk Factors', fontsize=14, fontweight='bold')

        for i, (feat, imp) in enumerate(sorted_features):
            ax.annotate(
                f'{imp:.3f}',
                xy=(imp, i),
                xytext=(5, 0),
                textcoords='offset points',
                va='center',
                fontsize=9
            )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_per_class_metrics(
        self,
        classification_report: Dict[str, Any],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """Plot precision, recall, and F1-score per class.

        Args:
            classification_report: Classification report dictionary.
            save_path: Optional path to save the figure.
            figsize: Figure size.

        Returns:
            Matplotlib figure object.
        """
        classes = [c for c in classification_report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        
        metrics = {
            'Precision': [classification_report[c]['precision'] for c in classes],
            'Recall': [classification_report[c]['recall'] for c in classes],
            'F1-Score': [classification_report[c]['f1-score'] for c in classes]
        }

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(classes))
        width = 0.25

        for i, (metric_name, values) in enumerate(metrics.items()):
            ax.bar(x + i * width, values, width, label=metric_name)

        ax.set_xlabel('Disease', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Per-class metrics plot saved to {save_path}")

        return fig

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary of all evaluation results.

        Returns:
            Dictionary with summary metrics.
        """
        if not self.evaluation_results:
            return {}

        summary = {}
        for model_name, results in self.evaluation_results.items():
            summary[model_name] = {
                'accuracy': results.get('accuracy', 0),
                'roc_auc': results.get('roc_auc', 0),
                'classification_report': results.get('classification_report', {})
            }

        return summary