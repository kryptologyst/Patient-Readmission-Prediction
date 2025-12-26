"""Evaluation metrics for patient readmission prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ClinicalMetrics:
    """Clinical evaluation metrics for readmission prediction."""
    
    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize clinical metrics.
        
        Args:
            threshold: Classification threshold.
        """
        self.threshold = threshold
        
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute comprehensive clinical metrics.
        
        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            y_pred: Predicted labels (if None, computed from probabilities).
            
        Returns:
            Dictionary of metrics.
        """
        if y_pred is None:
            y_pred = (y_pred_proba >= self.threshold).astype(int)
            
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
        
        # Clinical metrics
        metrics["sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["specificity"] = self._compute_specificity(y_true, y_pred)
        metrics["ppv"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["npv"] = self._compute_npv(y_true, y_pred)
        
        # Probability-based metrics
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["auc_pr"] = average_precision_score(y_true, y_pred_proba)
        
        # Calibration metrics
        metrics["brier_score"] = self._compute_brier_score(y_true, y_pred_proba)
        metrics["ece"] = self._compute_ece(y_true, y_pred_proba)
        
        # Clinical utility metrics
        metrics["lr_positive"] = self._compute_lr_positive(y_true, y_pred)
        metrics["lr_negative"] = self._compute_lr_negative(y_true, y_pred)
        
        return metrics
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _compute_npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute negative predictive value."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    def _compute_brier_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute Brier score (calibration metric)."""
        return np.mean((y_pred_proba - y_true) ** 2)
    
    def _compute_ece(self, y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def _compute_lr_positive(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute positive likelihood ratio."""
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._compute_specificity(y_true, y_pred)
        return sensitivity / (1 - specificity) if specificity < 1 else np.inf
    
    def _compute_lr_negative(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute negative likelihood ratio."""
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._compute_specificity(y_true, y_pred)
        return (1 - sensitivity) / specificity if specificity > 0 else np.inf


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize model evaluator.
        
        Args:
            threshold: Classification threshold.
        """
        self.threshold = threshold
        self.clinical_metrics = ClinicalMetrics(threshold)
        
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """Evaluate model comprehensively.
        
        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            y_pred: Predicted labels.
            model_name: Name of the model.
            
        Returns:
            Comprehensive evaluation results.
        """
        results = {
            "model_name": model_name,
            "threshold": self.threshold,
            "metrics": self.clinical_metrics.compute_metrics(y_true, y_pred_proba, y_pred),
            "confusion_matrix": self._compute_confusion_matrix(y_true, y_pred_proba, y_pred),
            "classification_report": self._compute_classification_report(y_true, y_pred_proba, y_pred),
        }
        
        return results
    
    def _compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute confusion matrix."""
        if y_pred is None:
            y_pred = (y_pred_proba >= self.threshold).astype(int)
        return confusion_matrix(y_true, y_pred)
    
    def _compute_classification_report(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
    ) -> str:
        """Compute classification report."""
        if y_pred is None:
            y_pred = (y_pred_proba >= self.threshold).astype(int)
        return classification_report(y_true, y_pred, zero_division=0)
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auc_pr = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"PR Curve (AUC = {auc_pr:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot calibration curve."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confusion matrix."""
        cm = self._compute_confusion_matrix(y_true, y_pred_proba, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def create_leaderboard(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a model leaderboard.
    
    Args:
        results: List of evaluation results.
        
    Returns:
        Leaderboard dataframe.
    """
    leaderboard_data = []
    
    for result in results:
        metrics = result["metrics"]
        leaderboard_data.append({
            "Model": result["model_name"],
            "AUC-ROC": metrics["auc_roc"],
            "AUC-PR": metrics["auc_pr"],
            "Sensitivity": metrics["sensitivity"],
            "Specificity": metrics["specificity"],
            "PPV": metrics["ppv"],
            "NPV": metrics["npv"],
            "F1-Score": metrics["f1_score"],
            "Brier Score": metrics["brier_score"],
            "ECE": metrics["ece"],
        })
    
    leaderboard = pd.DataFrame(leaderboard_data)
    leaderboard = leaderboard.sort_values("AUC-ROC", ascending=False)
    
    return leaderboard
