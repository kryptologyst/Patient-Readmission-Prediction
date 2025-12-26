"""Explainability utilities for patient readmission prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Model explainability utilities."""
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize model explainer.
        
        Args:
            model: Trained model.
            feature_names: Names of features.
            feature_names: Names of features.
            background_data: Background data for SHAP explainer.
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        if background_data is not None:
            self._init_shap_explainer()
            
    def _init_shap_explainer(self) -> None:
        """Initialize SHAP explainer."""
        try:
            # Try different SHAP explainers based on model type
            if hasattr(self.model, "predict_proba"):
                # Tree-based models
                self.shap_explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, "coef_"):
                # Linear models
                self.shap_explainer = shap.LinearExplainer(self.model, self.background_data)
            else:
                # Generic explainer
                self.shap_explainer = shap.Explainer(self.model, self.background_data)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def compute_shap_values(
        self,
        X: np.ndarray,
        max_samples: int = 100,
    ) -> Optional[np.ndarray]:
        """Compute SHAP values.
        
        Args:
            X: Input features.
            max_samples: Maximum number of samples to explain.
            
        Returns:
            SHAP values.
        """
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not initialized")
            return None
            
        try:
            # Limit samples for computational efficiency
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
                
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                # For binary classification, take positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
            return shap_values
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None
    
    def plot_shap_summary(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
        max_display: int = 20,
    ) -> None:
        """Plot SHAP summary plot.
        
        Args:
            X: Input features.
            save_path: Path to save plot.
            max_display: Maximum number of features to display.
        """
        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X[:len(shap_values)],
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Summary Plot")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_shap_waterfall(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot SHAP waterfall plot for a single prediction.
        
        Args:
            X: Input features.
            sample_idx: Index of sample to explain.
            save_path: Path to save plot.
        """
        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return
            
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap_values[sample_idx],
            X[sample_idx],
            feature_names=self.feature_names,
            show=False,
        )
        plt.title(f"SHAP Waterfall Plot - Sample {sample_idx}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "shap",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature importance.
        
        Args:
            X: Input features.
            y: Target labels.
            method: Method to use ('shap', 'permutation').
            save_path: Path to save plot.
        """
        if method == "shap":
            self._plot_shap_importance(X, save_path)
        elif method == "permutation":
            self._plot_permutation_importance(X, y, save_path)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _plot_shap_importance(
        self,
        X: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot SHAP-based feature importance."""
        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return
            
        # Compute mean absolute SHAP values
        importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.xlabel("Mean |SHAP value|")
        plt.title("Feature Importance (SHAP)")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def _plot_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot permutation-based feature importance."""
        if not hasattr(self.model, "score"):
            logger.warning("Model does not support permutation importance")
            return
            
        # Compute permutation importance
        perm_importance = permutation_importance(
            self.model,
            X,
            y,
            n_repeats=10,
            random_state=42,
        )
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": perm_importance.importances_mean,
            "std": perm_importance.importances_std,
        }).sort_values("importance", ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(
            importance_df["feature"],
            importance_df["importance"],
            xerr=importance_df["std"],
        )
        plt.xlabel("Permutation Importance")
        plt.title("Feature Importance (Permutation)")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def explain_prediction(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
    ) -> Dict[str, Any]:
        """Explain a single prediction.
        
        Args:
            X: Input features.
            sample_idx: Index of sample to explain.
            
        Returns:
            Explanation dictionary.
        """
        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return {}
            
        # Get prediction
        if hasattr(self.model, "predict_proba"):
            pred_proba = self.model.predict_proba(X[sample_idx:sample_idx+1])[0, 1]
        else:
            pred_proba = self.model.predict(X[sample_idx:sample_idx+1])[0]
            
        # Get SHAP values for this sample
        sample_shap = shap_values[sample_idx]
        sample_features = X[sample_idx]
        
        # Create explanation
        explanation = {
            "prediction_probability": float(pred_proba),
            "prediction_class": int(pred_proba > 0.5),
            "feature_contributions": {},
            "top_positive_features": [],
            "top_negative_features": [],
        }
        
        # Feature contributions
        for i, feature_name in enumerate(self.feature_names):
            explanation["feature_contributions"][feature_name] = {
                "value": float(sample_features[i]),
                "shap_value": float(sample_shap[i]),
            }
            
        # Sort features by SHAP value
        feature_shap_pairs = [
            (name, sample_shap[i]) for i, name in enumerate(self.feature_names)
        ]
        feature_shap_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Top positive and negative features
        explanation["top_positive_features"] = [
            {"feature": name, "shap_value": float(shap_val)}
            for name, shap_val in feature_shap_pairs[:5]
            if shap_val > 0
        ]
        
        explanation["top_negative_features"] = [
            {"feature": name, "shap_value": float(shap_val)}
            for name, shap_val in feature_shap_pairs[-5:]
            if shap_val < 0
        ]
        
        return explanation
    
    def get_feature_importance_ranking(self, X: np.ndarray) -> pd.DataFrame:
        """Get feature importance ranking.
        
        Args:
            X: Input features.
            
        Returns:
            DataFrame with feature importance ranking.
        """
        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            return pd.DataFrame()
            
        # Compute mean absolute SHAP values
        importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create ranking dataframe
        ranking_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
            "rank": range(1, len(self.feature_names) + 1),
        }).sort_values("importance", ascending=False)
        
        ranking_df["rank"] = range(1, len(ranking_df) + 1)
        
        return ranking_df
