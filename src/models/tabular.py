"""Models for patient readmission prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


class TabularNet(nn.Module):
    """Deep tabular neural network for EHR data."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_residual: bool = False,
    ) -> None:
        """Initialize TabularNet.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            dropout_rate: Dropout rate.
            activation: Activation function ('relu', 'gelu', 'swish').
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Build layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim)
            self.layers.append(layer)
            prev_dim = hidden_dim
            
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_dims
        ]) if use_batch_norm else None
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        residual = None
        
        for i, layer in enumerate(self.layers):
            # Store input for residual connection
            if self.use_residual and i > 0:
                residual = x
                
            x = layer(x)
            
            # Apply batch normalization
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
                
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout
            x = self.dropout(x)
            
            # Add residual connection
            if self.use_residual and residual is not None and x.shape == residual.shape:
                x = x + residual
                
        # Output layer
        x = self.output_layer(x)
        
        return x


class AttentionTabularNet(nn.Module):
    """Tabular network with attention mechanism."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        num_heads: int = 4,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize AttentionTabularNet.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            num_heads: Number of attention heads.
            dropout_rate: Dropout rate.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        
        # Feed-forward layers
        self.feed_forward = nn.ModuleList()
        prev_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            self.feed_forward.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
            
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dims[0])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # Residual connection
        x = self.layer_norm(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Feed-forward layers
        for layer in self.feed_forward:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
            
        # Output layer
        x = self.output_layer(x)
        
        return x


class ModelFactory:
    """Factory for creating different types of models."""
    
    @staticmethod
    def create_model(
        model_type: str,
        input_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[
        RandomForestClassifier,
        GradientBoostingClassifier,
        LogisticRegression,
        SVC,
        xgb.XGBClassifier,
        lgb.LGBMClassifier,
        CatBoostClassifier,
        TabularNet,
        AttentionTabularNet,
    ]:
        """Create a model instance.
        
        Args:
            model_type: Type of model to create.
            input_dim: Input dimension (for neural networks).
            **kwargs: Additional model parameters.
            
        Returns:
            Model instance.
        """
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                random_state=kwargs.get("random_state", 42),
            )
            
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 3),
                random_state=kwargs.get("random_state", 42),
            )
            
        elif model_type == "logistic_regression":
            return LogisticRegression(
                C=kwargs.get("C", 1.0),
                random_state=kwargs.get("random_state", 42),
                max_iter=kwargs.get("max_iter", 1000),
            )
            
        elif model_type == "svm":
            return SVC(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "rbf"),
                probability=True,
                random_state=kwargs.get("random_state", 42),
            )
            
        elif model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 6),
                random_state=kwargs.get("random_state", 42),
                eval_metric="logloss",
            )
            
        elif model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                max_depth=kwargs.get("max_depth", 6),
                random_state=kwargs.get("random_state", 42),
                verbose=-1,
            )
            
        elif model_type == "catboost":
            return CatBoostClassifier(
                iterations=kwargs.get("iterations", 100),
                learning_rate=kwargs.get("learning_rate", 0.1),
                depth=kwargs.get("depth", 6),
                random_state=kwargs.get("random_state", 42),
                verbose=False,
            )
            
        elif model_type == "tabular_net":
            if input_dim is None:
                raise ValueError("input_dim required for neural networks")
            return TabularNet(
                input_dim=input_dim,
                hidden_dims=kwargs.get("hidden_dims", [128, 64, 32]),
                dropout_rate=kwargs.get("dropout_rate", 0.3),
                activation=kwargs.get("activation", "relu"),
                use_batch_norm=kwargs.get("use_batch_norm", True),
                use_residual=kwargs.get("use_residual", False),
            )
            
        elif model_type == "attention_tabular_net":
            if input_dim is None:
                raise ValueError("input_dim required for neural networks")
            return AttentionTabularNet(
                input_dim=input_dim,
                hidden_dims=kwargs.get("hidden_dims", [128, 64]),
                num_heads=kwargs.get("num_heads", 4),
                dropout_rate=kwargs.get("dropout_rate", 0.3),
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def get_model_config(model_type: str) -> Dict[str, Any]:
    """Get default configuration for a model type.
    
    Args:
        model_type: Type of model.
        
    Returns:
        Default configuration dictionary.
    """
    configs = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 0.8,
        },
        "logistic_regression": {
            "C": 1.0,
            "penalty": "l2",
            "max_iter": 1000,
        },
        "svm": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
        },
        "xgboost": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "lightgbm": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "catboost": {
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 6,
            "subsample": 0.8,
        },
        "tabular_net": {
            "hidden_dims": [128, 64, 32],
            "dropout_rate": 0.3,
            "activation": "relu",
            "use_batch_norm": True,
            "use_residual": False,
        },
        "attention_tabular_net": {
            "hidden_dims": [128, 64],
            "num_heads": 4,
            "dropout_rate": 0.3,
        },
    }
    
    return configs.get(model_type, {})
