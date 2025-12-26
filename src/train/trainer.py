"""Training utilities for patient readmission prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib

from ..utils.core import EarlyStopping, get_device
from ..data.utils import compute_class_weights
from ..metrics.clinical import ModelEvaluator

logger = logging.getLogger(__name__)


class TabularTrainer:
    """Trainer for tabular models."""
    
    def __init__(
        self,
        model: Union[nn.Module, Any],
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train.
            device: Device to use for training.
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            class_weights: Class weights for imbalanced data.
        """
        self.model = model
        self.device = device or get_device()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        
        # Move model to device if it's a PyTorch model
        if isinstance(model, nn.Module):
            self.model = model.to(self.device)
            
        # Initialize optimizer and loss function
        if isinstance(model, nn.Module):
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            
            if class_weights:
                weights = torch.FloatTensor([class_weights[0], class_weights[1]]).to(self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])
            else:
                self.criterion = nn.BCEWithLogitsLoss()
                
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            
        Returns:
            Dictionary of metrics for this epoch.
        """
        if isinstance(self.model, nn.Module):
            return self._train_pytorch_epoch(train_loader, val_loader)
        else:
            return self._train_sklearn_epoch(train_loader, val_loader)
    
    def _train_pytorch_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Train PyTorch model for one epoch."""
        self.model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).float()
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x).squeeze()
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        metrics = {"train_loss": train_loss}
        
        if val_loader:
            val_metrics = self._validate_pytorch(val_loader)
            metrics.update(val_metrics)
            
        return metrics
    
    def _train_sklearn_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Train sklearn model."""
        # Extract data from loader
        train_x, train_y = [], []
        for batch_x, batch_y in train_loader:
            train_x.append(batch_x.numpy())
            train_y.append(batch_y.numpy())
            
        train_x = np.vstack(train_x)
        train_y = np.hstack(train_y)
        
        # Train model
        self.model.fit(train_x, train_y)
        
        # Predict on training data
        train_pred_proba = self.model.predict_proba(train_x)[:, 1]
        train_auc = roc_auc_score(train_y, train_pred_proba)
        
        metrics = {"train_auc": train_auc}
        
        if val_loader:
            val_metrics = self._validate_sklearn(val_loader)
            metrics.update(val_metrics)
            
        return metrics
    
    def _validate_pytorch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate PyTorch model."""
        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device).float()
                
                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        return {"val_loss": val_loss, "val_auc": val_auc}
    
    def _validate_sklearn(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate sklearn model."""
        val_x, val_y = [], []
        for batch_x, batch_y in val_loader:
            val_x.append(batch_x.numpy())
            val_y.append(batch_y.numpy())
            
        val_x = np.vstack(val_x)
        val_y = np.hstack(val_y)
        
        val_pred_proba = self.model.predict_proba(val_x)[:, 1]
        val_auc = roc_auc_score(val_y, val_pred_proba)
        
        return {"val_auc": val_auc}
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            epochs: Number of epochs.
            batch_size: Batch size.
            patience: Early stopping patience.
            verbose: Whether to print progress.
            
        Returns:
            Training history.
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val),
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Training loop
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader, val_loader)
            
            # Update history
            for key, value in metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
                
            # Early stopping check
            if val_loader and "val_auc" in metrics:
                if early_stopping(metrics["val_auc"], self.model):
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
            # Print progress
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: {metrics}")
                
        return self.history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted probabilities.
        """
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor).squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                return torch.sigmoid(outputs).cpu().numpy()
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, path: str) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save the model.
        """
        if isinstance(self.model, nn.Module):
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            }, path)
        else:
            joblib.dump(self.model, path)


class CrossValidator:
    """Cross-validation for model evaluation."""
    
    def __init__(
        self,
        model_factory: Any,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        """Initialize cross-validator.
        
        Args:
            model_factory: Function to create model instances.
            cv_folds: Number of CV folds.
            random_state: Random seed.
        """
        self.model_factory = model_factory
        self.cv_folds = cv_folds
        self.random_state = random_state
        
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Perform cross-validation.
        
        Args:
            X: Features.
            y: Labels.
            model_params: Model parameters.
            
        Returns:
            Cross-validation results.
        """
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = self.model_factory(**(model_params or {}))
            
            if isinstance(model, nn.Module):
                trainer = TabularTrainer(model)
                trainer.train(X_train, y_train, X_val, y_val, epochs=50, verbose=False)
                val_pred_proba = trainer.predict_proba(X_val)
            else:
                model.fit(X_train, y_train)
                val_pred_proba = model.predict_proba(X_val)[:, 1]
                
            # Compute AUC
            auc = roc_auc_score(y_val, val_pred_proba)
            cv_scores.append(auc)
            
            logger.info(f"Fold {fold + 1}: AUC = {auc:.4f}")
            
        return {
            "mean_auc": np.mean(cv_scores),
            "std_auc": np.std(cv_scores),
            "cv_scores": cv_scores,
        }
