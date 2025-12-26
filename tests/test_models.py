"""Tests for model implementations."""

import pytest
import numpy as np
import torch

from src.models.tabular import ModelFactory, TabularNet, AttentionTabularNet, get_model_config


class TestModelFactory:
    """Test ModelFactory class."""
    
    def test_create_sklearn_model(self):
        """Test creation of sklearn models."""
        models = ["random_forest", "gradient_boosting", "logistic_regression", "svm"]
        
        for model_type in models:
            model = ModelFactory.create_model(model_type)
            assert model is not None
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            
    def test_create_xgboost_model(self):
        """Test XGBoost model creation."""
        model = ModelFactory.create_model("xgboost")
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
        
    def test_create_neural_network(self):
        """Test neural network model creation."""
        input_dim = 10
        
        # Test TabularNet
        model = ModelFactory.create_model("tabular_net", input_dim=input_dim)
        assert isinstance(model, TabularNet)
        assert model.input_dim == input_dim
        
        # Test AttentionTabularNet
        model = ModelFactory.create_model("attention_tabular_net", input_dim=input_dim)
        assert isinstance(model, AttentionTabularNet)
        assert model.input_dim == input_dim
        
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError):
            ModelFactory.create_model("invalid_model")
            
    def test_neural_network_missing_input_dim(self):
        """Test that neural networks require input_dim."""
        with pytest.raises(ValueError):
            ModelFactory.create_model("tabular_net")


class TestTabularNet:
    """Test TabularNet class."""
    
    def test_init(self):
        """Test TabularNet initialization."""
        model = TabularNet(input_dim=10)
        assert model.input_dim == 10
        assert len(model.layers) > 0
        assert model.output_layer is not None
        
    def test_forward(self):
        """Test TabularNet forward pass."""
        model = TabularNet(input_dim=5, hidden_dims=[10, 5])
        x = torch.randn(3, 5)
        
        output = model(x)
        assert output.shape == (3, 1)
        
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ["relu", "gelu", "swish"]
        
        for activation in activations:
            model = TabularNet(input_dim=5, activation=activation)
            x = torch.randn(2, 5)
            output = model(x)
            assert output.shape == (2, 1)
            
    def test_batch_norm(self):
        """Test batch normalization."""
        model = TabularNet(input_dim=5, use_batch_norm=True)
        assert model.batch_norms is not None
        
        model_no_bn = TabularNet(input_dim=5, use_batch_norm=False)
        assert model_no_bn.batch_norms is None
        
    def test_residual_connections(self):
        """Test residual connections."""
        model = TabularNet(input_dim=5, use_residual=True)
        x = torch.randn(2, 5)
        output = model(x)
        assert output.shape == (2, 1)


class TestAttentionTabularNet:
    """Test AttentionTabularNet class."""
    
    def test_init(self):
        """Test AttentionTabularNet initialization."""
        model = AttentionTabularNet(input_dim=10)
        assert model.input_dim == 10
        assert model.num_heads == 4
        assert model.attention is not None
        
    def test_forward(self):
        """Test AttentionTabularNet forward pass."""
        model = AttentionTabularNet(input_dim=5, hidden_dims=[10, 5])
        x = torch.randn(3, 5)
        
        output = model(x)
        assert output.shape == (3, 1)
        
    def test_different_num_heads(self):
        """Test different number of attention heads."""
        model = AttentionTabularNet(input_dim=5, num_heads=8)
        assert model.num_heads == 8
        assert model.attention.num_heads == 8


class TestModelConfig:
    """Test model configuration utilities."""
    
    def test_get_model_config(self):
        """Test getting model configurations."""
        configs = [
            "random_forest", "gradient_boosting", "logistic_regression",
            "svm", "xgboost", "lightgbm", "catboost",
            "tabular_net", "attention_tabular_net"
        ]
        
        for model_type in configs:
            config = get_model_config(model_type)
            assert isinstance(config, dict)
            assert len(config) > 0
            
    def test_invalid_model_config(self):
        """Test invalid model type returns empty config."""
        config = get_model_config("invalid_model")
        assert config == {}


class TestModelTraining:
    """Test model training functionality."""
    
    def test_sklearn_model_training(self):
        """Test sklearn model training."""
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        model = ModelFactory.create_model("random_forest")
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(y), 2)
        
    def test_neural_network_training(self):
        """Test neural network training."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        X = torch.randn(100, 5)
        y = torch.randint(0, 2, (100,))
        
        model = TabularNet(input_dim=5, hidden_dims=[10, 5])
        
        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.BCEWithLogitsLoss()
        
        dataset = TensorDataset(X, y.float())
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model.train()
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        # Test prediction
        model.eval()
        with torch.no_grad():
            test_output = model(X[:10])
            assert test_output.shape == (10, 1)
