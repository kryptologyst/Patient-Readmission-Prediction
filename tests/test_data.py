"""Tests for data utilities."""

import pytest
import numpy as np
import pandas as pd

from src.data.utils import (
    DataProcessor,
    create_patient_level_split,
    compute_class_weights,
    generate_synthetic_readmission_data,
)


class TestDataProcessor:
    """Test DataProcessor class."""
    
    def test_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        assert processor.categorical_features == []
        assert processor.numerical_features == []
        assert processor.target_column == "readmitted"
        assert processor.scaler_type == "standard"
        
    def test_fit_transform(self):
        """Test fit_transform method."""
        # Create test data
        data = {
            "age": [25, 30, 35],
            "gender": ["Male", "Female", "Male"],
            "readmitted": [0, 1, 0],
        }
        df = pd.DataFrame(data)
        
        processor = DataProcessor(
            categorical_features=["gender"],
            numerical_features=["age"],
            target_column="readmitted",
        )
        
        X, y = processor.fit_transform(df)
        
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        assert len(processor.feature_names) == 2
        
    def test_transform(self):
        """Test transform method."""
        # Create test data
        data = {
            "age": [25, 30, 35],
            "gender": ["Male", "Female", "Male"],
            "readmitted": [0, 1, 0],
        }
        df = pd.DataFrame(data)
        
        processor = DataProcessor(
            categorical_features=["gender"],
            numerical_features=["age"],
            target_column="readmitted",
        )
        
        # Fit on training data
        processor.fit_transform(df)
        
        # Transform new data
        new_data = {
            "age": [40, 45],
            "gender": ["Female", "Male"],
        }
        new_df = pd.DataFrame(new_data)
        
        X_transformed = processor.transform(new_df)
        assert X_transformed.shape == (2, 2)


class TestDataSplitting:
    """Test data splitting utilities."""
    
    def test_create_patient_level_split(self):
        """Test patient-level splitting."""
        # Create test data with patient IDs
        data = {
            "patient_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "feature": np.random.randn(10),
            "readmitted": [0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
        }
        df = pd.DataFrame(data)
        
        train_df, val_df, test_df = create_patient_level_split(
            df, "patient_id", test_size=0.2, val_size=0.2
        )
        
        # Check that patients don't overlap between splits
        train_patients = set(train_df["patient_id"].unique())
        val_patients = set(val_df["patient_id"].unique())
        test_patients = set(test_df["patient_id"].unique())
        
        assert len(train_patients & val_patients) == 0
        assert len(train_patients & test_patients) == 0
        assert len(val_patients & test_patients) == 0
        
    def test_compute_class_weights(self):
        """Test class weight computation."""
        y = np.array([0, 0, 0, 1, 1])
        weights = compute_class_weights(y)
        
        assert isinstance(weights, dict)
        assert 0 in weights
        assert 1 in weights
        assert weights[1] > weights[0]  # Minority class should have higher weight


class TestSyntheticData:
    """Test synthetic data generation."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        df = generate_synthetic_readmission_data(n_samples=100, n_patients=50)
        
        assert len(df) == 100
        assert len(df["patient_id"].unique()) <= 50
        assert "readmitted" in df.columns
        assert df["readmitted"].isin([0, 1]).all()
        
    def test_synthetic_data_features(self):
        """Test that synthetic data has expected features."""
        df = generate_synthetic_readmission_data(n_samples=50)
        
        expected_features = [
            "patient_id", "age", "length_of_stay", "num_lab_procedures",
            "num_medications", "num_procedures", "num_diagnoses",
            "admission_type", "discharge_disposition", "admission_source",
            "diag_primary", "gender", "race", "readmitted"
        ]
        
        for feature in expected_features:
            assert feature in df.columns
            
    def test_synthetic_data_ranges(self):
        """Test that synthetic data has reasonable ranges."""
        df = generate_synthetic_readmission_data(n_samples=100)
        
        assert df["age"].min() >= 18
        assert df["age"].max() <= 100
        assert df["length_of_stay"].min() >= 1
        assert df["readmitted"].mean() > 0  # Should have some positive cases
        assert df["readmitted"].mean() < 1  # Should have some negative cases
