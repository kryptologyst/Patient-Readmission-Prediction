"""Data utilities for patient readmission prediction."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data processing utilities for EHR/tabular data."""
    
    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        target_column: str = "readmitted",
        scaler_type: str = "standard",
    ) -> None:
        """Initialize data processor.
        
        Args:
            categorical_features: List of categorical feature names.
            numerical_features: List of numerical feature names.
            target_column: Name of target column.
            scaler_type: Type of scaler ('standard', 'robust').
        """
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.target_column = target_column
        self.scaler_type = scaler_type
        
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.feature_names: List[str] = []
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit processors and transform data.
        
        Args:
            df: Input dataframe.
            
        Returns:
            Tuple of (features, target).
        """
        # Auto-detect features if not specified
        if not self.categorical_features and not self.numerical_features:
            self._auto_detect_features(df)
            
        # Process categorical features
        df_processed = df.copy()
        for feature in self.categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df_processed[feature] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
                
        # Process numerical features
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols:
            if self.scaler_type == "standard":
                self.scaler = StandardScaler()
            elif self.scaler_type == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.scaler_type}")
                
            df_processed[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            
        # Prepare features and target
        feature_cols = self.categorical_features + numerical_cols
        self.feature_names = feature_cols
        
        X = df_processed[feature_cols].values
        y = df_processed[self.target_column].values if self.target_column in df.columns else None
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted processors.
        
        Args:
            df: Input dataframe.
            
        Returns:
            Transformed features.
        """
        df_processed = df.copy()
        
        # Transform categorical features
        for feature in self.categorical_features:
            if feature in df.columns and feature in self.label_encoders:
                le = self.label_encoders[feature]
                # Handle unseen categories
                df_processed[feature] = df[feature].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df_processed[feature] = le.transform(df_processed[feature])
                
        # Transform numerical features
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols and self.scaler is not None:
            df_processed[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
        return df_processed[self.feature_names].values
    
    def _auto_detect_features(self, df: pd.DataFrame) -> None:
        """Auto-detect feature types from dataframe.
        
        Args:
            df: Input dataframe.
        """
        for col in df.columns:
            if col == self.target_column:
                continue
                
            if df[col].dtype in ["object", "category"]:
                self.categorical_features.append(col)
            else:
                self.numerical_features.append(col)
                
        logger.info(f"Auto-detected categorical features: {self.categorical_features}")
        logger.info(f"Auto-detected numerical features: {self.numerical_features}")


def create_patient_level_split(
    df: pd.DataFrame,
    patient_id_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create patient-level train/validation/test splits.
    
    Args:
        df: Input dataframe.
        patient_id_col: Name of patient ID column.
        test_size: Proportion of patients for test set.
        val_size: Proportion of patients for validation set.
        random_state: Random seed.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    unique_patients = df[patient_id_col].unique()
    
    # Split patients
    train_patients, temp_patients = train_test_split(
        unique_patients,
        test_size=test_size + val_size,
        random_state=random_state,
    )
    
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
    )
    
    # Create splits
    train_df = df[df[patient_id_col].isin(train_patients)]
    val_df = df[df[patient_id_col].isin(val_patients)]
    test_df = df[df[patient_id_col].isin(test_patients)]
    
    logger.info(f"Train: {len(train_df)} samples from {len(train_patients)} patients")
    logger.info(f"Val: {len(val_df)} samples from {len(val_patients)} patients")
    logger.info(f"Test: {len(test_df)} samples from {len(test_patients)} patients")
    
    return train_df, val_df, test_df


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced datasets.
    
    Args:
        y: Target labels.
        
    Returns:
        Dictionary mapping class indices to weights.
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def generate_synthetic_readmission_data(
    n_samples: int = 1000,
    n_patients: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate synthetic patient readmission data for demonstration.
    
    Args:
        n_samples: Number of samples to generate.
        n_patients: Number of unique patients.
        random_state: Random seed.
        
    Returns:
        Synthetic dataframe.
    """
    np.random.seed(random_state)
    
    # Generate patient IDs
    patient_ids = np.random.choice(range(n_patients), size=n_samples)
    
    # Generate features
    data = {
        "patient_id": patient_ids,
        "age": np.random.randint(18, 90, size=n_samples),
        "length_of_stay": np.random.exponential(5, size=n_samples).astype(int) + 1,
        "num_lab_procedures": np.random.poisson(20, size=n_samples),
        "num_medications": np.random.poisson(8, size=n_samples),
        "num_procedures": np.random.poisson(3, size=n_samples),
        "num_diagnoses": np.random.poisson(5, size=n_samples),
        "admission_type": np.random.choice(
            ["Emergency", "Urgent", "Elective", "Newborn"], 
            size=n_samples,
            p=[0.4, 0.3, 0.2, 0.1]
        ),
        "discharge_disposition": np.random.choice(
            ["Home", "SNF", "Rehab", "AMA", "Expired"], 
            size=n_samples,
            p=[0.6, 0.15, 0.1, 0.1, 0.05]
        ),
        "admission_source": np.random.choice(
            ["Physician Referral", "Emergency Room", "Transfer", "Other"], 
            size=n_samples,
            p=[0.3, 0.4, 0.2, 0.1]
        ),
        "diag_primary": np.random.choice(
            ["Diabetes", "Heart Failure", "Pneumonia", "COPD", "Stroke", "Other"], 
            size=n_samples,
            p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.3]
        ),
        "gender": np.random.choice(["Male", "Female"], size=n_samples),
        "race": np.random.choice(
            ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"], 
            size=n_samples,
            p=[0.5, 0.2, 0.15, 0.05, 0.1]
        ),
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic readmission probabilities based on features
    readmission_prob = (
        0.1 +  # Base probability
        0.02 * (df["age"] > 65) +  # Age factor
        0.03 * (df["length_of_stay"] > 7) +  # LOS factor
        0.02 * (df["num_medications"] > 10) +  # Medication factor
        0.02 * (df["num_diagnoses"] > 8) +  # Comorbidity factor
        0.05 * (df["admission_type"] == "Emergency") +  # Emergency admission
        0.03 * (df["discharge_disposition"].isin(["SNF", "Rehab"])) +  # Discharge factor
        0.02 * (df["diag_primary"].isin(["Heart Failure", "COPD"]))  # Diagnosis factor
    )
    
    # Add some randomness
    readmission_prob += np.random.normal(0, 0.02, size=n_samples)
    readmission_prob = np.clip(readmission_prob, 0, 1)
    
    df["readmitted"] = np.random.binomial(1, readmission_prob)
    
    return df
