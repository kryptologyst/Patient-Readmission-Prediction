#!/usr/bin/env python3
"""Main training script for patient readmission prediction."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data.utils import (
    DataProcessor,
    create_patient_level_split,
    compute_class_weights,
    generate_synthetic_readmission_data,
)
from models.tabular import ModelFactory, get_model_config
from train.trainer import TabularTrainer, CrossValidator
from metrics.clinical import ModelEvaluator, create_leaderboard
from utils.core import set_seed, get_device, create_experiment_dir, log_experiment_info
from utils.explainability import ModelExplainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(config: DictConfig) -> pd.DataFrame:
    """Load or generate data.
    
    Args:
        config: Configuration object.
        
    Returns:
        Loaded dataframe.
    """
    data_path = config.get("data_path")
    
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info("Generating synthetic data")
        df = generate_synthetic_readmission_data(
            n_samples=config.data.n_samples,
            n_patients=config.data.n_patients,
            random_state=config.data.random_state,
        )
        
    logger.info(f"Loaded data with shape: {df.shape}")
    logger.info(f"Readmission rate: {df[config.data.target_col].mean():.3f}")
    
    return df


def prepare_data(
    df: pd.DataFrame,
    config: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataProcessor]:
    """Prepare data for training.
    
    Args:
        df: Input dataframe.
        config: Configuration object.
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, processor).
    """
    # Create patient-level splits
    train_df, val_df, test_df = create_patient_level_split(
        df,
        patient_id_col=config.data.patient_id_col,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.data.random_state,
    )
    
    # Initialize data processor
    processor = DataProcessor(
        categorical_features=config.data.categorical_features,
        numerical_features=config.data.numerical_features,
        target_column=config.data.target_col,
    )
    
    # Process training data
    X_train, y_train = processor.fit_transform(train_df)
    
    # Process validation and test data
    X_val = processor.transform(val_df)
    y_val = val_df[config.data.target_col].values
    
    X_test = processor.transform(test_df)
    y_test = test_df[config.data.target_col].values
    
    logger.info(f"Training set: {X_train.shape}, {y_train.shape}")
    logger.info(f"Validation set: {X_val.shape}, {y_val.shape}")
    logger.info(f"Test set: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, processor


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: DictConfig,
) -> Any:
    """Train the model.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        config: Configuration object.
        
    Returns:
        Trained model.
    """
    model_type = config.model.type
    model_config = get_model_config(model_type)
    
    # Update config with model-specific parameters
    if model_type in config.model:
        model_config.update(config.model[model_type])
        
    # Create model
    if model_type in ["tabular_net", "attention_tabular_net"]:
        model = ModelFactory.create_model(
            model_type,
            input_dim=X_train.shape[1],
            **model_config,
        )
    else:
        model = ModelFactory.create_model(model_type, **model_config)
        
    logger.info(f"Created {model_type} model")
    
    # Compute class weights if needed
    class_weights = None
    if config.training.use_class_weights:
        class_weights = compute_class_weights(y_train)
        logger.info(f"Class weights: {class_weights}")
        
    # Train model
    if model_type in ["tabular_net", "attention_tabular_net"]:
        trainer = TabularTrainer(
            model=model,
            device=get_device(),
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            class_weights=class_weights,
        )
        
        history = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            patience=config.training.patience,
        )
        
        logger.info(f"Training completed. Best validation AUC: {max(history['val_auc']):.4f}")
        return trainer.model
    else:
        # Train sklearn models
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_auc = ModelEvaluator().clinical_metrics.compute_metrics(y_val, val_pred_proba)["auc_roc"]
        logger.info(f"Validation AUC: {val_auc:.4f}")
        
        return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    processor: DataProcessor,
    config: DictConfig,
    output_dir: str,
) -> Dict[str, Any]:
    """Evaluate the trained model.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        processor: Data processor.
        config: Configuration object.
        output_dir: Output directory.
        
    Returns:
        Evaluation results.
    """
    # Get predictions
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For PyTorch models
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(get_device())
            outputs = model(X_test_tensor).squeeze()
            y_pred_proba = torch.sigmoid(outputs).cpu().numpy()
            
    # Evaluate model
    evaluator = ModelEvaluator(threshold=config.evaluation.threshold)
    results = evaluator.evaluate(
        y_true=y_test,
        y_pred_proba=y_pred_proba,
        model_name=config.model.type,
    )
    
    logger.info(f"Test AUC: {results['metrics']['auc_roc']:.4f}")
    logger.info(f"Test Sensitivity: {results['metrics']['sensitivity']:.4f}")
    logger.info(f"Test Specificity: {results['metrics']['specificity']:.4f}")
    
    # Save plots if requested
    if config.evaluation.save_plots:
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        evaluator.plot_roc_curve(
            y_test, y_pred_proba,
            save_path=os.path.join(plots_dir, "roc_curve.png"),
        )
        
        evaluator.plot_precision_recall_curve(
            y_test, y_pred_proba,
            save_path=os.path.join(plots_dir, "pr_curve.png"),
        )
        
        evaluator.plot_calibration_curve(
            y_test, y_pred_proba,
            save_path=os.path.join(plots_dir, "calibration_curve.png"),
        )
        
        evaluator.plot_confusion_matrix(
            y_test, y_pred_proba,
            save_path=os.path.join(plots_dir, "confusion_matrix.png"),
        )
        
    return results


def explain_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    processor: DataProcessor,
    config: DictConfig,
    output_dir: str,
) -> None:
    """Generate model explanations.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        processor: Data processor.
        config: Configuration object.
        output_dir: Output directory.
    """
    # Create explainer
    explainer = ModelExplainer(
        model=model,
        feature_names=processor.feature_names,
        background_data=X_test[:config.explainability.shap.background_samples],
    )
    
    # Generate explanations
    explanations_dir = os.path.join(output_dir, "explanations")
    os.makedirs(explanations_dir, exist_ok=True)
    
    # Feature importance
    explainer.plot_feature_importance(
        X_test,
        y_test,
        method=config.explainability.method,
        save_path=os.path.join(explanations_dir, "feature_importance.png"),
    )
    
    # SHAP summary plot
    explainer.plot_shap_summary(
        X_test,
        save_path=os.path.join(explanations_dir, "shap_summary.png"),
        max_display=config.explainability.max_display,
    )
    
    # Individual explanations
    for i in range(min(5, len(X_test))):
        explanation = explainer.explain_prediction(X_test, sample_idx=i)
        
        # Save explanation
        explanation_path = os.path.join(explanations_dir, f"explanation_sample_{i}.json")
        import json
        with open(explanation_path, "w") as f:
            json.dump(explanation, f, indent=2)
            
    logger.info("Model explanations generated")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train patient readmission prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (optional)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output.base_dir = args.output_dir
        
    # Set random seed
    set_seed(config.reproducibility.seed)
    
    # Create output directory
    output_dir = create_experiment_dir(
        config.output.base_dir,
        config.output.experiment_name,
    )
    
    # Save configuration
    OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))
    
    logger.info(f"Starting experiment in {output_dir}")
    
    try:
        # Load data
        df = load_data(config)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, processor = prepare_data(df, config)
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val, config)
        
        # Evaluate model
        results = evaluate_model(model, X_test, y_test, processor, config, output_dir)
        
        # Generate explanations
        explain_model(model, X_test, y_test, processor, config, output_dir)
        
        # Save model
        if config.output.save_model:
            model_path = os.path.join(output_dir, "model.pkl")
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), model_path)
            else:
                import joblib
                joblib.dump(model, model_path)
                
        # Save results
        log_experiment_info(config, results["metrics"], os.path.join(output_dir, "results.yaml"))
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
