"""Core utilities for patient readmission prediction project."""

import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def save_config(config: Union[DictConfig, Dict[str, Any]], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        config_path: Path to save configuration.
    """
    OmegaConf.save(config, config_path)


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments.
        experiment_name: Name of the experiment.
        
    Returns:
        Path to created experiment directory.
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def log_experiment_info(
    config: DictConfig,
    metrics: Dict[str, float],
    save_path: str,
) -> None:
    """Log experiment information to file.
    
    Args:
        config: Experiment configuration.
        metrics: Experiment metrics.
        save_path: Path to save log file.
    """
    log_info = {
        "config": OmegaConf.to_container(config, resolve=True),
        "metrics": metrics,
        "timestamp": str(np.datetime64("now")),
    }
    
    with open(save_path, "w") as f:
        yaml.dump(log_info, f, default_flow_style=False)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ) -> None:
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            model: Model to potentially save weights from.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
        """
        self.best_weights = model.state_dict().copy()
