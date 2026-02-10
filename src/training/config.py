"""Configuration file parsing and validation."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration."""
    train_data_path: str
    val_data_path: str
    image_size: int = 1024
    batch_size: int = 32
    num_workers: int = 8


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "NextDiT"
    in_channels: int = 3
    mask_channels: int = 1
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    patch_size: int = 2
    use_flash_attn: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    save_every: int = 5
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    log_every: int = 100
    log_dir: str = "logs"


@dataclass
class LossConfig:
    """Loss configuration."""
    type: str = "flow_matching"
    weighting: str = "uniform"


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str = "AdamW"
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "cosine"
    min_lr: float = 1e-6
    warmup_type: str = "linear"


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with all settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required parameters are missing
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate required sections
    required_sections = ['data', 'model', 'training']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required config section: {section}")
    
    # Parse each section
    data_config = DataConfig(**config_dict['data'])
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    loss_config = LossConfig(**config_dict.get('loss', {}))
    optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
    scheduler_config = SchedulerConfig(**config_dict.get('scheduler', {}))
    
    # Validate paths
    if not Path(data_config.train_data_path).exists():
        print(f"Warning: Training data path does not exist: {data_config.train_data_path}")
    
    if not Path(data_config.val_data_path).exists():
        print(f"Warning: Validation data path does not exist: {data_config.val_data_path}")
    
    # Create output directories
    Path(training_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(training_config.log_dir).mkdir(parents=True, exist_ok=True)
    
    return Config(
        data=data_config,
        model=model_config,
        training=training_config,
        loss=loss_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config
    )


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object
        save_path: Path to save YAML file
    """
    config_dict = {
        'data': config.data.__dict__,
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'loss': config.loss.__dict__,
        'optimizer': config.optimizer.__dict__,
        'scheduler': config.scheduler.__dict__,
    }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def print_config(config: Config):
    """Print configuration in a readable format."""
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    
    print("\n[Data]")
    for key, value in config.data.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Model]")
    for key, value in config.model.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Training]")
    for key, value in config.training.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Loss]")
    for key, value in config.loss.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Optimizer]")
    for key, value in config.optimizer.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Scheduler]")
    for key, value in config.scheduler.__dict__.items():
        print(f"  {key}: {value}")
    
    print("="*70 + "\n")
