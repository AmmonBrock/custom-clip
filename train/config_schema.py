from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

# File for training config schema

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_shard_pattern: str
    val_shard_pattern: str
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    text_max_length: int = 77
    shuffle: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2
    truncate_data_at: Optional[int] = None

@dataclass
class ModelConfig:
    """Configuration for CLIP model architecture."""
    vision_model: str = "ViT-B/32"  # or "ResNet-50", etc.
    text_model: str = "transformer"
    embed_dim: int = 512
    vision_layers: int = 12
    vision_width: int = 768
    vision_patch_size: int = 32
    context_length: int = 77
    vocab_size: int = 49408
    transformer_width: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 12

@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""
    name: str = "AdamW"
    lr: float = 5e-4
    weight_decay: float = 0.2
    betas: List[float] = field(default_factory=lambda: [0.9, 0.98])
    eps: float = 1e-6

@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    name: str = "cosine"
    warmup_steps: int = 2000
    max_steps: int = 100000
    min_lr_ratio: float = 0.0

@dataclass
class TrainConfig:
    """Configuration for training process."""
    epochs: int = 100
    save_frequency: int = 10
    eval_frequency: int = 5
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    accumulate_grad_batches: int = 1
    temperature: float = 0.07

@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    experiment_name: str = "clip_experiment"
    log_frequency: int = 100
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    run_dir: str = "runs"
    stdout_log: bool = True
    stderr_log: bool = True
    do_tqdm: bool = True

@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""
    resume_from: Optional[str] = None
    save_top_k: int = 3
    monitor_metric: str = "val_loss"

@dataclass
class CLIPConfig:
    """Main configuration class for CLIP training."""
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    train: TrainConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig
    seed: int = 42
    device: str = "cuda"

def read_config(config_path: str) -> CLIPConfig:
    """Reads a configuration file and returns a CLIPConfig object."""
    import yaml
    from dacite import from_dict

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return from_dict(data_class=CLIPConfig, data=config_dict)