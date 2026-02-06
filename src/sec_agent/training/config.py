"""Training configuration for MLX GRPO fine-tuning on Apple Silicon."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters and model configuration for MLX GRPO training."""

    # Model
    model_name: str = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
    max_seq_length: int = 1024
    lora_rank: int = 16

    # GRPO training
    num_generations: int = 4
    max_steps: int = 250
    learning_rate: float = 1e-6
    temperature: float = 0.8
    clip_eps: float = 0.2
    kl_coeff: float = 0.04
    grad_accumulation_steps: int = 4
    sync_interval: int = 10
    num_epochs: int = 1

    # Output
    output_dir: str = "outputs"
    save_steps: int = 50
    logging_steps: int = 1

    # Dataset
    dataset_size: int = 500

    # Verbosity
    verbose: bool = False
