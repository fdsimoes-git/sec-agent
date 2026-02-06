"""CLI entry point for MLX GRPO fine-tuning on Apple Silicon."""

import argparse

from .config import TrainingConfig
from .trainer import run_training


def main():
    """Run the MLX GRPO fine-tuning pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a small LLM for security agent tasks using GRPO on Apple Silicon.",
    )
    parser.add_argument(
        "--model",
        default=TrainingConfig.model_name,
        help="MLX model name or path (default: %(default)s)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=TrainingConfig.lora_rank,
        help="LoRA rank (default: %(default)s)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=TrainingConfig.max_steps,
        help="Maximum training steps (default: %(default)s)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=TrainingConfig.max_seq_length,
        help="Maximum sequence length (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TrainingConfig.learning_rate,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TrainingConfig.temperature,
        help="Sampling temperature for GRPO generations (default: %(default)s)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=TrainingConfig.num_generations,
        help="Number of generations per prompt for GRPO (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=TrainingConfig.dataset_size,
        help="Number of training examples (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=TrainingConfig.output_dir,
        help="Output directory for model checkpoints (default: %(default)s)",
    )
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=TrainingConfig.clip_eps,
        help="PPO clipping epsilon (default: %(default)s)",
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=TrainingConfig.kl_coeff,
        help="KL divergence penalty coefficient (default: %(default)s)",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=TrainingConfig.grad_accumulation_steps,
        help="Gradient accumulation steps (default: %(default)s)",
    )
    parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Fuse LoRA and export for Ollama after training",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print generated completions and reward breakdowns each step",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        num_generations=args.num_generations,
        dataset_size=args.dataset_size,
        output_dir=args.output_dir,
        clip_eps=args.clip_eps,
        kl_coeff=args.kl_coeff,
        grad_accumulation_steps=args.grad_accumulation_steps,
        verbose=args.verbose,
    )

    run_training(config, export_gguf=args.export_gguf)


if __name__ == "__main__":
    main()
