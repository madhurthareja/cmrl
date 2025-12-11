#!/usr/bin/env python3
"""
VLM Finetuning Script for Medical Vision-Language Models
Supports both custom datasets and UMIE medical imaging datasets
"""

import os
import sys
import argparse

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.training.vlm_trainer import VLMTrainer, VLMTrainingConfig
from backend.training.vlm_config import VLM_MODELS, TRAINING_PRESETS, UMIE_DATASETS, validate_data_directory
from backend.retrieval.umie_data_loader import create_umie_vqa_dataset

def main():
    parser = argparse.ArgumentParser(description="Finetune Medical VLM")
    parser.add_argument("--model", type=str, default="llava-med",
                       choices=list(VLM_MODELS.keys()) + ["custom"],
                       help="VLM model to finetune")
    parser.add_argument("--custom_model", type=str,
                       help="Custom HuggingFace model name (when model=custom)")
    parser.add_argument("--preset", type=str, default="medical_finetune",
                       choices=list(TRAINING_PRESETS.keys()),
                       help="Training preset configuration")
    parser.add_argument("--output_dir", type=str, default="./models/vlm_finetuned",
                       help="Output directory for trained model")
    parser.add_argument("--data_dir", type=str, default="./data/medvlm_data",
                       help="Directory containing VLM training data")
    parser.add_argument("--umie_dataset", type=str,
                       choices=list(UMIE_DATASETS.keys()),
                       help="Use UMIE dataset instead of local data")
    parser.add_argument("--umie_samples", type=int, default=1000,
                       help="Maximum samples to download from UMIE dataset")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--resume_from", type=str,
                       help="Resume training from checkpoint")

    args = parser.parse_args()

    # Handle UMIE dataset
    if args.umie_dataset:
        print(f"Using UMIE dataset: {args.umie_dataset}")
        dataset_info = UMIE_DATASETS[args.umie_dataset]
        print(f"Description: {dataset_info['description']}")
        print(f"Domain: {dataset_info['domain']}")
        print(f"Estimated samples: {dataset_info['estimated_samples']}")

        # Download and convert UMIE dataset
        umie_output = os.path.join(args.data_dir, f"umie_{args.umie_dataset}_annotations.json")
        print(f"Downloading and converting UMIE dataset (max {args.umie_samples} samples)...")

        try:
            create_umie_vqa_dataset(
                config_name=args.umie_dataset,
                max_samples=args.umie_samples,
                output_file=umie_output
            )
            print(f"✓ UMIE dataset prepared: {umie_output}")

            # Update data directory to point to the UMIE annotations
            args.data_dir = os.path.dirname(umie_output)

        except Exception as e:
            print(f"✗ Failed to prepare UMIE dataset: {e}")
            sys.exit(1)

    # Validate data directory
    print(f"Validating data directory: {args.data_dir}")
    validation = validate_data_directory(args.data_dir)
    if not validation["valid"]:
        print("Data validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        sys.exit(1)

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    print(f"Data stats: {validation['stats']}")

    # Select model
    if args.model == "custom":
        if not args.custom_model:
            print("Error: --custom_model required when model=custom")
            sys.exit(1)
        model_key = args.custom_model
    else:
        model_key = args.model

    print(f"Starting VLM finetuning with model: {model_key}")
    print(f"Training preset: {args.preset}")
    print(f"Output directory: {args.output_dir}")

    # Create configuration
    config = VLMTrainingConfig(
        model_key=model_key,
        preset=args.preset,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        use_wandb=not args.no_wandb
    )

    # Initialize trainer
    trainer = VLMTrainer(config)

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        # Load checkpoint logic would go here

    # Start training
    try:
        trainer.train()
        print("VLM finetuning completed successfully!")
        print(f"Model saved to: {args.output_dir}")

        # Print evaluation command
        print("\nTo evaluate the model, run:")
        print(f"python evaluate_vlm.py --model_path {args.output_dir}/final_model --annotations_file {args.data_dir}/annotations.json --image_dir {args.data_dir}/images")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

def main():
    parser = argparse.ArgumentParser(description="Finetune Medical VLM")
    parser.add_argument("--model", type=str, default="llava-med",
                       choices=list(VLM_MODELS.keys()) + ["custom"],
                       help="VLM model to finetune")
    parser.add_argument("--custom_model", type=str,
                       help="Custom HuggingFace model name (when model=custom)")
    parser.add_argument("--preset", type=str, default="medical_finetune",
                       choices=list(TRAINING_PRESETS.keys()),
                       help="Training preset configuration")
    parser.add_argument("--output_dir", type=str, default="./models/vlm_finetuned",
                       help="Output directory for trained model")
    parser.add_argument("--data_dir", type=str, default="./data/medvlm_data",
                       help="Directory containing VLM training data")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--resume_from", type=str,
                       help="Resume training from checkpoint")

    args = parser.parse_args()

    # Validate data directory
    print(f"Validating data directory: {args.data_dir}")
    validation = validate_data_directory(args.data_dir)
    if not validation["valid"]:
        print("Data validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        sys.exit(1)

    if validation["warnings"]:
        print("Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    print(f"Data stats: {validation['stats']}")

    # Select model
    if args.model == "custom":
        if not args.custom_model:
            print("Error: --custom_model required when model=custom")
            sys.exit(1)
        model_key = args.custom_model
    else:
        model_key = args.model

    print(f"Starting VLM finetuning with model: {model_key}")
    print(f"Training preset: {args.preset}")
    print(f"Output directory: {args.output_dir}")

    # Create configuration
    config = VLMTrainingConfig(
        model_key=model_key,
        preset=args.preset,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        use_wandb=not args.no_wandb
    )

    # Initialize trainer
    trainer = VLMTrainer(config)

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        # Load checkpoint logic would go here

    # Start training
    try:
        trainer.train()
        print("VLM finetuning completed successfully!")
        print(f"Model saved to: {args.output_dir}")

        # Print evaluation command
        print("\nTo evaluate the model, run:")
        print(f"python evaluate_vlm.py --model_path {args.output_dir}/final_model --annotations_file {args.data_dir}/annotations.json --image_dir {args.data_dir}/images")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()