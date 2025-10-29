# VLM Finetuning Module for Medical Vision-Language Models
# Supports LLaVA, MedVLM, and other medical VLMs

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, AutoModelForVision2Seq,
    Trainer, TrainingArguments,
    BitsAndBytesConfig, AutoTokenizer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np
from dataclasses import dataclass
from accelerate import Accelerator
import wandb

from retrieval.medical_data_loader import MedicalDataLoader, MedicalVQA
from .vlm_config import get_model_config, get_training_preset, validate_data_directory

logger = logging.getLogger(__name__)

@dataclass
class VLMTrainingConfig:
    """Configuration for VLM finetuning"""
    model_key: str = "llava-med"  # Model key from vlm_config.py
    output_dir: str = "./models/vlm_finetuned"
    preset: str = "medical_finetune"  # Training preset
    data_dir: str = "./data/medvlm_data"
    use_wandb: bool = True
    wandb_project: str = "medical-vlm-finetuning"

    # Will be populated from model config and preset
    model_name: str = ""
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_8bit: bool = True
    use_gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False

class MedicalVLMDataset(Dataset):
    """Dataset for Medical VLM training"""

    def __init__(self, vqa_data: List[MedicalVQA], processor, max_length: int = 512, image_dir: str = None):
        self.vqa_data = vqa_data
        self.processor = processor
        self.max_length = max_length
        self.image_dir = image_dir

        # Filter out items that can't be processed
        self.valid_indices = []
        for idx, item in enumerate(vqa_data):
            try:
                # Test if item can be processed by checking image exists
                image_path = item.image_path
                if os.path.exists(image_path):
                    self.valid_indices.append(idx)
                else:
                    logger.warning(f"Image not found for item {idx}: {image_path}")
            except Exception:
                logger.warning(f"Skipping unprocessable item {idx}")
                continue

        logger.info(f"Dataset initialized with {len(self.valid_indices)} valid items out of {len(vqa_data)} total")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        item = self.vqa_data[actual_idx]
        return self._process_item(item, actual_idx)

    def _create_conversation(self, item: MedicalVQA):
        """Create conversation format for the model"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": item.question},
                    {"type": "image"}
                ]
            },
            {
                "role": "assistant",
                "content": item.answer
            }
        ]

    def _process_item(self, item: MedicalVQA, idx: int):
        """Process a single VQA item"""
        # Load and process image - use annotation path directly
        image_path = item.image_path

        try:
            image = Image.open(image_path).convert("RGB")
            # Resize image to model's expected size
            image = image.resize((224, 224))  # BLIP-2 expects 224x224
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Create a placeholder image
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Process with processor for BLIP-2 format
        try:
            # For BLIP-2, we need to format the input as question + context
            question = item.question
            answer = item.answer

            # Create input text as question + answer for training
            input_text = f"Question: {question} Answer: {answer}"

            processed = self.processor(images=image, text=input_text, return_tensors="pt")

            # For causal LM training, labels should be the same as input_ids
            return {
                "input_ids": processed["input_ids"].squeeze(),
                "attention_mask": processed["attention_mask"].squeeze(),
                "pixel_values": processed["pixel_values"].squeeze(),
                "labels": processed["input_ids"].squeeze()  # For language modeling loss
            }
        except Exception as e:
            logger.warning(f"Error processing item {idx}: {e}")
            # This should not happen since we filter in __init__, but just in case
            raise e

class VLMTrainer:
    """Medical VLM Finetuning Trainer"""

    def __init__(self, config: VLMTrainingConfig):
        self.config = config
        self.accelerator = Accelerator()
        self.setup_logging()

        # Load model and training configurations
        self.load_configurations()

        if config.use_wandb:
            wandb.init(project=config.wandb_project, config=config.__dict__)

    def load_configurations(self):
        """Load model and training configurations"""
        # Get model configuration
        model_config = get_model_config(self.config.model_key)
        self.config.model_name = model_config["model_name"]
        self.config.max_length = model_config.get("max_length", 512)

        # Get training preset
        preset_config = get_training_preset(self.config.preset)
        for key, value in preset_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Validate data directory
        validation = validate_data_directory(self.config.data_dir)
        if not validation["valid"]:
            raise ValueError(f"Invalid data directory: {validation['errors']}")

        logger.info(f"Loaded configuration for model: {self.config.model_key}")
        logger.info(f"Training preset: {self.config.preset}")
        logger.info(f"Data validation: {validation['stats']}")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def load_model_and_processor(self):
        """Load VLM model and processor"""
        logger.info(f"Loading model: {self.config.model_name}")

        # Configure quantization (only if CUDA is available)
        if self.config.use_8bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        else:
            quantization_config = None
            if self.config.use_8bit:
                logger.warning("CUDA not available, disabling 8-bit quantization")

        # Load model and processor
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            logger.warning(f"AutoModelForVision2Seq failed, trying AutoModelForImageTextToText: {e}")
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

        # Apply LoRA if enabled
        if self.config.use_lora:
            self.model = self.prepare_model_for_lora()

        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Ensure model is in training mode and parameters require gradients
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad_(True)

        logger.info("Model and processor loaded successfully")

    def prepare_model_for_lora(self):
        """Prepare model for LoRA training"""
        logger.info("Applying LoRA configuration")

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Apply LoRA
        model = get_peft_model(self.model, lora_config)
        model.print_trainable_parameters()

        return model

    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        logger.info("Preparing datasets")

        # Load VLM data
        data_loader = MedicalDataLoader(medvlm_dir=self.config.data_dir)
        vqa_data = data_loader.load_medvlm_data()

        if len(vqa_data) == 0:
            raise ValueError("No VLM training data found. Please check your medvlm_data directory.")

        # Split into train/val (80/20)
        train_size = int(0.8 * len(vqa_data))
        train_data = vqa_data[:train_size]
        val_data = vqa_data[train_size:]

        logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

        # Create datasets - don't pass image_dir since annotation paths are already complete
        self.train_dataset = MedicalVLMDataset(
            train_data, self.processor, self.config.max_length, image_dir=None
        )
        self.val_dataset = MedicalVLMDataset(
            val_data, self.processor, self.config.max_length, image_dir=None
        )

    def get_training_arguments(self):
        """Get training arguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_total_limit=3,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            bf16=torch.cuda.is_available(),
            report_to="wandb" if self.config.use_wandb else "none",
            gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

    def train(self):
        """Run the training process"""
        logger.info("Starting VLM finetuning")

        # Load model and data
        self.load_model_and_processor()
        self.prepare_datasets()

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.get_training_arguments(),
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.processor,
        )

        # Start training
        trainer.train()

        # Save final model
        trainer.save_model(os.path.join(self.config.output_dir, "final_model"))
        self.processor.save_pretrained(os.path.join(self.config.output_dir, "final_model"))

        logger.info("Training completed successfully")

    def evaluate(self, model_path: Optional[str] = None):
        """Evaluate the trained model"""
        if model_path:
            logger.info(f"Loading model from {model_path}")
            self.model = AutoModelForVision2Seq.from_pretrained(model_path)
            self.processor = AutoProcessor.from_pretrained(model_path)

        # Create test dataset (using validation data for now)
        test_dataset = MedicalVLMDataset(
            self.val_dataset.vqa_data[:10],  # Test on first 10 samples
            self.processor,
            self.config.max_length
        )

        trainer = Trainer(
            model=self.model,
            args=self.get_training_arguments(),
            eval_dataset=test_dataset,
            tokenizer=self.processor,
        )

        results = trainer.evaluate()
        logger.info(f"Evaluation results: {results}")
        return results

def main():
    """Main function for VLM training"""
    import argparse

    parser = argparse.ArgumentParser(description="Finetune Medical VLM")
    parser.add_argument("--model_name", type=str, default="microsoft/llava-med-v1.5-mistral-7b",
                       help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="./models/vlm_finetuned",
                       help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Create config
    config = VLMTrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        use_wandb=not args.no_wandb
    )

    # Initialize and run trainer
    trainer = VLMTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()