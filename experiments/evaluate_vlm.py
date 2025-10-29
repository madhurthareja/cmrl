#!/usr/bin/env python3
"""
VLM Evaluation Script for Medical Vision-Language Models
"""

import os
import sys
import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import json
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLMEvaluator:
    """Evaluate finetuned VLM models"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.load_model()

    def load_model(self):
        """Load the VLM model and processor"""
        logger.info(f"Loading model from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        logger.info("Model loaded successfully")

    def generate_answer(self, image_path: str, question: str, max_length: int = 512) -> str:
        """Generate answer for a single image-question pair"""
        # Load and process image
        if not os.path.exists(image_path):
            return "Error: Image not found"

        image = Image.open(image_path).convert('RGB')

        # Format conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"}
                ]
            }
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            return_tensors="pt",
            add_generation_prompt=True
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        # Decode response
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract assistant's response (remove the question part)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        elif "[/INST]" in response:
            response = response.split("[/INST]")[-1].strip()

        return response

    def evaluate_dataset(self, annotations_file: str, image_dir: str,
                        output_file: str = None, max_samples: int = None) -> Dict:
        """Evaluate model on a dataset"""
        logger.info(f"Evaluating on dataset: {annotations_file}")

        # Load annotations
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        results = []
        correct = 0
        total = len(data)

        for i, item in enumerate(data):
            image_path = os.path.join(image_dir, item['image_path'])
            question = item['question']
            expected_answer = item['answer']

            logger.info(f"Processing sample {i+1}/{total}: {question[:50]}...")

            try:
                predicted_answer = self.generate_answer(image_path, question)

                # Simple accuracy check (exact match)
                is_correct = predicted_answer.lower().strip() == expected_answer.lower().strip()

                result = {
                    'question': question,
                    'expected_answer': expected_answer,
                    'predicted_answer': predicted_answer,
                    'correct': is_correct,
                    'image_path': item['image_path']
                }

                results.append(result)

                if is_correct:
                    correct += 1

                logger.info(f"Result: {'✓' if is_correct else '✗'}")

            except Exception as e:
                logger.error(f"Error processing sample {i+1}: {e}")
                result = {
                    'question': question,
                    'expected_answer': expected_answer,
                    'predicted_answer': f"Error: {str(e)}",
                    'correct': False,
                    'image_path': item['image_path']
                }
                results.append(result)

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0

        evaluation_results = {
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'results': results
        }

        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            logger.info(f"Results saved to {output_file}")

        logger.info(".2%")
        return evaluation_results

    def interactive_mode(self):
        """Interactive evaluation mode"""
        print("VLM Interactive Evaluation Mode")
        print("Enter 'quit' to exit")
        print("-" * 50)

        while True:
            # Get image path
            image_path = input("Image path (relative to ./data/medvlm_data/images/): ").strip()
            if image_path.lower() == 'quit':
                break

            full_image_path = os.path.join("./data/medvlm_data/images", image_path)
            if not os.path.exists(full_image_path):
                print(f"Error: Image not found at {full_image_path}")
                continue

            # Get question
            question = input("Question: ").strip()
            if not question:
                continue

            # Generate answer
            try:
                answer = self.generate_answer(full_image_path, question)
                print(f"\nAnswer: {answer}\n")
            except Exception as e:
                print(f"Error: {e}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Medical VLM")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to finetuned model")
    parser.add_argument("--annotations_file", type=str,
                       help="Path to annotations JSON file")
    parser.add_argument("--image_dir", type=str,
                       help="Directory containing images")
    parser.add_argument("--output_file", type=str,
                       help="Output file for evaluation results")
    parser.add_argument("--max_samples", type=int,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run model on")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = VLMEvaluator(args.model_path, args.device)

    if args.interactive:
        evaluator.interactive_mode()
    elif args.annotations_file and args.image_dir:
        results = evaluator.evaluate_dataset(
            args.annotations_file,
            args.image_dir,
            args.output_file,
            args.max_samples
        )
        print(".2%")
    else:
        print("Error: Either use --interactive or provide --annotations_file and --image_dir")
        sys.exit(1)

if __name__ == "__main__":
    main()