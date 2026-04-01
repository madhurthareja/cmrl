
import os
import sys
import json
import asyncio
import logging
import io
import string
import time
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Add root paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'backend'))

from backend.models.medgemma_vqa import MedGemmaVQAClient, MedGemmaConfig

# Logging setup
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vqa_rad_benchmark.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VQARADEvaluator:
    def __init__(self, limit: int = None, use_symbolic: bool = True):
        self.medgemma_url = "http://localhost:8000"
        self.medgemma_config = MedGemmaConfig(base_url=self.medgemma_url)
        self.limit = limit
        
        # Load Dataset
        logger.info("Loading VQA-RAD dataset...")
        self.dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
        if self.limit:
            self.dataset = self.dataset.select(range(self.limit))
        logger.info(f"Loaded {len(self.dataset)} samples.")

    def _pil_to_bytes(self, image: Image.Image) -> bytes:
        img_byte_arr = io.BytesIO()
        # Check format, default to PNG if not available
        fmt = image.format if image.format else 'PNG'
        image.save(img_byte_arr, format=fmt)
        return img_byte_arr.getvalue()

    def score_answer(self, prediction: str, truth: str, type_: str) -> float:
        # Normalize
        pred_norm = prediction.lower().strip()
        truth_norm = truth.lower().strip()
        
        # Remove punctuation/standardize
        for char in string.punctuation:
            pred_norm = pred_norm.replace(char, ' ')
            truth_norm = truth_norm.replace(char, ' ')
            
        pred_norm = " ".join(pred_norm.split())
        truth_norm = " ".join(truth_norm.split())
        
        if type_ == "CLOSED":
            # Exact match (binary/multiple choice)
            # Check if reasoner output contains truth
            # Our Symbolic Agent is verbose ("The diagnosis is..."). 
            # We strictly verify if the keyword is present.
            return 1.0 if truth_norm in pred_norm else 0.0
        else:
            # Open ended - Token overlap (bag of words)
            truth_tokens = set(truth_norm.split())
            if not truth_tokens: return 0.0
            pred_tokens = set(pred_norm.split())
            
            overlap = truth_tokens.intersection(pred_tokens)
            return len(overlap) / len(truth_tokens)

    async def evaluate_sample(self, sample):
        image = sample['image']
        question = sample['question']
        truth = sample['answer']
        # VQA-RAD has 'answer_type' but dataset features might name it differently.
        # Looking at HuggingFace viewer, sometimes it's implied. 
        # For now, if truth is "yes"/"no", consider CLOSED.
        type_ = "CLOSED" if str(truth).lower() in ['yes', 'no'] else "OPEN"
        
        # 1. Convert Image
        image_bytes = self._pil_to_bytes(image)
        
        # MedGemma path: perform visual analysis and final answer in one pass.
        vqa_client = MedGemmaVQAClient(self.medgemma_config)

        answer_prompt = (
            f"Question: {question}\n"
            "Provide a clinically grounded answer based on the image. "
            "If the question is yes/no, begin with Yes or No. "
            "If uncertain due to modality limits, state uncertainty briefly."
        )
        
        try:
            vqa_response = await vqa_client.answer_question_async(
                image_bytes=image_bytes,
                question=answer_prompt
            )
            prediction = vqa_response.get("answer", "")
        except Exception as e:
            logger.error(f"VQA Error: {e}")
            prediction = "Error in reasoning."

        score = self.score_answer(prediction, str(truth), type_)
        
        return {
            "question": question,
            "truth": str(truth),
            "prediction": prediction,
            "score": score,
            "type": type_
        }

    async def run(self):
        results = []
        print(f"Starting Benchmark on {len(self.dataset)} samples...")
        
        for i, sample in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            res = await self.evaluate_sample(sample)
            results.append(res)
            
            # Rate limiting or nice logging
            if i % 5 == 0:
                # Save partial
                df = pd.DataFrame(results)
                df.to_csv("benchmark_vqa_rad_partial.csv", index=False)
        
        # Final Save
        df = pd.DataFrame(results)
        df.to_csv("benchmark_vqa_rad_final.csv", index=False)
        
        # Summary
        print("\n--- Benchmark Complete ---")
        print(f"Total Samples: {len(df)}")
        print(f"Average Accuracy: {df['score'].mean():.4f}")
        print(f"Accuracy (Closed): {df[df['type']=='CLOSED']['score'].mean():.4f}")
        print(f"Accuracy (Open): {df[df['type']=='OPEN']['score'].mean():.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Limit number of samples for testing")
    args = parser.parse_args()
    
    evaluator = VQARADEvaluator(limit=args.limit)
    asyncio.run(evaluator.run())
