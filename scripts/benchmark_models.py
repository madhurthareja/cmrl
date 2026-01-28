import os
import sys
import asyncio
import time
import string
import logging
import pandas as pd
import numpy as np
import nltk
import re
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Add root paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'backend'))

from backend.models.medgemma_vqa import MedGemmaVQAClient, MedGemmaConfig
from backend.agents.e2h_medical_agent import E2HMedicalAgent
from backend.agents.llm_interface import OllamaLLMInterface
from backend.retrieval.pubmed_service import PubMedService

# Logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineRAGAgent:
    """
    Baseline Agent that uses Visual Findings -> PubMed Search -> LLM Generation.
    """
    def __init__(self, model_name="qwen3:1.7b"):
        self.llm = OllamaLLMInterface(model_name=model_name)
        self.pubmed = PubMedService()
        self.name = "Baseline_RAG"

    async def answer(self, question: str, visual_findings: str) -> str:
        # Search PubMed
        search_query = f"{question} {visual_findings}"[:100]
        pmids = self.pubmed.search(search_query, retmax=3)
        abstracts = []
        if pmids:
            details = self.pubmed.fetch_details(pmids)
            for d in details:
                if d.get('abstract'):
                    abstracts.append(d['abstract'][:300])
        
        context = "\n".join(abstracts)
        
        prompt = (
            f"You are a medical assistant performing a VQA task. \n"
            f"Visual Findings: {visual_findings}\n"
            f"Literature Context: {context}\n"
            f"Question: {question}\n"
            f"Instruction: Answer the question directly. If it is a Yes/No question, start with Yes or No. Be concise.\n"
            f"Answer:"
        )
        
        response = await self.llm.generate(prompt, temperature=0.1)
        return response

class ZeroShotAgent:
    """
    Simple Zero-Shot Agent: Visual Findings -> LLM -> Answer.
    """
    def __init__(self, model_name="qwen3:1.7b"):
        self.llm = OllamaLLMInterface(model_name=model_name)
        self.name = "Zero_Shot"

    async def answer(self, question: str, visual_findings: str) -> str:
        prompt = (
            f"Visual Findings: {visual_findings}\n"
            f"Question: {question}\n"
            f"Instruction: Answer the question directly. If it is a Yes/No question, start with Yes or No. Be concise.\n"
            f"Answer:"
        )
        response = await self.llm.generate(prompt, temperature=0.1)
        return response

class ModelBenchmark:
    def __init__(self, limit=None):
        self.limit = limit
        self.medgemma_config = MedGemmaConfig(base_url="http://localhost:8000")
        
        # Initialize Models
        self.models = [
            E2HMedicalAgent(),      # Proposed
            BaselineRAGAgent(),     # Baseline with PubMed
            ZeroShotAgent()         # Baseline without PubMed
        ]
        
        # Initialize Metrics Models
        print("Loading Semantic Similarity Model (all-MiniLM-L6-v2)...")
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bleu_smooth = SmoothingFunction().method1

    async def get_visual_findings(self, image_bytes, question):
        vqa_client = MedGemmaVQAClient(self.medgemma_config)
        prompt = f"Analyze this medical image for the question: '{question}'. List only objective clinical findings."
        try:
            resp = await vqa_client.answer_question_async(image_bytes=image_bytes, question=prompt)
            return resp.get("answer", "No findings.")
        except:
            return "Error retrieving findings."

    def clean_text(self, text):
        text = str(text).lower().strip()
        # Remove markdown bolding
        text = text.replace("**", "").replace("*", "")
        # Remove punctuation for word matching
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.split())

    def compute_metrics(self, pred, truth, q_type, structured_answer=None):
        pred_norm = self.clean_text(pred)
        truth_norm = self.clean_text(truth)
        
        metrics = {}
        
        # 1. Binary Truth Alignment (BTA) - Layer A
        bta = 0.0
        if q_type == "CLOSED":
            # If we have a structured answer extracted by the agent, use it (Triagic Native)
            if structured_answer and structured_answer in ["yes", "no"]:
                if structured_answer == truth_norm:
                    bta = 1.0
            else:
                # Legacy / Baseline Fallback: Check text overlap
                truth_word = truth_norm.split()[0]
                pred_words = pred_norm.split()
                
                # Direct match in first 10 words
                if truth_word in pred_words[:10]:
                    bta = 1.0
                
                # Negation check correction
                if truth_word == 'yes' and ('no' in pred_words[:5] or 'not' in pred_words[:5]):
                     bta = 0.0 
                if truth_word == 'no' and ('yes' in pred_words[:5] or 'confirm' in pred_words[:5]):
                     bta = 0.0
        else:
            # For OPEN questions, fallback to string containment for BTA
            if truth_norm in pred_norm:
                bta = 1.0
        
        metrics['binary_truth_alignment'] = bta

        # 2. Modality Awareness Score (MAS) - Layer B (Heuristic)
        # Did the model mention advanced imaging or limitations?
        mas = 0.0
        modality_keywords = ["ct", "mri", "limitations", "cannot rule out", "further evaluation", "angiogram", "scan", "advanced imaging"]
        if any(kw in pred_norm for kw in modality_keywords):
            mas = 1.0
        metrics['modality_awareness'] = mas

        # 3. Exact Match (Legacy/Surface)
        metrics['exact_match'] = bta # For now, BTA replaces Exact Match functionality but helps sorting

        # 4. Semantic Similarity
        emb1 = self.sim_model.encode(pred_norm, convert_to_tensor=True)
        emb2 = self.sim_model.encode(truth_norm, convert_to_tensor=True)
        metrics['semantic_sim'] = float(util.cos_sim(emb1, emb2)[0][0])
        
        # 5. BLEU Score
        truth_tokens = truth_norm.split()
        pred_tokens = pred_norm.split()
        if len(truth_tokens) > 0:
            metrics['bleu'] = sentence_bleu([truth_tokens], pred_tokens, smoothing_function=self.bleu_smooth)
        else:
            metrics['bleu'] = 0.0
            
        return metrics

    async def run(self):
        dataset = load_dataset("flaviagiammarino/vqa-rad", split="test", trust_remote_code=True)
        # Filter for quality samples if needed, or just take first N
        if self.limit:
            dataset = dataset.select(range(self.limit))
            
        results = []
        
        print(f"Starting Multi-Model Benchmark on {len(dataset)} samples...")
        
        # Cache visual findings
        cached_findings = {}
        
        for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
            question = sample['question']
            truth = str(sample['answer'])
            q_type = "CLOSED" if str(sample['answer']).lower() in ['yes', 'no'] else "OPEN"
            
            # 1. Visual Findings
            from io import BytesIO
            img = sample['image']
            buf = BytesIO()
            img.save(buf, format='JPEG')
            img_bytes = buf.getvalue()
            
            if i not in cached_findings:
                cached_findings[i] = await self.get_visual_findings(img_bytes, question)
            visual_findings = cached_findings[i]
            
            # 2. Evaluate Models
            for model in self.models:
                model_name = getattr(model, 'name', 'Triagic_Curriculum_Symbolic')
                structured_answer = None

                try:
                    start_time = time.time()
                    if isinstance(model, E2HMedicalAgent):
                        # Construct context string for E2H
                        ctx = f"Visual Findings: {visual_findings}\nIntent: diagnostic_investigation"
                        query = f"{question}. Findings: {visual_findings}"
                        resp = await model.process_medical_query(query, context=ctx)
                        # Extract answer from the elaborate report if possible
                        prediction = resp.answer
                        # Access the structured field
                        try:
                            structured_answer = resp.structured_answer
                        except:
                            structured_answer = None
                    else:
                        prediction = await model.answer(question, visual_findings)
                    latency = time.time() - start_time
                        
                    m = self.compute_metrics(prediction, truth, q_type, structured_answer)
                    
                    results.append({
                        "sample_id": i,
                        "question": question,
                        "type": q_type,
                        "truth": truth,
                        "model": model_name,
                        "prediction": prediction,
                        "structured_answer": structured_answer, # Log this
                        "latency": latency,
                        "binary_truth_alignment": m['binary_truth_alignment'],
                        "modality_awareness": m['modality_awareness'],
                        "semantic_sim": m['semantic_sim'],
                        "bleu": m['bleu']
                    })
                    
                except Exception as e:
                    logger.error(f"Error {model_name}: {e}")
                    
            if i % 5 == 0:
                pd.DataFrame(results).to_csv("multi_model_benchmark_partial.csv", index=False)
                
        # Final Save
        df = pd.DataFrame(results)
        df.to_csv("multi_model_benchmark_final.csv", index=False)
        
        # Print Summary
        print("\n--- Reasoning Metrics (Layer A & B) ---")
        reasoning_summary = df.groupby(['model', 'type'])[['binary_truth_alignment', 'modality_awareness']].mean()
        print(reasoning_summary)
        
        print("\n--- Surface Metrics (Legacy) ---")
        surface_summary = df.groupby(['model', 'type'])[['semantic_sim', 'bleu', 'latency']].mean()
        print(surface_summary)

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    benchmark = ModelBenchmark(limit=limit)
    asyncio.run(benchmark.run())
