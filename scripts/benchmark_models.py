import os
import sys
import asyncio
import time
import logging
import json
import pandas as pd
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

# Logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedGemmaAgent:
    """Single-model medical benchmark agent using MedGemma."""

    def __init__(self, medgemma_config: MedGemmaConfig):
        self.client = MedGemmaVQAClient(medgemma_config)
        self.name = "MedGemma"

    def _extract_json(self, text: str):
        """Best-effort JSON extraction from model output."""
        if not text:
            return None

        cleaned = text.strip().replace("```json", "").replace("```", "")
        # First try direct parse
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Fallback: find first balanced JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start:end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None

    async def answer(self, image_bytes: bytes, question: str):
        prompt = (
            "You are a medical VQA specialist.\n"
            f"Question: {question}\n"
            "Return output as valid JSON (no markdown) with EXACT keys:\n"
            "direct_answer, task_answer, confidence, evidence_strength, modality_limits_acknowledged, escalation_needed, recommended_next_step, reasoning\n"
            "Rules:\n"
            "- task_answer must be one of: yes, no, indeterminate, not_applicable\n"
            "- confidence must be a float between 0 and 1\n"
            "- evidence_strength must be one of: weak, moderate, strong\n"
            "- modality_limits_acknowledged must be true/false\n"
            "- escalation_needed must be true/false\n"
            "- direct_answer should be concise and clinically grounded\n"
            "- reasoning should be short and factual"
        )
        response = await self.client.answer_question_async(image_bytes=image_bytes, question=prompt)
        answer = response.get("answer", "")
        parsed = self._extract_json(answer)

        if isinstance(parsed, dict):
            return {
                "raw": answer,
                "direct_answer": str(parsed.get("direct_answer", "")).strip(),
                "task_answer": str(parsed.get("task_answer", "not_applicable")).strip().lower(),
                "confidence": parsed.get("confidence", 0.5),
                "evidence_strength": str(parsed.get("evidence_strength", "moderate")).strip().lower(),
                "modality_limits_acknowledged": bool(parsed.get("modality_limits_acknowledged", False)),
                "escalation_needed": bool(parsed.get("escalation_needed", False)),
                "recommended_next_step": str(parsed.get("recommended_next_step", "")).strip(),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
                "json_valid": True,
            }

        # Fallback keeps benchmark robust even if model returns plain text.
        return {
            "raw": answer,
            "direct_answer": answer,
            "task_answer": "not_applicable",
            "confidence": 0.5,
            "evidence_strength": "moderate",
            "modality_limits_acknowledged": False,
            "escalation_needed": False,
            "recommended_next_step": "",
            "reasoning": "",
            "json_valid": False,
        }

class ModelBenchmark:
    def __init__(self, limit=None):
        self.limit = limit
        self.medgemma_config = MedGemmaConfig(base_url="http://localhost:8000")
        self.model = MedGemmaAgent(self.medgemma_config)
        
        # Initialize Metrics Models
        print("Loading Semantic Similarity Model (all-MiniLM-L6-v2)...")
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bleu_smooth = SmoothingFunction().method1

    def extract_structured_answer(self, prediction: str, q_type: str):
        if q_type != "CLOSED":
            return None

        pred_norm = self.clean_text(prediction)
        words = pred_norm.split()
        head = words[:12]
        if "yes" in head:
            return "yes"
        if "no" in head:
            return "no"
        if "indeterminate" in words or "uncertain" in words or "cannot" in head:
            return "indeterminate"
        return None

    def normalize_structured_answer(self, task_answer: str, q_type: str):
        if q_type != "CLOSED":
            return None
        if task_answer in ["yes", "no", "indeterminate"]:
            return task_answer
        return None

    def clean_text(self, text):
        text = str(text).lower().strip()
        # Remove markdown bolding
        text = text.replace("**", "").replace("*", "")
        # Remove punctuation for word matching
        text = re.sub(r'[^\w\s]', ' ', text)
        return " ".join(text.split())

    def compute_metrics(self, pred, truth, q_type, structured_answer=None, meta=None):
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

        # 3. Binary Task Compliance (BTC): schema compliance for CLOSED tasks.
        btc = 0.0
        if q_type == "CLOSED" and structured_answer in ["yes", "no", "indeterminate"]:
            btc = 1.0
        metrics['binary_task_compliance'] = btc

        # 4. Epistemic Gate checks (heuristic neuro-symbolic validation).
        contradiction = 0.0
        unsupported_certainty = 0.0
        escalation_mismatch = 0.0
        modality_mismatch = 0.0

        if meta is not None:
            confidence = float(meta.get("confidence", 0.5))
            evidence_strength = str(meta.get("evidence_strength", "moderate")).lower()
            escalation_needed = bool(meta.get("escalation_needed", False))
            modality_ack = bool(meta.get("modality_limits_acknowledged", False))

            # Detect answer polarity contradiction between structured and free text head.
            inferred = self.extract_structured_answer(pred, q_type)
            if q_type == "CLOSED" and structured_answer in ["yes", "no"] and inferred in ["yes", "no"]:
                if structured_answer != inferred:
                    contradiction = 1.0

            # High confidence with weak evidence is epistemically suspicious.
            if confidence >= 0.85 and evidence_strength == "weak":
                unsupported_certainty = 1.0

            # Strong confidence + strong evidence should not always escalate.
            if confidence >= 0.85 and evidence_strength == "strong" and escalation_needed:
                escalation_mismatch = 1.0

            # If modality limits are ignored while strong certainty language appears.
            certainty_words = ["definitive", "certain", "confirmed", "conclusive"]
            if (not modality_ack) and any(w in pred_norm for w in certainty_words) and q_type == "CLOSED":
                modality_mismatch = 1.0

        metrics['contradiction_flag'] = contradiction
        metrics['unsupported_certainty_flag'] = unsupported_certainty
        metrics['escalation_mismatch_flag'] = escalation_mismatch
        metrics['modality_mismatch_flag'] = modality_mismatch

        epistemic_penalty = (contradiction + unsupported_certainty + escalation_mismatch + modality_mismatch) / 4.0
        metrics['epistemic_validity'] = 1.0 - epistemic_penalty

        # 5. Actionability score: escalation should align with uncertainty profile.
        actionability = 0.5
        if meta is not None:
            confidence = float(meta.get("confidence", 0.5))
            evidence_strength = str(meta.get("evidence_strength", "moderate")).lower()
            escalation_needed = bool(meta.get("escalation_needed", False))

            if evidence_strength == "weak" or confidence < 0.55:
                actionability = 1.0 if escalation_needed else 0.0
            elif evidence_strength == "strong" and confidence >= 0.8:
                actionability = 1.0 if not escalation_needed else 0.0
            else:
                actionability = 0.7
        metrics['actionability'] = actionability

        # 6. Triadic Reasoning Score (publishable aggregate).
        metrics['triadic_reasoning_score'] = (
            0.45 * metrics['binary_truth_alignment']
            + 0.35 * metrics['epistemic_validity']
            + 0.20 * metrics['actionability']
        )

        # 7. Exact Match (Legacy/Surface)
        metrics['exact_match'] = bta # For now, BTA replaces Exact Match functionality but helps sorting

        # 8. Semantic Similarity
        emb1 = self.sim_model.encode(pred_norm, convert_to_tensor=True)
        emb2 = self.sim_model.encode(truth_norm, convert_to_tensor=True)
        metrics['semantic_sim'] = float(util.cos_sim(emb1, emb2)[0][0])
        
        # 9. BLEU Score
        truth_tokens = truth_norm.split()
        pred_tokens = pred_norm.split()
        if len(truth_tokens) > 0:
            metrics['bleu'] = sentence_bleu([truth_tokens], pred_tokens, smoothing_function=self.bleu_smooth)
        else:
            metrics['bleu'] = 0.0
            
        return metrics

    async def run(self):
        dataset = load_dataset("flaviagiammarino/vqa-rad", split="test")
        # Filter for quality samples if needed, or just take first N
        if self.limit:
            dataset = dataset.select(range(self.limit))
            
        results = []
        
        print(f"Starting MedGemma Benchmark on {len(dataset)} samples...")
        
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
            
            try:
                start_time = time.time()
                model_out = await self.model.answer(img_bytes, question)
                latency = time.time() - start_time

                prediction = model_out.get("direct_answer", "")
                task_answer = model_out.get("task_answer", "not_applicable")
                confidence = model_out.get("confidence", 0.5)
                evidence_strength = model_out.get("evidence_strength", "moderate")
                modality_limits_ack = model_out.get("modality_limits_acknowledged", False)
                escalation_needed = model_out.get("escalation_needed", False)
                recommended_next_step = model_out.get("recommended_next_step", "")
                reasoning = model_out.get("reasoning", "")
                json_valid = model_out.get("json_valid", False)

                structured_answer = self.normalize_structured_answer(task_answer, q_type)
                if structured_answer is None:
                    structured_answer = self.extract_structured_answer(prediction, q_type)

                m = self.compute_metrics(
                    prediction,
                    truth,
                    q_type,
                    structured_answer,
                    meta={
                        "confidence": confidence,
                        "evidence_strength": evidence_strength,
                        "escalation_needed": escalation_needed,
                        "modality_limits_acknowledged": modality_limits_ack,
                    }
                )

                results.append({
                    "sample_id": i,
                    "question": question,
                    "type": q_type,
                    "truth": truth,
                    "model": self.model.name,
                    "prediction": prediction,
                    "structured_answer": structured_answer,
                    "json_valid": json_valid,
                    "task_answer": task_answer,
                    "confidence": confidence,
                    "evidence_strength": evidence_strength,
                    "modality_limits_acknowledged": modality_limits_ack,
                    "escalation_needed": escalation_needed,
                    "recommended_next_step": recommended_next_step,
                    "reasoning": reasoning,
                    "latency": latency,
                    "binary_truth_alignment": m['binary_truth_alignment'],
                    "modality_awareness": m['modality_awareness'],
                    "binary_task_compliance": m['binary_task_compliance'],
                    "epistemic_validity": m['epistemic_validity'],
                    "actionability": m['actionability'],
                    "triadic_reasoning_score": m['triadic_reasoning_score'],
                    "contradiction_flag": m['contradiction_flag'],
                    "unsupported_certainty_flag": m['unsupported_certainty_flag'],
                    "escalation_mismatch_flag": m['escalation_mismatch_flag'],
                    "modality_mismatch_flag": m['modality_mismatch_flag'],
                    "semantic_sim": m['semantic_sim'],
                    "bleu": m['bleu']
                })
            except Exception as e:
                logger.error(f"Error {self.model.name}: {e}")
                    
            if i % 5 == 0:
                pd.DataFrame(results).to_csv("multi_model_benchmark_partial.csv", index=False)
                
        # Final Save
        df = pd.DataFrame(results)
        df.to_csv("multi_model_benchmark_final.csv", index=False)

        if df.empty:
            print("No successful predictions were generated. Check MedGemma server availability at http://localhost:8000.")
            return
        
        # Print Summary
        print("\n--- Reasoning Metrics (Layer A & B) ---")
        reasoning_summary = df.groupby(['model', 'type'])[
            [
                'binary_truth_alignment',
                'modality_awareness',
                'binary_task_compliance',
                'epistemic_validity',
                'actionability',
                'triadic_reasoning_score',
            ]
        ].mean()
        print(reasoning_summary)

        print("\n--- Epistemic Gate Diagnostics ---")
        gate_summary = df.groupby(['model', 'type'])[
            [
                'contradiction_flag',
                'unsupported_certainty_flag',
                'escalation_mismatch_flag',
                'modality_mismatch_flag',
            ]
        ].mean()
        print(gate_summary)
        
        print("\n--- Surface Metrics (Legacy) ---")
        surface_summary = df.groupby(['model', 'type'])[['semantic_sim', 'bleu', 'latency']].mean()
        print(surface_summary)

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    benchmark = ModelBenchmark(limit=limit)
    asyncio.run(benchmark.run())
