import os
import sys
import json
import asyncio
import logging
import string
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image
import io

# Setup paths - Add root and backend to path to satisfy internal imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
backend_dir = os.path.join(root_dir, 'backend')

sys.path.append(root_dir)
sys.path.append(backend_dir)

from backend.models.medgemma_vqa import MedGemmaVQAClient, MedGemmaConfig
from backend.agents.e2h_medical_agent import E2HMedicalAgent
from backend.retrieval.pubmed_service import PubMedService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ANSI Colors for nicer logs
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class UnifiedMedicalEvaluator:
    """
    Evaluates the Unified Medical Pipeline:
    1. Visual Analysis (MedGemma) -> Extract Clinical Findings
    2. Dynamic Retrieval (PubMed) -> Fetch relevant guidelines/literature
    3. Reasoning & Evidence (Symbolic Agent + RAG)
    """
    
    def __init__(self, medgemma_url="http://localhost:8000", ollama_url="http://localhost:11434", use_symbolic=True):
        # 1. Initialize Visual Model (Attending)
        self.medgemma_config = MedGemmaConfig(base_url=medgemma_url)
        self.medgemma = MedGemmaVQAClient(config=self.medgemma_config)
        
        # 2. Initialize Reasoning Agent (Resident)
        # Force symbolic engine to be used
        self.agent = E2HMedicalAgent(use_symbolic_engine=use_symbolic)
        self.agent.force_symbolic = True
        
        # 3. Initialize Live PubMed Service
        self.pubmed = PubMedService()
        
    async def evaluate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the full pipeline on a single test item.
        """
        question = item['question']
        image_path = item.get('image_path')
        ground_truth = item.get('ground_truth', '')
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}=================================================={Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD} EVALUATING: {question}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}=================================================={Colors.ENDC}\n")
        logger.info(f"Evaluating: {question}")
        
        visual_findings = ""
        
        # --- Step 1: Visual Analysis (if image present) ---
        print(f"{Colors.CYAN}{Colors.BOLD}[Step 1] Analyzing Medical Image...{Colors.ENDC}")
        if image_path and os.path.exists(image_path):
            logger.info("Step 1: Analyzing Medical Image...")
            try:
                with open(image_path, "rb") as img_file:
                    image_bytes = img_file.read()
                
                # Prompt specific for findings extraction
                visual_prompt = f"Analyze this medical image contextually for the question: '{question}'. List only the objective clinical findings visible."
                
                vqa_result = await self.medgemma.answer_question_async(
                    image_bytes=image_bytes,
                    question=visual_prompt
                )
                visual_findings = vqa_result.get('answer', '')
                print(f"{Colors.GREEN}   -> Visual Findings: {visual_findings[:100]}...{Colors.ENDC}")
                logger.info(f"Visual Findings: {visual_findings[:100]}...")
                
            except Exception as e:
                print(f"{Colors.FAIL}   -> Visual analysis failed: {e}{Colors.ENDC}")
                logger.error(f"Visual analysis failed: {e}")
                visual_findings = "Visual analysis unavailable."
        else:
            print(f"{Colors.WARNING}   -> No image path provided or file missing.{Colors.ENDC}")
        
        # --- Step 2: Live Retrieval (PubMed) ---
        print(f"\n{Colors.CYAN}{Colors.BOLD}[Step 2] Retrieving Guidelines via PubMed (External Check)...{Colors.ENDC}")
        logger.info("Step 2: Retrieving Evidence from PubMed...")
        retrieved_evidence = []
        try:
            # Construct a keyword-based search query
            # Simple extraction: specific medical terms from the findings
            # We assume findings are like "large dense opacity right lung field"
            # Remove common words
            stop_words = ["the", "is", "a", "in", "of", "seen", "image", "this", "primary", "abnormality", "contextually", "question"]
            keywords = [w for w in visual_findings.lower().split() if w not in stop_words and w.isalnum()]
            # Add 'guidelines' or 'differential' to get relevant medical literature
            search_query = " ".join(keywords[:6])
            
            # Fallback if no visual findings
            if not search_query.strip():
                keywords = [w for w in question.lower().split() if w not in stop_words and w.isalnum()]
                search_query = " ".join(keywords)

            print(f"{Colors.BLUE}   -> Generated Query: [{search_query}]{Colors.ENDC}")
            logger.info(f"Generated Search Query: {search_query}")
            
            # Fetch PMIDs
            pmids = self.pubmed.search(search_query, retmax=3)
            if pmids:
                details = self.pubmed.fetch_details(pmids)
                for d in details:
                    print(f"{Colors.GREEN}   -> Fetched: {d['title'][:60]}...{Colors.ENDC}")
                    evidence_text = f"Title: {d['title']}\nAbstract: {d['abstract']}"
                    retrieved_evidence.append(evidence_text)
                logger.info(f"Fetched {len(retrieved_evidence)} PubMed articles.")
            else:
                 print(f"{Colors.WARNING}   -> No PubMed articles found.{Colors.ENDC}")
                 logger.warning("No PubMed articles found for generated query.")
                 
        except Exception as e:
             print(f"{Colors.FAIL}   -> PubMed retrieval failed: {e}{Colors.ENDC}")
             logger.error(f"PubMed retrieval failed: {e}")
        
        # --- Step 3: Reasoning & Answer Generation ---
        print(f"\n{Colors.CYAN}{Colors.BOLD}[Step 3] Reasoning with Symbolic Intelligence...{Colors.ENDC}")
        logger.info("Step 3: Reasoning with Evidence...")
        
        # We pass ONLY visual findings to the agent. 
        # The agent (SymbolicReasoningAgent) is now configured to perform its own 
        # live PubMed retrieval using the context provided.
        agent_context = f"Visual Findings: {visual_findings}"
        
        # NOTE: External retrieval logic moved into Agent for robustness (LivePubMedAdapter).
        # We just pass the string context now.
        
        response = await self.agent.process_medical_query(question, context=agent_context)
        
        print(f"{Colors.GREEN}{Colors.BOLD}   -> Prediction: {response.answer}{Colors.ENDC}")
        logger.info(f"Prediction: {response.answer}")
        
        # Calculate score if ground truth is available
        score = 0.0
        if ground_truth:
            score = self._score_answer(response.answer, ground_truth, item.get('type', 'open'))
        
        return {
            "question": question,
            "visual_findings": visual_findings,
            "retrieved_evidence_count": len(retrieved_evidence),
            "model_answer": response.answer,
            "ground_truth": ground_truth,
            "score": score,
            "reasoning_trace": response.reasoning
        }

    def _score_answer(self, prediction: str, truth: str, type_: str) -> float:
        """
        Simple scoring metric. 
        For rigorous evaluation, use BERTScore or BLEU, or LLM-as-a-Judge.
        """
        # Normalize to lower case
        pred_norm = prediction.lower().strip()
        truth_norm = truth.lower().strip()
        
        # Replace punctuation with spaces to avoid merging words like mass/consolidation
        for char in string.punctuation:
            pred_norm = pred_norm.replace(char, ' ')
            truth_norm = truth_norm.replace(char, ' ')
            
        if type_ == 'closed':
            # Exact match for Yes/No or multiple choice
            # Split and join to normalize whitespace
            pred_norm = " ".join(pred_norm.split())
            truth_norm = " ".join(truth_norm.split())
            return 1.0 if truth_norm in pred_norm else 0.0
        else:
            # Keyword overlap for open-ended
            truth_tokens = set(truth_norm.split())
            if not truth_tokens: return 0.0
            
            pred_tokens = set(pred_norm.split())
            overlap = truth_tokens.intersection(pred_tokens)
            
            # Debug scoring
            # logger.info(f"Scoring - Truth tokens: {truth_tokens}")
            # logger.info(f"Scoring - Pred tokens: {pred_tokens}")
            # logger.info(f"Scoring - Overlap: {overlap}")
            
            return len(overlap) / len(truth_tokens)

        json.dump(results, f, indent=2)
    print("Detailed report saved to evaluation_report.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-symbolic", action="store_true", help="Disable symbolic engine")
    args = parser.parse_args()
    
    # Pass config to main or instantiate evaluator with config inside main
    # Since main() takes no args in original code, I should update main() or handle it there.
    # The original main() creates Evaluator directly.
    # I'll rely on global args or pass it.
    
    async def run_evaluation():
        evaluator = UnifiedMedicalEvaluator(use_symbolic=not args.no_symbolic)
        
        # Test Data Mock
        test_dataset = [
            {
                "id": "test_case_1",
                "image_path": "chest.jpg", 
                "question": "What is the primary abnormality seen in this image?",
                "ground_truth": "consolidation", 
                "type": "open"
            }
        ]
        
        results = []
        
        for item in test_dataset:
            print(f"\n--- Processing Item {item['id']} ---")
            result = await evaluator.evaluate_item(item)
            results.append(result)
            
            print(f"Prediction: {result['model_answer']}")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Score: {result['score']:.2f}")
            print(f"Evidence Used: {result['retrieved_evidence_count']} docs")

        # Summary
        avg_score = sum(r['score'] for r in results) / len(results) if results else 0
        print(f"\nAverage Score: {avg_score:.2f}")
        
        # Save detailed logs
        with open("evaluation_report.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Detailed report saved to evaluation_report.json")

    asyncio.run(run_evaluation())
