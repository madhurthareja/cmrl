#!/usr/bin/env python3
"""
Benchmark Pipeline for E2H Medical Agent
Evaluates the complete pipeline without training/finetuning
"""

import argparse
import json
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd

from backend.agents.e2h_medical_agent import E2HMedicalAgent
from backend.agents.medical_agent_core import (
    MedicalQuery, MedicalDomain, DifficultyLevel,
    DomainClassifier, MedicalDifficultyClassifier
)
from backend.retrieval.medrag_system import MultiDomainRAGSystem, MockLLMInterface
try:
    # optional semantic scorer
    from sentence_transformers import SentenceTransformer
    import numpy as _np
    _ST_AVAILABLE = True
    _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _ST_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Store results of a single benchmark run"""
    query: str
    ground_truth: Optional[str]
    agent_response: str
    retrieval_contexts: List[str]
    response_time: float
    confidence_score: Optional[float]
    error: Optional[str] = None

@dataclass
class BenchmarkMetrics:
    """Aggregate benchmark metrics"""
    total_queries: int
    successful_queries: int
    failed_queries: int
    average_response_time: float
    total_time: float
    accuracy_score: Optional[float] = None
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None

class PipelineBenchmark:
    """Benchmark the complete E2H Medical Agent pipeline"""
    
    def __init__(self, config_path: Optional[str] = None, use_mock_llm: bool = False):
        """Initialize the benchmark with agent configurations"""
        self.results: List[BenchmarkResult] = []
        
        # Initialize components
        try:
            logger.info("Initializing E2H Medical Agent...")
            self.e2h_agent = E2HMedicalAgent()

            logger.info("Initializing MedRAG System...")
            self.medrag_system = MultiDomainRAGSystem()

            # Optionally inject a Mock LLM to avoid external dependencies and speed up tests
            if use_mock_llm:
                logger.info("Using Mock LLM for fast, deterministic benchmarking")
                mock_llm = MockLLMInterface()
                # Inject into main agent and specialists if present
                try:
                    self.e2h_agent.llm_interface = mock_llm
                    for s in getattr(self.e2h_agent, 'specialists', []):
                        s.llm_interface = mock_llm
                except Exception:
                    logger.warning("Failed to inject mock LLM into E2H agent (non-fatal)")

            logger.info("Pipeline components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def load_test_queries(self, test_file: str) -> List[Dict[str, Any]]:
        """Load test queries from a JSON file"""
        test_path = Path(test_file)
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        with open(test_path, 'r') as f:
            queries = json.load(f)
        
        logger.info(f"Loaded {len(queries)} test queries from {test_file}")
        return queries
    
    def create_sample_queries(self) -> List[Dict[str, Any]]:
        """Create sample medical queries for testing"""
        sample_queries = [
            {
                "query": "What are the symptoms of diabetes?",
                "category": "general_medicine",
                "expected_topics": ["diabetes", "symptoms", "blood sugar"]
            },
            {
                "query": "Explain the difference between Type 1 and Type 2 diabetes",
                "category": "endocrinology", 
                "expected_topics": ["diabetes", "type 1", "type 2", "insulin"]
            },
            {
                "query": "What is the normal blood pressure range?",
                "category": "cardiology",
                "expected_topics": ["blood pressure", "normal range", "hypertension"]
            },
            {
                "query": "How is pneumonia diagnosed?",
                "category": "pulmonology",
                "expected_topics": ["pneumonia", "diagnosis", "chest x-ray", "symptoms"]
            },
            {
                "query": "What are the side effects of aspirin?",
                "category": "pharmacology",
                "expected_topics": ["aspirin", "side effects", "medication"]
            }
        ]
        
        logger.info(f"Created {len(sample_queries)} sample queries for testing")
        return sample_queries
    
    def benchmark_single_query(self, query_data: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark a single query through the pipeline"""
        query = query_data["query"]
        ground_truth = query_data.get("ground_truth")
        
        start_time = time.time()
        
        try:
            # Step 1: Use MedRAG for retrieval
            logger.debug(f"Retrieving contexts for: {query}")
            retrieval_results = self.medrag_system.retrieve_documents(query, top_k=5)
            contexts = [getattr(doc, 'content', '') for doc in retrieval_results]

            # Step 2: Use E2H Agent for initial processing (async API)
            logger.debug(f"Processing with E2H Agent: {query}")
            try:
                # If agent exposes async `process_medical_query`, run it
                e2h_resp = asyncio.run(self.e2h_agent.process_medical_query(query, context='\n'.join(contexts)))
            except RuntimeError:
                # If already in event loop, fallback to creating a new task
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = loop.create_task(self.e2h_agent.process_medical_query(query, context='\n'.join(contexts)))
                    e2h_resp = loop.run_until_complete(task)
                else:
                    e2h_resp = asyncio.run(self.e2h_agent.process_medical_query(query, context='\n'.join(contexts)))

            # Convert AgentResponse to text
            final_response = ''
            if hasattr(e2h_resp, 'answer'):
                final_response = e2h_resp.answer
            elif isinstance(e2h_resp, dict):
                final_response = e2h_resp.get('response', '')
            else:
                final_response = str(e2h_resp)

            response_time = time.time() - start_time

            # Compare to ground truth (if provided)
            success = False
            similarity = None
            if ground_truth:
                if _ST_AVAILABLE:
                    # semantic similarity using sentence-transformers
                    emb_q = _ST_MODEL.encode([final_response, ground_truth])
                    import numpy as _np
                    sim = _np.dot(emb_q[0], emb_q[1]) / (_np.linalg.norm(emb_q[0]) * _np.linalg.norm(emb_q[1]) + 1e-8)
                    similarity = float(sim)
                    success = similarity > 0.7
                else:
                    # simple substring match
                    success = ground_truth.lower() in final_response.lower()

            return BenchmarkResult(
                query=query,
                ground_truth=ground_truth,
                agent_response=final_response,
                retrieval_contexts=contexts,
                response_time=response_time,
                confidence_score=similarity,
                error=None
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Error processing query '{query}': {e}")
            
            return BenchmarkResult(
                query=query,
                ground_truth=ground_truth,
                agent_response="",
                retrieval_contexts=[],
                response_time=response_time,
                confidence_score=None,
                error=str(e)
            )
    
    def benchmark_retrieval_only(self, query_data: Dict[str, Any]) -> BenchmarkResult:
        """Benchmark only the retrieval component"""
        query = query_data["query"]
        
        start_time = time.time()
        
        try:
            retrieval_results = self.medrag_system.retrieve_documents(query, top_k=10)
            contexts = [doc.get("content", doc.get("text", "")) for doc in retrieval_results]
            response_time = time.time() - start_time
            
            return BenchmarkResult(
                query=query,
                ground_truth=query_data.get("ground_truth"),
                agent_response="[Retrieval Only]",
                retrieval_contexts=contexts,
                response_time=response_time,
                confidence_score=None
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return BenchmarkResult(
                query=query,
                ground_truth=query_data.get("ground_truth"),
                agent_response="",
                retrieval_contexts=[],
                response_time=response_time,
                confidence_score=None,
                error=str(e)
            )
    
    def run_benchmark(self, 
                     test_queries: List[Dict[str, Any]], 
                     mode: str = "full",
                     output_file: Optional[str] = None) -> BenchmarkMetrics:
        """Run the complete benchmark suite"""
        
        logger.info(f"Starting benchmark with {len(test_queries)} queries in '{mode}' mode")
        
        total_start_time = time.time()
        self.results = []
        
        for i, query_data in enumerate(test_queries, 1):
            logger.info(f"Processing query {i}/{len(test_queries)}: {query_data['query'][:50]}...")
            
            if mode == "full":
                result = self.benchmark_single_query(query_data)
            elif mode == "retrieval":
                result = self.benchmark_retrieval_only(query_data)
            else:
                raise ValueError(f"Unknown benchmark mode: {mode}")
            
            self.results.append(result)
            
            # Log progress
            if result.error:
                logger.warning(f"Query {i} failed: {result.error}")
            else:
                logger.info(f"Query {i} completed in {result.response_time:.2f}s")
        
        total_time = time.time() - total_start_time
        
        # Calculate metrics
        successful_results = [r for r in self.results if not r.error]
        failed_results = [r for r in self.results if r.error]
        
        avg_response_time = (
            sum(r.response_time for r in successful_results) / len(successful_results)
            if successful_results else 0
        )
        
        metrics = BenchmarkMetrics(
            total_queries=len(test_queries),
            successful_queries=len(successful_results),
            failed_queries=len(failed_results),
            average_response_time=avg_response_time,
            total_time=total_time
        )
        
        # Save results if output file specified
        if output_file:
            self.save_results(output_file, metrics)
        
        self.print_summary(metrics)
        return metrics
    
    def save_results(self, output_file: str, metrics: BenchmarkMetrics):
        """Save benchmark results to file"""
        output_path = Path(output_file)
        
        # Prepare data for saving
        results_data = {
            "metrics": {
                "total_queries": metrics.total_queries,
                "successful_queries": metrics.successful_queries,
                "failed_queries": metrics.failed_queries,
                "success_rate": metrics.successful_queries / metrics.total_queries,
                "average_response_time": metrics.average_response_time,
                "total_time": metrics.total_time
            },
            "detailed_results": [
                {
                    "query": r.query,
                    "ground_truth": r.ground_truth,
                    "agent_response": r.agent_response,
                    "retrieval_contexts": r.retrieval_contexts,
                    "response_time": r.response_time,
                    "confidence_score": r.confidence_score,
                    "error": r.error,
                    "success": r.error is None
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Also save as CSV for easy analysis
        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame([
            {
                "query": r.query,
                "response_time": r.response_time,
                "success": r.error is None,
                "error": r.error,
                "num_contexts": len(r.retrieval_contexts)
            }
            for r in self.results
        ])
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV summary saved to {csv_path}")
    
    def print_summary(self, metrics: BenchmarkMetrics):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)
        print(f"Total Queries: {metrics.total_queries}")
        print(f"Successful: {metrics.successful_queries}")
        print(f"Failed: {metrics.failed_queries}")
        print(f"Success Rate: {metrics.successful_queries/metrics.total_queries*100:.1f}%")
        print(f"Average Response Time: {metrics.average_response_time:.2f}s")
        print(f"Total Benchmark Time: {metrics.total_time:.2f}s")
        print("="*60)
        
        if self.results:
            print("\nSample Results:")
            for i, result in enumerate(self.results[:3], 1):
                print(f"\n--- Query {i} ---")
                print(f"Q: {result.query}")
                if result.error:
                    print(f"Error: {result.error}")
                else:
                    print(f"Response: {result.agent_response[:200]}...")
                    print(f"Time: {result.response_time:.2f}s")
                    print(f"Contexts Retrieved: {len(result.retrieval_contexts)}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark E2H Medical Agent Pipeline")
    parser.add_argument("--test-file", type=str, help="JSON file with test queries")
    parser.add_argument("--output", type=str, default="benchmark_results.json", 
                       help="Output file for results")
    parser.add_argument("--mode", choices=["full", "retrieval"], default="full",
                       help="Benchmark mode: full pipeline or retrieval only")
    parser.add_argument("--sample", action="store_true",
                       help="Use sample queries instead of test file")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    try:
        benchmark = PipelineBenchmark()
    except Exception as e:
        logger.error(f"Failed to initialize benchmark: {e}")
        return 1
    
    # Load test queries
    if args.sample or not args.test_file:
        test_queries = benchmark.create_sample_queries()
    else:
        try:
            test_queries = benchmark.load_test_queries(args.test_file)
        except Exception as e:
            logger.error(f"Failed to load test queries: {e}")
            return 1
    
    # Run benchmark
    try:
        metrics = benchmark.run_benchmark(
            test_queries=test_queries,
            mode=args.mode,
            output_file=args.output
        )
        
        logger.info("Benchmark completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())