"""
MedVLM Benchmark Runner
Executes comprehensive benchmarking of E2H Medical Agent against MedVLM frameworks
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from dataclasses import asdict

from medvlm_extension import (
    MultimodalMedicalAgent, 
    MedVLMBenchmarkEvaluator,
    MultimodalMedicalQuery,
    MedicalImageInput,
    MedicalImageType
)
from medvlm_benchmark_setup import MedVLMDatasetManager, MedVLMBenchmarkConfig
from medical_agent_core import MedicalDomain, DifficultyLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveMedVLMBenchmark:
    """Comprehensive benchmarking suite for medical vision-language models"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.dataset_manager = MedVLMDatasetManager()
        self.benchmark_config = MedVLMBenchmarkConfig()
        self.results = {}
        
    async def run_full_benchmark_suite(self) -> Dict:
        """Run comprehensive benchmarking across all datasets and curriculum levels"""
        
        logger.info("🚀 Starting Comprehensive MedVLM Benchmark Suite")
        start_time = time.time()
        
        # Initialize multimodal medical agent
        agent = MultimodalMedicalAgent(model_name="ollama", vision_model="llava")
        evaluator = MedVLMBenchmarkEvaluator(agent)
        
        # Benchmark configuration
        datasets = ["vqa_rad", "pathvqa", "slake"]
        curriculum_levels = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        
        benchmark_results = {
            "experiment_info": {
                "name": "E2H Medical Agent MedVLM Benchmark",
                "timestamp": datetime.now().isoformat(),
                "model": "E2H Medical Agent with Vision",
                "datasets_evaluated": datasets,
                "curriculum_levels": [level.value for level in curriculum_levels]
            },
            "dataset_results": {},
            "curriculum_analysis": {},
            "comparative_analysis": {},
            "summary_metrics": {}
        }
        
        # Run evaluation on each dataset
        for dataset_name in datasets:
            logger.info(f"📊 Evaluating on {dataset_name.upper()} dataset")
            
            dataset_results = await self.evaluate_dataset_comprehensive(
                dataset_name, agent, curriculum_levels
            )
            benchmark_results["dataset_results"][dataset_name] = dataset_results
        
        # Analyze curriculum learning effectiveness
        benchmark_results["curriculum_analysis"] = self.analyze_curriculum_effectiveness(
            benchmark_results["dataset_results"]
        )
        
        # Compare against baseline models (simulated)
        benchmark_results["comparative_analysis"] = await self.compare_against_baselines(
            benchmark_results["dataset_results"]
        )
        
        # Calculate summary metrics
        benchmark_results["summary_metrics"] = self.calculate_summary_metrics(
            benchmark_results["dataset_results"]
        )
        
        # Save comprehensive results
        results_file = self.output_dir / f"comprehensive_benchmark_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        total_time = time.time() - start_time
        benchmark_results["execution_time_seconds"] = total_time
        
        logger.info(f"✅ Benchmark completed in {total_time:.2f} seconds")
        logger.info(f"📁 Results saved to {results_file}")
        
        return benchmark_results
    
    async def evaluate_dataset_comprehensive(
        self, 
        dataset_name: str, 
        agent: MultimodalMedicalAgent,
        curriculum_levels: List[DifficultyLevel]
    ) -> Dict:
        """Comprehensive evaluation on a single dataset across curriculum levels"""
        
        # Load dataset samples (mock for now)
        if dataset_name == "vqa_rad":
            samples = self.dataset_manager.create_mock_vqa_rad_data()["samples"]
        elif dataset_name == "pathvqa":
            samples = self.dataset_manager.create_mock_pathvqa_data()["samples"]
        else:
            # Mock SLAKE data
            samples = [
                {
                    "id": "slake_001",
                    "question": "What abnormality is visible in this brain MRI?",
                    "answer": "Hyperintense lesion suggesting demyelination",
                    "domain": "neurology",
                    "difficulty": "hard",
                    "image_type": "mri"
                }
            ]
        
        dataset_results = {
            "dataset_name": dataset_name,
            "total_samples": len(samples),
            "curriculum_results": {},
            "performance_metrics": {},
            "error_analysis": {}
        }
        
        # Evaluate at each curriculum level
        for curriculum_level in curriculum_levels:
            logger.info(f"  📝 Evaluating at {curriculum_level.value} level")
            
            level_results = await self.evaluate_curriculum_level(
                samples, agent, curriculum_level, dataset_name
            )
            
            dataset_results["curriculum_results"][curriculum_level.value] = level_results
        
        # Calculate cross-curriculum metrics
        dataset_results["performance_metrics"] = self.calculate_dataset_metrics(
            dataset_results["curriculum_results"]
        )
        
        return dataset_results
    
    async def evaluate_curriculum_level(
        self, 
        samples: List[Dict], 
        agent: MultimodalMedicalAgent,
        curriculum_level: DifficultyLevel,
        dataset_name: str
    ) -> Dict:
        """Evaluate agent performance at specific curriculum level"""
        
        correct_predictions = 0
        total_predictions = 0
        response_times = []
        confidence_scores = []
        detailed_results = []
        
        for sample in samples:
            start_time = time.time()
            
            try:
                # Create multimodal query (mock image for now)
                query = MultimodalMedicalQuery(
                    text_query=sample["question"],
                    images=[],  # Would load actual images in real implementation
                    domain=self.map_domain(sample.get("domain", "general"))
                )
                
                # Get agent response
                response = await agent.process_multimodal_query(query, curriculum_level)
                
                # Evaluate response quality
                prediction_correct = self.evaluate_response_quality(
                    response["text_response"], sample["answer"]
                )
                
                if prediction_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Record metrics
                response_time = time.time() - start_time
                response_times.append(response_time)
                confidence_scores.append(response.get("confidence_score", 0.0))
                
                detailed_results.append({
                    "sample_id": sample.get("id", f"sample_{total_predictions}"),
                    "question": sample["question"],
                    "ground_truth": sample["answer"],
                    "prediction": response["text_response"],
                    "correct": prediction_correct,
                    "confidence": response.get("confidence_score", 0.0),
                    "response_time": response_time,
                    "curriculum_level": curriculum_level.value
                })
                
            except Exception as e:
                logger.error(f"Error evaluating sample: {e}")
                detailed_results.append({
                    "sample_id": sample.get("id", f"sample_{total_predictions}"),
                    "error": str(e),
                    "correct": False
                })
                total_predictions += 1
        
        # Calculate level-specific metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_response_time = np.mean(response_times) if response_times else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
            "avg_response_time": avg_response_time,
            "avg_confidence": avg_confidence,
            "detailed_results": detailed_results[:5]  # Sample of detailed results
        }
    
    def evaluate_response_quality(self, prediction: str, ground_truth: str) -> bool:
        """Evaluate quality of model response against ground truth"""
        # Simple keyword-based evaluation (would use more sophisticated metrics in practice)
        pred_words = set(prediction.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        # Calculate word overlap
        overlap = len(pred_words.intersection(truth_words))
        overlap_ratio = overlap / len(truth_words) if truth_words else 0.0
        
        # Consider correct if significant overlap
        return overlap_ratio > 0.3
    
    def map_domain(self, domain_string: str) -> MedicalDomain:
        """Map string domain to MedicalDomain enum"""
        domain_mapping = {
            "radiology": MedicalDomain.RADIOLOGY,
            "pathology": MedicalDomain.PATHOLOGY,
            "cardiology": MedicalDomain.CARDIOLOGY,
            "neurology": MedicalDomain.NEUROLOGY,
            "dermatology": MedicalDomain.DERMATOLOGY,
            "general": MedicalDomain.GENERAL
        }
        return domain_mapping.get(domain_string.lower(), MedicalDomain.GENERAL)
    
    def analyze_curriculum_effectiveness(self, dataset_results: Dict) -> Dict:
        """Analyze effectiveness of curriculum learning across datasets"""
        
        curriculum_analysis = {
            "curriculum_progression": {},
            "difficulty_scaling": {},
            "learning_efficiency": {}
        }
        
        # Analyze accuracy progression across curriculum levels
        for dataset_name, results in dataset_results.items():
            curriculum_accuracies = {}
            
            for level, level_results in results["curriculum_results"].items():
                curriculum_accuracies[level] = level_results["accuracy"]
            
            curriculum_analysis["curriculum_progression"][dataset_name] = curriculum_accuracies
            
            # Calculate curriculum learning benefit
            if "easy" in curriculum_accuracies and "hard" in curriculum_accuracies:
                learning_gain = curriculum_accuracies["hard"] - curriculum_accuracies["easy"]
                curriculum_analysis["difficulty_scaling"][dataset_name] = {
                    "learning_gain": learning_gain,
                    "relative_improvement": learning_gain / curriculum_accuracies["easy"] if curriculum_accuracies["easy"] > 0 else 0.0
                }
        
        return curriculum_analysis
    
    async def compare_against_baselines(self, dataset_results: Dict) -> Dict:
        """Compare E2H Medical Agent against baseline MedVLM models"""
        
        # Simulated baseline results (would come from actual model evaluation)
        baseline_results = {
            "llava_med": {
                "vqa_rad_accuracy": 0.72,
                "pathvqa_accuracy": 0.68,
                "slake_accuracy": 0.65
            },
            "med_flamingo": {
                "vqa_rad_accuracy": 0.75,
                "pathvqa_accuracy": 0.71, 
                "slake_accuracy": 0.69
            },
            "chatcad": {
                "vqa_rad_accuracy": 0.70,
                "pathvqa_accuracy": 0.66,
                "slake_accuracy": 0.63
            }
        }
        
        # Calculate E2H agent average performance
        e2h_performance = {}
        for dataset_name, results in dataset_results.items():
            # Average across curriculum levels
            accuracies = [
                level_results["accuracy"] 
                for level_results in results["curriculum_results"].values()
            ]
            e2h_performance[f"{dataset_name}_accuracy"] = np.mean(accuracies)
        
        # Compare against baselines
        comparative_analysis = {
            "e2h_medical_agent": e2h_performance,
            "baseline_models": baseline_results,
            "performance_comparison": {},
            "competitive_analysis": {}
        }
        
        # Calculate competitive metrics
        for dataset in ["vqa_rad", "pathvqa", "slake"]:
            dataset_key = f"{dataset}_accuracy"
            if dataset_key in e2h_performance:
                e2h_score = e2h_performance[dataset_key]
                
                comparative_analysis["performance_comparison"][dataset] = {
                    "e2h_score": e2h_score,
                    "best_baseline": max([
                        baseline[dataset_key] 
                        for baseline in baseline_results.values()
                        if dataset_key in baseline
                    ]),
                    "competitive": e2h_score > 0.7  # Threshold for competitive performance
                }
        
        return comparative_analysis
    
    def calculate_dataset_metrics(self, curriculum_results: Dict) -> Dict:
        """Calculate comprehensive metrics for a dataset"""
        
        all_accuracies = [results["accuracy"] for results in curriculum_results.values()]
        all_confidences = [results["avg_confidence"] for results in curriculum_results.values()]
        all_response_times = [results["avg_response_time"] for results in curriculum_results.values()]
        
        return {
            "overall_accuracy": np.mean(all_accuracies),
            "accuracy_std": np.std(all_accuracies),
            "avg_confidence": np.mean(all_confidences),
            "avg_response_time": np.mean(all_response_times),
            "curriculum_consistency": 1.0 - np.std(all_accuracies),  # Higher is more consistent
            "best_curriculum_level": max(curriculum_results.keys(), key=lambda k: curriculum_results[k]["accuracy"])
        }
    
    def calculate_summary_metrics(self, dataset_results: Dict) -> Dict:
        """Calculate overall summary metrics across all datasets"""
        
        all_accuracies = []
        all_confidences = []
        all_response_times = []
        
        for dataset_results_single in dataset_results.values():
            for level_results in dataset_results_single["curriculum_results"].values():
                all_accuracies.append(level_results["accuracy"])
                all_confidences.append(level_results["avg_confidence"])
                all_response_times.append(level_results["avg_response_time"])
        
        return {
            "overall_accuracy": np.mean(all_accuracies),
            "overall_confidence": np.mean(all_confidences),
            "overall_response_time": np.mean(all_response_times),
            "performance_consistency": 1.0 - np.std(all_accuracies),
            "datasets_evaluated": len(dataset_results),
            "total_evaluations": sum([
                sum([
                    level_results["total_predictions"] 
                    for level_results in dataset_results_single["curriculum_results"].values()
                ])
                for dataset_results_single in dataset_results.values()
            ])
        }
    
    def generate_benchmark_report(self, results: Dict) -> str:
        """Generate human-readable benchmark report"""
        
        report = []
        report.append("# MedVLM Benchmark Report")
        report.append(f"**Generated:** {results['experiment_info']['timestamp']}")
        report.append(f"**Model:** {results['experiment_info']['model']}")
        report.append("")
        
        # Summary metrics
        summary = results["summary_metrics"]
        report.append("## Summary Metrics")
        report.append(f"- **Overall Accuracy:** {summary['overall_accuracy']:.3f}")
        report.append(f"- **Average Confidence:** {summary['overall_confidence']:.3f}")
        report.append(f"- **Average Response Time:** {summary['overall_response_time']:.3f}s")
        report.append(f"- **Performance Consistency:** {summary['performance_consistency']:.3f}")
        report.append(f"- **Total Evaluations:** {summary['total_evaluations']}")
        report.append("")
        
        # Dataset-specific results
        report.append("## Dataset Results")
        for dataset_name, dataset_results in results["dataset_results"].items():
            report.append(f"### {dataset_name.upper()}")
            metrics = dataset_results["performance_metrics"]
            report.append(f"- **Overall Accuracy:** {metrics['overall_accuracy']:.3f}")
            report.append(f"- **Best Curriculum Level:** {metrics['best_curriculum_level']}")
            report.append(f"- **Curriculum Consistency:** {metrics['curriculum_consistency']:.3f}")
            report.append("")
        
        # Curriculum learning analysis
        curriculum_analysis = results["curriculum_analysis"]
        report.append("## Curriculum Learning Analysis")
        for dataset_name, progression in curriculum_analysis["curriculum_progression"].items():
            report.append(f"### {dataset_name.upper()} Progression")
            for level, accuracy in progression.items():
                report.append(f"- **{level.capitalize()}:** {accuracy:.3f}")
            report.append("")
        
        return "\n".join(report)

async def main():
    """Main benchmarking execution"""
    
    print("🏥 MedVLM Comprehensive Benchmarking Suite")
    print("=" * 50)
    
    # Initialize benchmark runner
    benchmark = ComprehensiveMedVLMBenchmark()
    
    try:
        # Run comprehensive benchmark
        results = await benchmark.run_full_benchmark_suite()
        
        # Generate and save report
        report = benchmark.generate_benchmark_report(results)
        report_file = benchmark.output_dir / "benchmark_report.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print("\n" + "=" * 50)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        print(f"Overall Accuracy: {results['summary_metrics']['overall_accuracy']:.3f}")
        print(f"Total Evaluations: {results['summary_metrics']['total_evaluations']}")
        print(f"Execution Time: {results['execution_time_seconds']:.2f}s")
        print(f"Results File: {benchmark.output_dir / 'comprehensive_benchmark_*.json'}")
        print(f"Report File: {report_file}")
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
