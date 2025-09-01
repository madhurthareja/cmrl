#!/usr/bin/env python3
"""
MedVLM Evaluation Suite
Comprehensive benchmarking of E2H Medical Agent against state-of-the-art MedVLM frameworks
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import logging
from PIL import Image
import torch
import requests
from io import BytesIO

# Import your existing components
from e2h_medical_agent import E2HMedicalAgent
from medvlm_extension import MultimodalMedicalAgent
from medical_agent_core import MedicalDomain, DifficultyLevel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Structured evaluation result"""
    model_name: str
    dataset: str
    task_type: str
    accuracy: float
    bleu_score: float
    rouge_l: float
    medical_accuracy: float
    response_time: float
    curriculum_level: str
    domain: str
    error_analysis: Dict[str, Any]
    sample_predictions: List[Dict[str, str]]

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation"""
    models_to_test: List[str]
    datasets: List[str]
    max_samples_per_dataset: int
    output_dir: str
    use_curriculum_learning: bool
    compare_baselines: bool

class MedVLMBenchmarkSuite:
    """Comprehensive MedVLM benchmarking suite"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        self.baseline_models = {}
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize your E2H model
        self.e2h_agent = None
        self.multimodal_agent = None
        
    async def initialize_models(self):
        """Initialize all models for evaluation"""
        logger.info("🚀 Initializing models for benchmarking...")
        
        try:
            # Initialize E2H Medical Agent
            self.e2h_agent = E2HMedicalAgent()
            await self.e2h_agent.initialize()
            
            # Initialize Multimodal Extension
            self.multimodal_agent = MultimodalMedicalAgent()
            await self.multimodal_agent.initialize()
            
            logger.info("✅ E2H models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
            raise
    
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load evaluation dataset"""
        dataset_path = Path(f"medvlm_data/{dataset_name}")
        
        if not dataset_path.exists():
            logger.warning(f"⚠️  Dataset {dataset_name} not found, using mock data")
            return self.create_mock_evaluation_data(dataset_name)
        
        # Load real dataset
        samples = []
        try:
            with open(dataset_path / "test.json", 'r') as f:
                data = json.load(f)
                samples = data.get('samples', [])[:self.config.max_samples_per_dataset]
                
            logger.info(f"📊 Loaded {len(samples)} samples from {dataset_name}")
            return samples
            
        except Exception as e:
            logger.error(f"❌ Error loading {dataset_name}: {e}")
            return self.create_mock_evaluation_data(dataset_name)
    
    def create_mock_evaluation_data(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Create mock evaluation data for testing"""
        mock_samples = {
            'vqa_rad': [
                {
                    'image_path': 'mock_chest_xray.jpg',
                    'question': 'What abnormality is visible in this chest X-ray?',
                    'answer': 'Pneumonia in the right lower lobe',
                    'domain': 'radiology',
                    'difficulty': 'intermediate',
                    'task_type': 'visual_qa'
                },
                {
                    'image_path': 'mock_ct_scan.jpg', 
                    'question': 'Describe the findings in this CT scan.',
                    'answer': 'No acute abnormalities detected',
                    'domain': 'radiology',
                    'difficulty': 'easy',
                    'task_type': 'medical_diagnosis'
                }
            ],
            'pathvqa': [
                {
                    'image_path': 'mock_histology.jpg',
                    'question': 'What type of tissue is shown in this histological image?',
                    'answer': 'Squamous cell carcinoma',
                    'domain': 'pathology',
                    'difficulty': 'hard',
                    'task_type': 'pathology_analysis'
                }
            ],
            'slake': [
                {
                    'image_path': 'mock_mri.jpg',
                    'question': 'What does this MRI scan reveal?',
                    'answer': 'Multiple sclerosis lesions in white matter',
                    'domain': 'neurology',
                    'difficulty': 'hard',
                    'task_type': 'bilingual_vqa'
                }
            ]
        }
        
        return mock_samples.get(dataset_name, [])
    
    async def evaluate_model_on_sample(
        self, 
        model_name: str, 
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a single model on a single sample"""
        
        start_time = datetime.now()
        
        try:
            if model_name == "e2h_medical_agent":
                # Use your E2H agent
                query = f"Image: {sample['image_path']}. Question: {sample['question']}"
                domain = MedicalDomain(sample.get('domain', 'general'))
                
                response = await self.e2h_agent.process_medical_query(
                    query=query,
                    domain=domain,
                    use_multimodal=True
                )
                prediction = response.get('final_answer', '')
                
            elif model_name == "multimodal_medical_agent":
                # Use multimodal extension
                prediction = await self.multimodal_agent.process_multimodal_query(
                    text_query=sample['question'],
                    image_path=sample['image_path']
                )
                
            else:
                # Baseline model (mock for now)
                prediction = await self.evaluate_baseline_model(model_name, sample)
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics
            metrics = self.calculate_metrics(prediction, sample['answer'])
            
            return {
                'prediction': prediction,
                'ground_truth': sample['answer'],
                'response_time': response_time,
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_name}: {e}")
            return {
                'prediction': '',
                'ground_truth': sample['answer'],
                'response_time': 0.0,
                'metrics': {'accuracy': 0.0, 'bleu': 0.0, 'rouge_l': 0.0},
                'success': False,
                'error': str(e)
            }
    
    async def evaluate_baseline_model(self, model_name: str, sample: Dict[str, Any]) -> str:
        """Mock evaluation for baseline models"""
        baseline_responses = {
            'llava_med': "This appears to be a medical imaging study showing possible pathological findings.",
            'med_flamingo': "Based on the visual features, this suggests a medical condition requiring further investigation.",
            'chatcad': "The imaging findings are consistent with the clinical presentation described.",
            'medllavago': "Analysis of the medical image indicates potential diagnostic considerations."
        }
        
        return baseline_responses.get(model_name, "Unable to process medical image.")
    
    def calculate_metrics(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Simplified metrics - in practice, use proper BLEU/ROUGE libraries
        
        # Exact match accuracy
        exact_match = 1.0 if prediction.lower().strip() == ground_truth.lower().strip() else 0.0
        
        # Word overlap (simplified BLEU-like)
        pred_words = set(prediction.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        if len(gt_words) == 0:
            word_overlap = 0.0
        else:
            word_overlap = len(pred_words & gt_words) / len(gt_words)
        
        # Medical keyword accuracy
        medical_keywords = ['diagnosis', 'condition', 'disease', 'symptoms', 'treatment']
        medical_accuracy = self.calculate_medical_accuracy(prediction, ground_truth, medical_keywords)
        
        return {
            'exact_match': exact_match,
            'word_overlap': word_overlap,
            'medical_accuracy': medical_accuracy,
            'bleu_score': word_overlap,  # Simplified
            'rouge_l': word_overlap      # Simplified
        }
    
    def calculate_medical_accuracy(self, pred: str, gt: str, keywords: List[str]) -> float:
        """Calculate medical domain-specific accuracy"""
        pred_lower = pred.lower()
        gt_lower = gt.lower()
        
        # Check for medical term presence
        med_terms_pred = sum(1 for kw in keywords if kw in pred_lower)
        med_terms_gt = sum(1 for kw in keywords if kw in gt_lower)
        
        if med_terms_gt == 0:
            return 1.0 if med_terms_pred == 0 else 0.5
        
        return min(med_terms_pred / med_terms_gt, 1.0)
    
    async def run_comprehensive_benchmark(self) -> List[EvaluationResult]:
        """Run comprehensive benchmark evaluation"""
        logger.info("🏥 Starting MedVLM Comprehensive Benchmark")
        logger.info("=" * 60)
        
        await self.initialize_models()
        
        all_results = []
        
        for dataset_name in self.config.datasets:
            logger.info(f"\n📊 Evaluating on dataset: {dataset_name}")
            
            # Load dataset
            samples = self.load_dataset(dataset_name)
            
            for model_name in self.config.models_to_test:
                logger.info(f"🤖 Testing model: {model_name}")
                
                model_results = []
                
                for i, sample in enumerate(samples):
                    logger.info(f"   Sample {i+1}/{len(samples)}")
                    
                    # Evaluate model on sample
                    result = await self.evaluate_model_on_sample(model_name, sample)
                    model_results.append(result)
                
                # Aggregate results
                aggregated = self.aggregate_results(
                    model_name, 
                    dataset_name, 
                    model_results, 
                    samples
                )
                all_results.append(aggregated)
        
        # Save results
        self.save_results(all_results)
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def aggregate_results(
        self, 
        model_name: str, 
        dataset_name: str, 
        results: List[Dict[str, Any]], 
        samples: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Aggregate results for a model on a dataset"""
        
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return EvaluationResult(
                model_name=model_name,
                dataset=dataset_name,
                task_type='unknown',
                accuracy=0.0,
                bleu_score=0.0,
                rouge_l=0.0,
                medical_accuracy=0.0,
                response_time=0.0,
                curriculum_level='unknown',
                domain='unknown',
                error_analysis={'total_errors': len(results)},
                sample_predictions=[]
            )
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['exact_match', 'word_overlap', 'medical_accuracy', 'bleu_score', 'rouge_l']:
            values = [r['metrics'][metric] for r in successful_results if metric in r['metrics']]
            avg_metrics[metric] = np.mean(values) if values else 0.0
        
        avg_response_time = np.mean([r['response_time'] for r in successful_results])
        
        # Sample predictions for analysis
        sample_predictions = [
            {
                'question': samples[i].get('question', ''),
                'prediction': results[i]['prediction'],
                'ground_truth': results[i]['ground_truth']
            }
            for i in range(min(3, len(results))) if results[i]['success']
        ]
        
        return EvaluationResult(
            model_name=model_name,
            dataset=dataset_name,
            task_type=samples[0].get('task_type', 'unknown') if samples else 'unknown',
            accuracy=avg_metrics.get('exact_match', 0.0),
            bleu_score=avg_metrics.get('bleu_score', 0.0),
            rouge_l=avg_metrics.get('rouge_l', 0.0),
            medical_accuracy=avg_metrics.get('medical_accuracy', 0.0),
            response_time=avg_response_time,
            curriculum_level=samples[0].get('difficulty', 'unknown') if samples else 'unknown',
            domain=samples[0].get('domain', 'unknown') if samples else 'unknown',
            error_analysis={
                'total_samples': len(results),
                'successful': len(successful_results),
                'failed': len(results) - len(successful_results)
            },
            sample_predictions=sample_predictions
        )
    
    def save_results(self, results: List[EvaluationResult]):
        """Save benchmark results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = self.output_dir / f"medvlm_benchmark_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump([asdict(result) for result in results], f, indent=2)
        
        # Save summary CSV
        summary_file = self.output_dir / f"medvlm_benchmark_summary_{timestamp}.csv"
        df = pd.DataFrame([asdict(result) for result in results])
        df.to_csv(summary_file, index=False)
        
        logger.info(f"📁 Results saved to {self.output_dir}")
        
    def generate_comparison_report(self, results: List[EvaluationResult]):
        """Generate comprehensive comparison report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"medvlm_comparison_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# MedVLM Benchmark Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall performance table
            f.write("## Overall Performance Comparison\n\n")
            f.write("| Model | Dataset | Accuracy | BLEU | ROUGE-L | Med Accuracy | Avg Response Time |\n")
            f.write("|-------|---------|----------|------|---------|--------------|-------------------|\n")
            
            for result in results:
                f.write(f"| {result.model_name} | {result.dataset} | "
                       f"{result.accuracy:.3f} | {result.bleu_score:.3f} | "
                       f"{result.rouge_l:.3f} | {result.medical_accuracy:.3f} | "
                       f"{result.response_time:.2f}s |\n")
            
            # Detailed analysis by dataset
            f.write("\n## Detailed Analysis by Dataset\n\n")
            
            datasets = list(set(r.dataset for r in results))
            for dataset in datasets:
                f.write(f"### {dataset.upper()}\n\n")
                dataset_results = [r for r in results if r.dataset == dataset]
                
                # Best performing model
                best = max(dataset_results, key=lambda x: x.medical_accuracy)
                f.write(f"**Best Model:** {best.model_name} (Medical Accuracy: {best.medical_accuracy:.3f})\n\n")
                
                # Sample predictions
                f.write("**Sample Predictions:**\n\n")
                if best.sample_predictions:
                    for i, pred in enumerate(best.sample_predictions[:2]):
                        f.write(f"**Example {i+1}:**\n")
                        f.write(f"- *Question:* {pred['question']}\n")
                        f.write(f"- *Ground Truth:* {pred['ground_truth']}\n")
                        f.write(f"- *Prediction:* {pred['prediction']}\n\n")
            
            # E2H Performance Analysis
            f.write("## E2H Curriculum Learning Analysis\n\n")
            e2h_results = [r for r in results if 'e2h' in r.model_name.lower()]
            if e2h_results:
                f.write("### Curriculum Learning Benefits\n\n")
                f.write("Your E2H agent shows the following performance characteristics:\n\n")
                for result in e2h_results:
                    f.write(f"- **{result.dataset}**: Medical Accuracy {result.medical_accuracy:.3f}, "
                           f"Response Time {result.response_time:.2f}s\n")
                
                f.write("\n### Recommendations for Improvement\n\n")
                f.write("1. **Multimodal Integration**: Enhance vision-language fusion\n")
                f.write("2. **Domain Specialization**: Fine-tune for specific medical domains\n")
                f.write("3. **Curriculum Optimization**: Adjust difficulty progression\n")
                f.write("4. **Real Data Training**: Integrate with actual medical datasets\n")
        
        logger.info(f"📄 Comparison report generated: {report_file}")

async def main():
    """Main evaluation function"""
    config = BenchmarkConfig(
        models_to_test=[
            "e2h_medical_agent",
            "multimodal_medical_agent", 
            "llava_med",
            "med_flamingo"
        ],
        datasets=["vqa_rad", "pathvqa", "slake"],
        max_samples_per_dataset=5,  # Small for demo
        output_dir="./benchmark_results",
        use_curriculum_learning=True,
        compare_baselines=True
    )
    
    benchmark_suite = MedVLMBenchmarkSuite(config)
    results = await benchmark_suite.run_comprehensive_benchmark()
    
    print("\n🏆 MEDVLM BENCHMARK COMPLETE!")
    print("=" * 50)
    print(f"📊 Evaluated {len(results)} model-dataset combinations")
    print(f"📁 Results saved to: {benchmark_suite.output_dir}")
    print("\n🎯 Key Findings:")
    
    # Show top performer
    if results:
        best_result = max(results, key=lambda x: x.medical_accuracy)
        print(f"   🥇 Best Medical Accuracy: {best_result.model_name} on {best_result.dataset}")
        print(f"      Score: {best_result.medical_accuracy:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
