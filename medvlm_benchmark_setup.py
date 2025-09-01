"""
MedVLM Benchmarking Configuration and Dataset Setup
Provides utilities to download and prepare medical VQA datasets for evaluation
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MedVLMDatasetManager:
    """Manages medical VQA datasets for benchmarking"""
    
    def __init__(self, data_root: str = "./medvlm_data"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True, parents=True)
        
        self.dataset_configs = {
            "vqa_rad": {
                "name": "VQA-RAD",
                "description": "Radiology Visual Question Answering Dataset",
                "size": "315 images, 3,515 QA pairs",
                "download_url": "https://osf.io/89kps/",  # Official VQA-RAD
                "paper": "https://arxiv.org/abs/1811.02883",
                "tasks": ["visual_qa", "medical_diagnosis"]
            },
            "pathvqa": {
                "name": "PathVQA", 
                "description": "Pathology Visual Question Answering Dataset",
                "size": "32,799 images, 234,775 QA pairs",
                "download_url": "https://github.com/UCSD-AI4H/PathVQA",
                "paper": "https://arxiv.org/abs/2003.10286",
                "tasks": ["pathology_analysis", "visual_qa"]
            },
            "slake": {
                "name": "SLAKE",
                "description": "Bilingual Medical VQA Dataset", 
                "size": "642 images, 14,028 QA pairs",
                "download_url": "https://www.med-vqa.com/slake/",
                "paper": "https://arxiv.org/abs/2102.09542",
                "tasks": ["bilingual_vqa", "medical_reasoning"]
            },
            "mimic_cxr": {
                "name": "MIMIC-CXR",
                "description": "Chest X-ray Database with Reports",
                "size": "377,110 images, 227,835 reports", 
                "download_url": "https://physionet.org/content/mimic-cxr/2.0.0/",
                "paper": "https://arxiv.org/abs/1901.07042",
                "tasks": ["report_generation", "image_captioning"]
            },
            "pmc_vqa": {
                "name": "PMC-VQA",
                "description": "PubMed Central Medical VQA",
                "size": "227,000+ image-text pairs",
                "download_url": "https://github.com/xiaoman-zhang/PMC-VQA",
                "paper": "https://arxiv.org/abs/2305.10415", 
                "tasks": ["medical_vqa", "multimodal_understanding"]
            }
        }
    
    def list_available_datasets(self) -> Dict:
        """List all available medical VQA datasets"""
        return self.dataset_configs
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get detailed information about a specific dataset"""
        if dataset_name not in self.dataset_configs:
            available = ", ".join(self.dataset_configs.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
        
        return self.dataset_configs[dataset_name]
    
    def prepare_dataset_structure(self, dataset_name: str) -> Path:
        """Create directory structure for a dataset"""
        dataset_path = self.data_root / dataset_name
        
        # Create standard structure
        (dataset_path / "images").mkdir(exist_ok=True, parents=True)
        (dataset_path / "annotations").mkdir(exist_ok=True, parents=True)
        (dataset_path / "splits").mkdir(exist_ok=True, parents=True)
        
        return dataset_path
    
    def create_mock_vqa_rad_data(self) -> Dict:
        """Create mock VQA-RAD samples for testing"""
        dataset_path = self.prepare_dataset_structure("vqa_rad")
        
        mock_samples = [
            {
                "id": "vqa_rad_001",
                "image": "synpic100.jpg",
                "question": "What is the primary abnormality in this chest X-ray?",
                "answer": "Pneumothorax on the right side",
                "question_type": "abnormality",
                "image_type": "chest_xray",
                "domain": "radiology",
                "difficulty": "medium"
            },
            {
                "id": "vqa_rad_002", 
                "image": "synpic200.jpg",
                "question": "Is there evidence of cardiomegaly?",
                "answer": "Yes, the cardiac silhouette is enlarged suggesting cardiomegaly",
                "question_type": "presence",
                "image_type": "chest_xray", 
                "domain": "cardiology",
                "difficulty": "easy"
            },
            {
                "id": "vqa_rad_003",
                "image": "synpic300.jpg",
                "question": "What is the most likely diagnosis based on this brain CT?",
                "answer": "Acute ischemic stroke in the middle cerebral artery territory",
                "question_type": "diagnosis",
                "image_type": "ct_brain",
                "domain": "neurology", 
                "difficulty": "hard"
            }
        ]
        
        # Save mock annotations
        annotations_file = dataset_path / "annotations" / "vqa_rad_mock.json"
        with open(annotations_file, 'w') as f:
            json.dump(mock_samples, f, indent=2)
        
        logger.info(f"Created mock VQA-RAD data at {annotations_file}")
        return {"samples": mock_samples, "path": dataset_path}
    
    def create_mock_pathvqa_data(self) -> Dict:
        """Create mock PathVQA samples for testing"""
        dataset_path = self.prepare_dataset_structure("pathvqa")
        
        mock_samples = [
            {
                "id": "pathvqa_001",
                "image": "path_adenocarcinoma_001.jpg",
                "question": "What type of tissue architecture is shown in this histopathology image?",
                "answer": "Adenocarcinoma with glandular formation and nuclear pleomorphism",
                "question_type": "architecture",
                "image_type": "histopathology",
                "domain": "pathology",
                "difficulty": "hard"
            },
            {
                "id": "pathvqa_002",
                "image": "path_inflammation_001.jpg", 
                "question": "Are inflammatory cells present in this tissue?",
                "answer": "Yes, there are abundant inflammatory cells including lymphocytes and neutrophils",
                "question_type": "presence",
                "image_type": "histopathology",
                "domain": "pathology",
                "difficulty": "medium"
            }
        ]
        
        annotations_file = dataset_path / "annotations" / "pathvqa_mock.json"
        with open(annotations_file, 'w') as f:
            json.dump(mock_samples, f, indent=2)
        
        logger.info(f"Created mock PathVQA data at {annotations_file}")
        return {"samples": mock_samples, "path": dataset_path}
    
    def create_evaluation_splits(self, dataset_name: str, train_ratio: float = 0.7) -> Dict:
        """Create train/val/test splits for evaluation"""
        dataset_path = self.data_root / dataset_name
        annotations_file = dataset_path / "annotations" / f"{dataset_name}_mock.json"
        
        if not annotations_file.exists():
            logger.error(f"Annotations file not found: {annotations_file}")
            return {}
        
        with open(annotations_file, 'r') as f:
            samples = json.load(f)
        
        # Simple split
        total = len(samples)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + 0.15))
        
        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end], 
            "test": samples[val_end:]
        }
        
        # Save splits
        splits_dir = dataset_path / "splits"
        for split_name, split_data in splits.items():
            split_file = splits_dir / f"{split_name}.json"
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
        
        logger.info(f"Created evaluation splits for {dataset_name}")
        return splits

class MedVLMBenchmarkConfig:
    """Configuration for MedVLM benchmarking experiments"""
    
    def __init__(self):
        self.benchmark_config = {
            "evaluation_metrics": {
                "accuracy": "Exact match accuracy",
                "bleu": "BLEU score for generated text",
                "rouge_l": "ROUGE-L for long text similarity",
                "cider": "CIDEr for image captioning",
                "medical_accuracy": "Medical concept accuracy"
            },
            "curriculum_levels": [
                "trivial", "easy", "medium", "hard"
            ],
            "medical_domains": [
                "radiology", "pathology", "cardiology", 
                "neurology", "dermatology", "ophthalmology"
            ],
            "image_types": [
                "chest_xray", "ct_scan", "mri", "ultrasound",
                "histopathology", "fundus", "dermatology"
            ]
        }
    
    def create_benchmark_experiment(
        self, 
        experiment_name: str,
        datasets: List[str],
        models: List[str],
        curriculum_levels: List[str]
    ) -> Dict:
        """Create a benchmarking experiment configuration"""
        
        experiment = {
            "name": experiment_name,
            "description": f"MedVLM benchmark comparing models on medical VQA tasks",
            "datasets": datasets,
            "models": models,
            "curriculum_levels": curriculum_levels,
            "evaluation_metrics": list(self.benchmark_config["evaluation_metrics"].keys()),
            "output_dir": f"./experiments/{experiment_name}",
            "created_date": "2025-09-01"
        }
        
        # Create experiment directory
        exp_dir = Path(experiment["output_dir"])
        exp_dir.mkdir(exist_ok=True, parents=True)
        
        # Save experiment config
        config_file = exp_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        logger.info(f"Created benchmark experiment: {experiment_name}")
        return experiment
    
    def get_baseline_models(self) -> Dict:
        """Get information about baseline MedVLM models for comparison"""
        return {
            "llava_med": {
                "name": "LLaVA-Med",
                "description": "Medical adaptation of LLaVA",
                "paper": "https://arxiv.org/abs/2306.00890",
                "implementation": "https://github.com/microsoft/LLaVA-Med"
            },
            "med_flamingo": {
                "name": "Med-Flamingo", 
                "description": "Medical few-shot vision-language model",
                "paper": "https://arxiv.org/abs/2307.15189",
                "implementation": "https://github.com/snap-stanford/med-flamingo"
            },
            "chatcad": {
                "name": "ChatCAD",
                "description": "Conversational Computer-Aided Diagnosis",
                "paper": "https://arxiv.org/abs/2302.07257", 
                "implementation": "https://github.com/zhaozh10/ChatCAD"
            },
            "medsam": {
                "name": "MedSAM",
                "description": "Medical Segment Anything Model",
                "paper": "https://arxiv.org/abs/2304.12306",
                "implementation": "https://github.com/bowang-lab/MedSAM"
            }
        }

def setup_medvlm_benchmarking():
    """Set up complete MedVLM benchmarking environment"""
    
    print("🏥 Setting up MedVLM Benchmarking Environment")
    print("=" * 50)
    
    # Initialize dataset manager
    dataset_manager = MedVLMDatasetManager()
    
    # List available datasets
    print("\n📊 Available Medical VQA Datasets:")
    for name, info in dataset_manager.list_available_datasets().items():
        print(f"  • {info['name']}: {info['description']}")
        print(f"    Size: {info['size']}")
        print(f"    Tasks: {', '.join(info['tasks'])}")
        print()
    
    # Create mock datasets for testing
    print("🔧 Creating Mock Datasets for Testing...")
    vqa_rad_data = dataset_manager.create_mock_vqa_rad_data()
    pathvqa_data = dataset_manager.create_mock_pathvqa_data()
    
    # Create evaluation splits
    print("✂️ Creating Evaluation Splits...")
    dataset_manager.create_evaluation_splits("vqa_rad")
    dataset_manager.create_evaluation_splits("pathvqa")
    
    # Set up benchmark configuration
    benchmark_config = MedVLMBenchmarkConfig()
    
    # Create sample experiment
    experiment = benchmark_config.create_benchmark_experiment(
        experiment_name="e2h_vs_baselines_2025",
        datasets=["vqa_rad", "pathvqa", "slake"],
        models=["e2h_medical_agent", "llava_med", "med_flamingo"],
        curriculum_levels=["easy", "medium", "hard"]
    )
    
    print("🎯 Created Benchmark Experiment:")
    print(f"  Name: {experiment['name']}")
    print(f"  Datasets: {', '.join(experiment['datasets'])}")
    print(f"  Models: {', '.join(experiment['models'])}")
    print(f"  Output: {experiment['output_dir']}")
    
    # Show baseline models for comparison
    print("\n🤖 Baseline Models for Comparison:")
    baselines = benchmark_config.get_baseline_models()
    for name, info in baselines.items():
        print(f"  • {info['name']}: {info['description']}")
    
    print("\n✅ MedVLM benchmarking environment setup complete!")
    print("Next steps:")
    print("1. Download real datasets from provided URLs")
    print("2. Implement vision model integration") 
    print("3. Run benchmark evaluation using medvlm_extension.py")
    print("4. Compare results against baseline models")

if __name__ == "__main__":
    setup_medvlm_benchmarking()
