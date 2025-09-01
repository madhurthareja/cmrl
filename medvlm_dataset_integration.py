"""
Real MedVLM Dataset Integration
Download and prepare real medical vision-language datasets for benchmarking
"""

import os
import json
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MedVLMDatasetDownloader:
    """Download and prepare real MedVLM datasets"""
    
    def __init__(self, data_dir: str = "./medvlm_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Real dataset URLs and info
        self.datasets_info = {
            "vqa_rad": {
                "name": "VQA-RAD",
                "description": "Radiology Visual Question Answering Dataset",
                "url": "https://osf.io/89kps/download",  # OSF repository
                "paper": "https://arxiv.org/abs/1811.02629",
                "size": "315 images, 3,515 QA pairs",
                "tasks": ["visual_qa", "medical_diagnosis"],
                "domains": ["radiology"],
                "license": "CC BY 4.0"
            },
            "pathvqa": {
                "name": "PathVQA", 
                "description": "Pathology Visual Question Answering Dataset",
                "url": "https://github.com/UCSD-AI4H/PathVQA",
                "paper": "https://arxiv.org/abs/2003.10286",
                "size": "32,799 images, 234,775 QA pairs", 
                "tasks": ["pathology_analysis", "visual_qa"],
                "domains": ["pathology"],
                "license": "MIT"
            },
            "slake": {
                "name": "SLAKE",
                "description": "Bilingual Medical VQA Dataset",
                "url": "https://www.med-vqa.com/slake/",
                "paper": "https://arxiv.org/abs/2102.09542",
                "size": "642 images, 14,028 QA pairs",
                "tasks": ["bilingual_vqa", "medical_reasoning"],
                "domains": ["radiology", "general"],
                "license": "CC BY-NC 4.0"
            },
            "mimic_cxr": {
                "name": "MIMIC-CXR",
                "description": "Chest X-ray Database with Reports", 
                "url": "https://physionet.org/content/mimic-cxr/2.0.0/",
                "paper": "https://arxiv.org/abs/1901.07042",
                "size": "377,110 images, 227,835 reports",
                "tasks": ["report_generation", "image_captioning"],
                "domains": ["radiology", "pulmonology"],
                "license": "PhysioNet Credentialed Health Data License"
            },
            "pmc_vqa": {
                "name": "PMC-VQA",
                "description": "PubMed Central Medical VQA",
                "url": "https://github.com/xiaoman-zhang/PMC-VQA",
                "paper": "https://arxiv.org/abs/2305.10415",
                "size": "227,000+ image-text pairs",
                "tasks": ["medical_vqa", "multimodal_understanding"],
                "domains": ["general", "radiology", "pathology"],
                "license": "Apache 2.0"
            }
        }
    
    def list_available_datasets(self):
        """List all available datasets with descriptions"""
        print("🏥 Available MedVLM Datasets for Benchmarking")
        print("=" * 60)
        
        for dataset_id, info in self.datasets_info.items():
            print(f"\n📊 {info['name']} ({dataset_id})")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Tasks: {', '.join(info['tasks'])}")
            print(f"   Domains: {', '.join(info['domains'])}")
            print(f"   Paper: {info['paper']}")
            print(f"   License: {info['license']}")
    
    def create_download_instructions(self, dataset_id: str):
        """Create detailed download instructions for a dataset"""
        if dataset_id not in self.datasets_info:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        
        info = self.datasets_info[dataset_id]
        instructions_file = self.data_dir / f"{dataset_id}_download_instructions.md"
        
        with open(instructions_file, 'w') as f:
            f.write(f"# {info['name']} Download Instructions\n\n")
            f.write(f"**Description:** {info['description']}\n\n")
            f.write(f"**Size:** {info['size']}\n\n")
            f.write(f"**License:** {info['license']}\n\n")
            
            f.write("## Download Steps\n\n")
            
            if dataset_id == "vqa_rad":
                f.write("1. Visit: https://osf.io/89kps/\n")
                f.write("2. Download VQA_RAD_Dataset.zip\n")
                f.write("3. Extract to ./medvlm_data/vqa_rad/\n")
                f.write("4. Structure should be:\n")
                f.write("   ```\n")
                f.write("   vqa_rad/\n")
                f.write("   ├── images/\n")
                f.write("   ├── trainset.json\n")
                f.write("   └── testset.json\n")
                f.write("   ```\n\n")
                
            elif dataset_id == "pathvqa":
                f.write("1. Visit: https://github.com/UCSD-AI4H/PathVQA\n")
                f.write("2. Follow repository instructions\n")
                f.write("3. Download images and QA pairs\n")
                f.write("4. Extract to ./medvlm_data/pathvqa/\n\n")
                
            elif dataset_id == "slake":
                f.write("1. Visit: https://www.med-vqa.com/slake/\n")
                f.write("2. Register and download dataset\n")
                f.write("3. Extract to ./medvlm_data/slake/\n\n")
                
            elif dataset_id == "mimic_cxr":
                f.write("1. Visit: https://physionet.org/content/mimic-cxr/2.0.0/\n")
                f.write("2. Complete credentialing process\n")
                f.write("3. Download dataset (Large: ~5TB)\n")
                f.write("4. Extract to ./medvlm_data/mimic_cxr/\n\n")
                
            elif dataset_id == "pmc_vqa":
                f.write("1. Visit: https://github.com/xiaoman-zhang/PMC-VQA\n")
                f.write("2. Clone repository\n")
                f.write("3. Follow data preparation instructions\n")
                f.write("4. Extract to ./medvlm_data/pmc_vqa/\n\n")
            
            f.write("## Citation\n\n")
            f.write(f"Paper: {info['paper']}\n\n")
            f.write("Please cite the original paper if you use this dataset.\n\n")
            
            f.write("## Integration with Benchmark\n\n")
            f.write("Once downloaded, the dataset will be automatically detected by:\n")
            f.write("```python\n")
            f.write("from medvlm_evaluation_suite import MedVLMBenchmarkSuite\n")
            f.write(f"# Dataset '{dataset_id}' will be available for benchmarking\n")
            f.write("```\n")
        
        print(f"📄 Download instructions saved: {instructions_file}")
        return instructions_file
    
    def create_mock_dataset_structure(self, dataset_id: str):
        """Create mock dataset structure for testing"""
        dataset_dir = self.data_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock test.json
        mock_data = {
            "dataset_info": self.datasets_info[dataset_id],
            "samples": [
                {
                    "id": f"{dataset_id}_sample_001",
                    "image_path": f"images/sample_001.jpg",
                    "question": "What abnormality is visible in this medical image?",
                    "answer": "No acute abnormalities detected",
                    "domain": self.datasets_info[dataset_id]["domains"][0],
                    "difficulty": "easy",
                    "task_type": self.datasets_info[dataset_id]["tasks"][0]
                },
                {
                    "id": f"{dataset_id}_sample_002", 
                    "image_path": f"images/sample_002.jpg",
                    "question": "Describe the pathological findings in this image.",
                    "answer": "Findings consistent with inflammatory changes",
                    "domain": self.datasets_info[dataset_id]["domains"][0],
                    "difficulty": "intermediate", 
                    "task_type": self.datasets_info[dataset_id]["tasks"][0]
                }
            ]
        }
        
        with open(dataset_dir / "test.json", 'w') as f:
            json.dump(mock_data, f, indent=2)
        
        # Create images directory
        images_dir = dataset_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        print(f"📁 Mock dataset structure created: {dataset_dir}")
    
    def validate_dataset(self, dataset_id: str) -> bool:
        """Validate that a dataset is properly downloaded and structured"""
        dataset_dir = self.data_dir / dataset_id
        
        if not dataset_dir.exists():
            return False
        
        # Check for required files
        required_files = ["test.json"]
        for file in required_files:
            if not (dataset_dir / file).exists():
                return False
        
        # Validate JSON structure
        try:
            with open(dataset_dir / "test.json", 'r') as f:
                data = json.load(f)
                if "samples" not in data or not isinstance(data["samples"], list):
                    return False
        except (json.JSONDecodeError, IOError):
            return False
        
        return True
    
    def get_benchmark_ready_datasets(self) -> List[str]:
        """Get list of datasets ready for benchmarking"""
        ready_datasets = []
        
        for dataset_id in self.datasets_info.keys():
            if self.validate_dataset(dataset_id):
                ready_datasets.append(dataset_id)
        
        return ready_datasets
    
    def prepare_all_datasets(self, create_mock: bool = True):
        """Prepare all datasets for benchmarking"""
        print("🔧 Preparing MedVLM Datasets for Benchmarking")
        print("=" * 50)
        
        for dataset_id in self.datasets_info.keys():
            print(f"\n📊 Processing {dataset_id}...")
            
            # Create download instructions
            self.create_download_instructions(dataset_id)
            
            # Create mock structure if requested
            if create_mock:
                self.create_mock_dataset_structure(dataset_id)
            
            print(f"✅ {dataset_id} prepared")
        
        ready_datasets = self.get_benchmark_ready_datasets()
        print(f"\n🎯 {len(ready_datasets)} datasets ready for benchmarking:")
        for dataset in ready_datasets:
            print(f"   ✓ {dataset}")
        
        print(f"\n📁 All files saved to: {self.data_dir}")

def create_baseline_comparison_config():
    """Create configuration for comparing against baseline models"""
    config = {
        "baseline_models": {
            "llava_med": {
                "name": "LLaVA-Med",
                "description": "Medical adaptation of LLaVA",
                "paper": "https://arxiv.org/abs/2306.00890", 
                "github": "https://github.com/microsoft/LLaVA-Med",
                "model_type": "vision_language_model",
                "medical_specialization": True
            },
            "med_flamingo": {
                "name": "Med-Flamingo",
                "description": "Medical few-shot vision-language model",
                "paper": "https://arxiv.org/abs/2307.15189",
                "github": "https://github.com/snap-stanford/med-flamingo",
                "model_type": "few_shot_learner",
                "medical_specialization": True
            },
            "chatcad": {
                "name": "ChatCAD", 
                "description": "Conversational Computer-Aided Diagnosis",
                "paper": "https://arxiv.org/abs/2302.07257",
                "model_type": "conversational_diagnosis",
                "medical_specialization": True
            },
            "medllavago": {
                "name": "Med-LLaVA",
                "description": "Medical Large Language and Vision Assistant",
                "model_type": "medical_assistant",
                "medical_specialization": True
            }
        },
        "evaluation_metrics": {
            "accuracy": "Exact match accuracy", 
            "bleu_score": "BLEU score for text generation",
            "rouge_l": "ROUGE-L score for text similarity",
            "medical_accuracy": "Domain-specific medical accuracy",
            "response_time": "Average response time in seconds"
        },
        "benchmark_tasks": {
            "visual_qa": "Visual Question Answering on medical images",
            "medical_diagnosis": "Medical diagnosis from visual input",
            "pathology_analysis": "Pathological finding analysis", 
            "report_generation": "Medical report generation",
            "image_captioning": "Medical image captioning"
        }
    }
    
    config_file = Path("./medvlm_data/baseline_comparison_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📋 Baseline comparison config saved: {config_file}")
    return config

def main():
    """Main setup function"""
    downloader = MedVLMDatasetDownloader()
    
    # List available datasets
    downloader.list_available_datasets()
    
    print("\n" + "="*60)
    
    # Prepare all datasets
    downloader.prepare_all_datasets(create_mock=True)
    
    print("\n" + "="*60)
    
    # Create baseline comparison config
    create_baseline_comparison_config()
    
    print("\n🚀 MedVLM Dataset Integration Complete!")
    print("\n📋 Next Steps:")
    print("1. Download real datasets using the generated instructions")
    print("2. Run benchmarks with: python medvlm_evaluation_suite.py")
    print("3. Compare your E2H agent against state-of-the-art MedVLM models")
    print("4. Analyze results and improve your curriculum learning approach")

if __name__ == "__main__":
    main()
