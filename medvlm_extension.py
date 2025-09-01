"""
Medical Vision-Language Model Extension for E2H Medical Agent
Extends the current text-based medical agent to handle multimodal medical inputs
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image
import torch
from transformers import (
    LlavaForConditionalGeneration, 
    LlavaProcessor,
    AutoTokenizer,
    AutoModel
)
import base64
from io import BytesIO

from medical_agent_core import MedicalDomain, DifficultyLevel
from e2h_medical_agent import E2HMedicalAgent

logger = logging.getLogger(__name__)

class MedicalImageType(Enum):
    """Types of medical images"""
    XRAY = "x-ray"
    CT_SCAN = "ct-scan"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    PATHOLOGY = "pathology"
    ENDOSCOPY = "endoscopy"
    DERMATOLOGY = "dermatology"
    FUNDUS = "fundus"

@dataclass
class MedicalImageInput:
    """Medical image input with metadata"""
    image: Image.Image
    image_type: MedicalImageType
    patient_context: Optional[str] = None
    clinical_history: Optional[str] = None
    image_metadata: Optional[Dict] = None

@dataclass
class MultimodalMedicalQuery:
    """Combined text and image medical query"""
    text_query: str
    images: List[MedicalImageInput]
    domain: MedicalDomain
    urgency_level: str = "routine"  # routine, urgent, emergent

class MedVLMBenchmarkDataset:
    """Interface for medical VQA benchmark datasets"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.samples = []
        
    def load_vqa_rad(self):
        """Load VQA-RAD dataset samples"""
        # Mock implementation - replace with actual dataset loading
        mock_samples = [
            {
                "image_path": "data/vqa_rad/synpic100.jpg",
                "question": "What is the primary finding in this chest X-ray?",
                "answer": "Pneumothorax on the right side",
                "image_type": MedicalImageType.XRAY,
                "domain": MedicalDomain.RADIOLOGY
            },
            {
                "image_path": "data/vqa_rad/synpic200.jpg", 
                "question": "Is there evidence of cardiomegaly?",
                "answer": "Yes, the cardiac silhouette is enlarged",
                "image_type": MedicalImageType.XRAY,
                "domain": MedicalDomain.CARDIOLOGY
            }
        ]
        return mock_samples
    
    def load_pathvqa(self):
        """Load PathVQA dataset samples"""
        mock_samples = [
            {
                "image_path": "data/pathvqa/path_001.jpg",
                "question": "What type of tissue architecture is shown?",
                "answer": "Adenocarcinoma with glandular formation",
                "image_type": MedicalImageType.PATHOLOGY,
                "domain": MedicalDomain.PATHOLOGY
            }
        ]
        return mock_samples
    
    def load_slake(self):
        """Load SLAKE bilingual medical VQA dataset"""
        mock_samples = [
            {
                "image_path": "data/slake/Brain_001.jpg",
                "question": "What abnormality is visible in this brain MRI?",
                "answer": "There is a hyperintense lesion in the white matter suggesting demyelination",
                "image_type": MedicalImageType.MRI,
                "domain": MedicalDomain.NEUROLOGY
            }
        ]
        return mock_samples

class MultimodalMedicalAgent(E2HMedicalAgent):
    """Extended medical agent with vision-language capabilities"""
    
    def __init__(self, model_name: str = "ollama", vision_model: str = "llava"):
        super().__init__(model_name)
        self.vision_model = vision_model
        self.image_processor = None
        self.vision_encoder = None
        self.initialize_vision_components()
    
    def initialize_vision_components(self):
        """Initialize vision processing components"""
        try:
            if self.vision_model == "llava":
                # Initialize LLaVA for medical image understanding
                model_path = "liuhaotian/llava-v1.6-mistral-7b"  # or llava-med if available
                self.vision_processor = LlavaProcessor.from_pretrained(model_path)
                self.vision_model_obj = LlavaForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=torch.float16, device_map="auto"
                )
                logger.info("Initialized LLaVA vision model")
            
            elif self.vision_model == "med_flamingo":
                # Initialize Med-Flamingo (would need actual model weights)
                logger.info("Med-Flamingo initialization would go here")
                
        except Exception as e:
            logger.warning(f"Could not initialize vision model: {e}")
            logger.info("Running in text-only mode")
    
    async def process_multimodal_query(
        self, 
        query: MultimodalMedicalQuery, 
        curriculum_level: DifficultyLevel = DifficultyLevel.EASY
    ) -> Dict:
        """Process combined text and image medical query"""
        
        logger.info(f"Processing multimodal query: {query.text_query[:50]}... with {len(query.images)} images")
        
        # Phase 1: Image Analysis
        image_analyses = []
        for img_input in query.images:
            image_analysis = await self.analyze_medical_image(
                img_input, query.text_query, curriculum_level
            )
            image_analyses.append(image_analysis)
        
        # Phase 2: Integrate with text-based medical reasoning
        combined_context = self.integrate_multimodal_context(
            query.text_query, image_analyses, query.domain
        )
        
        # Phase 3: Apply E2H curriculum learning
        curriculum_response = await self.apply_curriculum_to_multimodal(
            combined_context, curriculum_level, query.domain
        )
        
        # Phase 4: Generate comprehensive medical response
        final_response = await self.generate_multimodal_medical_response(
            curriculum_response, query.images, query.domain
        )
        
        return {
            "text_response": final_response,
            "image_analyses": image_analyses,
            "curriculum_level": curriculum_level.value,
            "confidence_score": curriculum_response.get("confidence", 0.0),
            "domain": query.domain.value,
            "multimodal_reasoning": True
        }
    
    async def analyze_medical_image(
        self, 
        img_input: MedicalImageInput, 
        text_query: str,
        curriculum_level: DifficultyLevel
    ) -> Dict:
        """Analyze a single medical image with curriculum awareness"""
        
        try:
            # Prepare image and text inputs
            if self.vision_processor and self.vision_model_obj:
                # Use LLaVA/vision model for image analysis
                prompt = self.create_medical_image_prompt(img_input, text_query, curriculum_level)
                
                inputs = self.vision_processor(
                    text=prompt,
                    images=img_input.image,
                    return_tensors="pt"
                )
                
                # Generate image analysis
                with torch.no_grad():
                    outputs = self.vision_model_obj.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7
                    )
                
                response = self.vision_processor.decode(outputs[0], skip_special_tokens=True)
                
                return {
                    "image_type": img_input.image_type.value,
                    "analysis": response,
                    "confidence": 0.85,  # Would calculate actual confidence
                    "findings": self.extract_medical_findings(response),
                    "curriculum_adapted": True
                }
            else:
                # Fallback: Mock image analysis
                return {
                    "image_type": img_input.image_type.value,
                    "analysis": f"Mock analysis of {img_input.image_type.value} image for curriculum level {curriculum_level.value}",
                    "confidence": 0.5,
                    "findings": ["Mock finding 1", "Mock finding 2"],
                    "curriculum_adapted": True
                }
                
        except Exception as e:
            logger.error(f"Error analyzing medical image: {e}")
            return {
                "image_type": img_input.image_type.value,
                "analysis": "Error processing image",
                "confidence": 0.0,
                "findings": [],
                "error": str(e)
            }
    
    def create_medical_image_prompt(
        self, 
        img_input: MedicalImageInput, 
        text_query: str, 
        curriculum_level: DifficultyLevel
    ) -> str:
        """Create curriculum-aware prompt for medical image analysis"""
        
        base_prompt = f"""You are analyzing a medical {img_input.image_type.value} image. 
Patient query: {text_query}

"""
        
        # Adapt prompt based on curriculum level
        if curriculum_level == DifficultyLevel.TRIVIAL:
            base_prompt += "Provide a basic, simple description of what you see in the image."
        elif curriculum_level == DifficultyLevel.EASY:
            base_prompt += "Describe the key medical findings visible in the image."
        elif curriculum_level == DifficultyLevel.MEDIUM:
            base_prompt += "Provide a detailed medical analysis including normal and abnormal findings."
        elif curriculum_level == DifficultyLevel.HARD:
            base_prompt += "Provide comprehensive medical interpretation with differential diagnoses and clinical correlations."
        
        # Add clinical context if available
        if img_input.clinical_history:
            base_prompt += f"\n\nClinical History: {img_input.clinical_history}"
        
        return base_prompt
    
    def integrate_multimodal_context(
        self, 
        text_query: str, 
        image_analyses: List[Dict], 
        domain: MedicalDomain
    ) -> str:
        """Integrate text query with image analysis results"""
        
        context = f"Patient Query: {text_query}\n\nMedical Domain: {domain.value}\n\n"
        context += "Image Analysis Results:\n"
        
        for i, analysis in enumerate(image_analyses, 1):
            context += f"\nImage {i} ({analysis['image_type']}):\n"
            context += f"- Analysis: {analysis['analysis']}\n"
            context += f"- Confidence: {analysis['confidence']:.2f}\n"
            if analysis.get('findings'):
                context += f"- Key Findings: {', '.join(analysis['findings'])}\n"
        
        return context
    
    async def apply_curriculum_to_multimodal(
        self, 
        combined_context: str, 
        curriculum_level: DifficultyLevel, 
        domain: MedicalDomain
    ) -> Dict:
        """Apply E2H curriculum learning to multimodal medical reasoning"""
        
        # Use the existing E2H medical agent processing with enhanced context
        response = await super().process_medical_query(
            combined_context, domain, curriculum_level
        )
        
        # Add multimodal-specific enhancements
        response["multimodal_processing"] = True
        response["vision_integration"] = "active"
        
        return response
    
    async def generate_multimodal_medical_response(
        self, 
        curriculum_response: Dict, 
        images: List[MedicalImageInput], 
        domain: MedicalDomain
    ) -> str:
        """Generate final medical response incorporating both text and image analysis"""
        
        base_response = curriculum_response.get("final_answer", "")
        
        # Enhance with multimodal reasoning
        multimodal_summary = "\n\n## Multimodal Analysis Summary:\n"
        multimodal_summary += f"- Processed {len(images)} medical images\n"
        multimodal_summary += f"- Medical domain: {domain.value}\n"
        multimodal_summary += f"- Curriculum level: {curriculum_response.get('curriculum_level', 'N/A')}\n"
        multimodal_summary += f"- Confidence: {curriculum_response.get('confidence', 0.0):.2f}\n"
        
        return base_response + multimodal_summary
    
    def extract_medical_findings(self, analysis_text: str) -> List[str]:
        """Extract key medical findings from image analysis"""
        # Simple keyword-based extraction (would use NLP in practice)
        findings = []
        medical_keywords = [
            "pneumothorax", "cardiomegaly", "consolidation", "effusion",
            "mass", "nodule", "fracture", "dislocation", "stenosis",
            "hemorrhage", "infarct", "edema", "inflammation"
        ]
        
        for keyword in medical_keywords:
            if keyword.lower() in analysis_text.lower():
                findings.append(keyword)
        
        return findings[:5]  # Limit to top 5 findings

class MedVLMBenchmarkEvaluator:
    """Evaluator for benchmarking against MedVLM datasets"""
    
    def __init__(self, agent: MultimodalMedicalAgent):
        self.agent = agent
        self.evaluation_metrics = {}
    
    async def evaluate_on_vqa_rad(self, num_samples: int = 50) -> Dict:
        """Evaluate on VQA-RAD dataset"""
        dataset = MedVLMBenchmarkDataset("VQA-RAD")
        samples = dataset.load_vqa_rad()[:num_samples]
        
        correct_answers = 0
        total_samples = len(samples)
        
        for sample in samples:
            # Mock evaluation - would load actual images
            query = MultimodalMedicalQuery(
                text_query=sample["question"],
                images=[],  # Would load actual image
                domain=sample["domain"]
            )
            
            response = await self.agent.process_multimodal_query(query)
            
            # Simple text matching for evaluation (would use BLEU/ROUGE in practice)
            predicted = response["text_response"].lower()
            ground_truth = sample["answer"].lower()
            
            if any(word in predicted for word in ground_truth.split()[:3]):
                correct_answers += 1
        
        accuracy = correct_answers / total_samples if total_samples > 0 else 0.0
        
        return {
            "dataset": "VQA-RAD",
            "accuracy": accuracy,
            "total_samples": total_samples,
            "correct_answers": correct_answers
        }
    
    async def evaluate_on_pathvqa(self, num_samples: int = 50) -> Dict:
        """Evaluate on PathVQA dataset"""
        # Similar implementation to VQA-RAD
        return {
            "dataset": "PathVQA",
            "accuracy": 0.0,  # Placeholder
            "total_samples": num_samples,
            "correct_answers": 0
        }
    
    async def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive evaluation across multiple datasets"""
        results = {}
        
        # Evaluate on multiple datasets
        results["vqa_rad"] = await self.evaluate_on_vqa_rad()
        results["pathvqa"] = await self.evaluate_on_pathvqa()
        
        # Calculate overall metrics
        total_accuracy = sum(r["accuracy"] for r in results.values()) / len(results)
        
        return {
            "individual_results": results,
            "overall_accuracy": total_accuracy,
            "benchmark_date": "2025-09-01",
            "model_type": "E2H Medical Agent with Vision"
        }

# Example usage
async def demo_medvlm_benchmarking():
    """Demonstrate MedVLM benchmarking capabilities"""
    
    # Initialize multimodal medical agent
    agent = MultimodalMedicalAgent(model_name="ollama", vision_model="llava")
    
    # Create sample multimodal query
    sample_image = Image.new('RGB', (512, 512), color='gray')  # Mock medical image
    
    medical_image = MedicalImageInput(
        image=sample_image,
        image_type=MedicalImageType.XRAY,
        clinical_history="Patient presents with chest pain and shortness of breath"
    )
    
    query = MultimodalMedicalQuery(
        text_query="What are the findings in this chest X-ray?",
        images=[medical_image],
        domain=MedicalDomain.RADIOLOGY
    )
    
    # Process multimodal query with curriculum learning
    response = await agent.process_multimodal_query(query, DifficultyLevel.MEDIUM)
    
    print("Multimodal Medical Analysis:")
    print(f"Response: {response['text_response']}")
    print(f"Confidence: {response['confidence_score']}")
    print(f"Curriculum Level: {response['curriculum_level']}")
    
    # Run benchmark evaluation
    evaluator = MedVLMBenchmarkEvaluator(agent)
    benchmark_results = await evaluator.run_comprehensive_benchmark()
    
    print("\nBenchmark Results:")
    for dataset, results in benchmark_results["individual_results"].items():
        print(f"{dataset}: {results['accuracy']:.3f} accuracy ({results['correct_answers']}/{results['total_samples']})")
    
    print(f"Overall Accuracy: {benchmark_results['overall_accuracy']:.3f}")

if __name__ == "__main__":
    asyncio.run(demo_medvlm_benchmarking())
