# E2H Medical Agent System - Core Architecture
# Integrating MMedAgent-RL + MMed-RAG with E2H Curriculum Learning

import asyncio
import json
import logging
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    TRIVIAL = "trivial"
    EASY = "easy" 
    MEDIUM = "medium"
    HARD = "hard"
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            levels = [DifficultyLevel.TRIVIAL, DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
            return levels.index(self) < levels.index(other)
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            levels = [DifficultyLevel.TRIVIAL, DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
            return levels.index(self) <= levels.index(other)
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            levels = [DifficultyLevel.TRIVIAL, DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
            return levels.index(self) > levels.index(other)
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            levels = [DifficultyLevel.TRIVIAL, DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
            return levels.index(self) >= levels.index(other)
        return NotImplemented

class MedicalDomain(Enum):
    GENERAL = "general"
    CARDIOLOGY = "cardiology"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    ONCOLOGY = "oncology"
    NEUROLOGY = "neurology"
    EMERGENCY = "emergency"

@dataclass
class MedicalQuery:
    question: str
    domain: MedicalDomain
    difficulty: DifficultyLevel
    context: Optional[str] = None
    image_path: Optional[str] = None
    
@dataclass
class SpecialistResponse:
    specialist_type: str
    response: str
    confidence: float
    reasoning: str

@dataclass
class RetrievedDocument:
    title: str
    content: str
    similarity_score: float
    domain: MedicalDomain
    complexity_level: float

class MedicalDifficultyClassifier:
    """Classifies medical text complexity based on terminology, reasoning depth, etc."""
    
    def __init__(self):
        # Medical complexity indicators
        self.basic_terms = {
            "headache", "fever", "pain", "cold", "flu", "rest", "sleep", 
            "water", "medicine", "doctor", "hospital", "symptom"
        }
        
        self.advanced_terms = {
            "pathophysiology", "differential diagnosis", "pharmacokinetics",
            "immunohistochemistry", "cytokine", "biomarker", "prognosis",
            "etiology", "phenotype", "genotype", "metabolism"
        }
        
        self.clinical_terms = {
            "diagnosis", "treatment", "therapy", "clinical", "patient",
            "condition", "disease", "syndrome", "disorder", "management"
        }
    
    def classify_complexity(self, text: str) -> float:
        """Returns complexity score from 0.0 (basic) to 1.0 (advanced)"""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        basic_count = len(words.intersection(self.basic_terms))
        clinical_count = len(words.intersection(self.clinical_terms))
        advanced_count = len(words.intersection(self.advanced_terms))
        
        # Calculate complexity score
        total_medical_terms = basic_count + clinical_count + advanced_count
        if total_medical_terms == 0:
            return 0.1  # Non-medical text
            
        # Weighted complexity score
        complexity_score = (
            (basic_count * 0.2) + 
            (clinical_count * 0.6) + 
            (advanced_count * 1.0)
        ) / total_medical_terms
        
        # Text length and structure complexity
        sentence_complexity = min(len(text.split('.')) / 5, 0.3)
        word_complexity = min(len(text.split()) / 100, 0.2)
        
        final_score = min(complexity_score + sentence_complexity + word_complexity, 1.0)
        return final_score
    
    def classify_difficulty_level(self, text: str) -> DifficultyLevel:
        """Convert complexity score to difficulty level"""
        score = self.classify_complexity(text)
        
        if score < 0.3:
            return DifficultyLevel.TRIVIAL
        elif score < 0.5:
            return DifficultyLevel.EASY
        elif score < 0.75:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HARD

class DomainClassifier:
    """Classifies medical questions into domains"""
    
    def __init__(self):
        self.domain_keywords = {
            MedicalDomain.CARDIOLOGY: {
                "heart", "cardiac", "cardiovascular", "ecg", "ekg", "chest pain",
                "blood pressure", "hypertension", "arrhythmia", "myocardial"
            },
            MedicalDomain.RADIOLOGY: {
                "x-ray", "ct", "mri", "ultrasound", "imaging", "scan", "radiograph",
                "contrast", "biopsy", "mammography", "fluoroscopy"
            },
            MedicalDomain.PATHOLOGY: {
                "tissue", "cell", "histology", "cytology", "biopsy", "malignant",
                "benign", "tumor", "carcinoma", "pathology", "microscopic"
            },
            MedicalDomain.ONCOLOGY: {
                "cancer", "tumor", "chemotherapy", "radiation", "oncology", "metastasis",
                "malignancy", "carcinoma", "lymphoma", "leukemia"
            },
            MedicalDomain.NEUROLOGY: {
                "brain", "neural", "nervous", "seizure", "stroke", "migraine",
                "alzheimer", "parkinson", "neurology", "neurological"
            },
            MedicalDomain.EMERGENCY: {
                "emergency", "trauma", "critical", "acute", "urgent", "shock",
                "resuscitation", "er", "ambulance", "life-threatening"
            }
        }
    
    def classify_domain(self, question: str) -> MedicalDomain:
        """Classify question into medical domain"""
        question_lower = question.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, default to GENERAL
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return MedicalDomain.GENERAL

class CurriculumScheduler:
    """E2H Curriculum scheduler adapted for medical learning"""
    
    def __init__(self, max_iterations: int = 100):
        self.iteration = 0
        self.max_iterations = max_iterations
        self.scheduler_type = 'cosine'
        self.levels = [DifficultyLevel.TRIVIAL, DifficultyLevel.EASY, 
                      DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
    
    def get_difficulty_distribution(self) -> Dict[DifficultyLevel, float]:
        """Get current curriculum difficulty distribution"""
        t = self.iteration / self.max_iterations
        
        if self.scheduler_type == 'cosine':
            return self._cosine_scheduler(t)
        else:
            return self._gaussian_scheduler(t)
    
    def _cosine_scheduler(self, t: float) -> Dict[DifficultyLevel, float]:
        """Cosine curriculum: start easy, progress to hard"""
        probs = {}
        for i, level in enumerate(self.levels):
            # Cosine progression from easy to hard
            phase = (t * np.pi / 2) + (i * np.pi / (2 * len(self.levels)))
            probs[level] = max(0, np.cos(phase - t * np.pi / 2))
        
        return self._normalize_probs(probs)
    
    def _gaussian_scheduler(self, t: float) -> Dict[DifficultyLevel, float]:
        """Gaussian curriculum centered on current difficulty"""
        center = t * (len(self.levels) - 1)
        probs = {}
        
        for i, level in enumerate(self.levels):
            distance = abs(i - center)
            probs[level] = np.exp(-(distance ** 2) / (2 * 1.0 ** 2))
        
        return self._normalize_probs(probs)
    
    def _normalize_probs(self, probs: Dict[DifficultyLevel, float]) -> Dict[DifficultyLevel, float]:
        """Normalize probabilities to sum to 1"""
        total = sum(probs.values())
        if total == 0:
            return {level: 0.25 for level in self.levels}
        return {level: prob / total for level, prob in probs.items()}
    
    def sample_difficulty(self) -> Tuple[DifficultyLevel, Dict[DifficultyLevel, float]]:
        """Sample difficulty level based on curriculum"""
        probs = self.get_difficulty_distribution()
        
        # Sample based on probabilities
        rand = np.random.random()
        cumulative = 0
        
        for level in self.levels:
            cumulative += probs[level]
            if rand < cumulative:
                return level, probs
        
        return DifficultyLevel.HARD, probs
    
    def update_iteration(self):
        """Progress to next curriculum iteration"""
        self.iteration = min(self.iteration + 1, self.max_iterations)

class PromptTemplateManager:
    """Manages difficulty-specific prompt templates for medical reasoning"""
    
    def __init__(self):
        self.templates = {
            # Triage prompts by difficulty
            "triage": {
                DifficultyLevel.TRIVIAL: """
                <think>
                This is a basic medical question. I need to identify the most appropriate medical specialty.
                Question: {question}
                
                Based on the key terms and context, this appears to be related to: {analysis}
                </think>
                
                <answer>
                Medical Specialty: {domain}
                Reason: {reasoning}
                </answer>
                """,
                
                DifficultyLevel.HARD: """
                <think>
                This is a complex medical case requiring detailed analysis of multiple systems and specialties.
                Question: {question}
                
                I need to consider:
                1. Primary presenting symptoms and their differential diagnoses
                2. Potential complications and comorbidities  
                3. Multiple specialist consultations that may be needed
                4. Urgency and priority of care
                
                Analysis: {analysis}
                </think>
                
                <answer>
                Primary Medical Specialty: {domain}
                Secondary Consultations: {secondary_domains}
                Clinical Reasoning: {reasoning}
                Urgency Level: {urgency}
                </answer>
                """
            },
            
            # Medical reasoning prompts by difficulty
            "medical_reasoning": {
                DifficultyLevel.TRIVIAL: """
                Question: {question}
                
                Please provide a simple, clear answer focusing on basic medical facts.
                
                <think>
                {thinking_process}
                </think>
                
                <answer>
                {answer}
                </answer>
                """,
                
                DifficultyLevel.HARD: """
                Clinical Case: {question}
                Retrieved Context: {context}
                Specialist Consultations: {specialist_responses}
                
                Please provide a comprehensive medical analysis including:
                1. Differential diagnosis with reasoning
                2. Pathophysiological mechanisms
                3. Evidence-based treatment recommendations
                4. Prognosis and follow-up considerations
                
                <think>
                {thinking_process}
                </think>
                
                <answer>
                {answer}
                </answer>
                """
            }
        }
    
    def get_template(self, template_type: str, difficulty: DifficultyLevel) -> str:
        """Get appropriate template for difficulty level"""
        if template_type in self.templates:
            templates_for_type = self.templates[template_type]
            
            # Use exact match or fallback to closest difficulty
            if difficulty in templates_for_type:
                return templates_for_type[difficulty]
            elif difficulty in [DifficultyLevel.TRIVIAL, DifficultyLevel.EASY]:
                # Try to get trivial template first, fallback to lowest available
                if DifficultyLevel.TRIVIAL in templates_for_type:
                    return templates_for_type[DifficultyLevel.TRIVIAL]
                else:
                    # Get the lowest difficulty level available
                    available_levels = list(templates_for_type.keys())
                    if available_levels:
                        return templates_for_type[min(available_levels)]
                    else:
                        return "Question: {question}\n\nAnswer: {answer}"
            else:
                # Try to get hard template first, fallback to highest available
                if DifficultyLevel.HARD in templates_for_type:
                    return templates_for_type[DifficultyLevel.HARD]
                else:
                    # Get the highest difficulty level available
                    available_levels = list(templates_for_type.keys())
                    if available_levels:
                        return templates_for_type[max(available_levels)]
                    else:
                        return "Question: {question}\n\nAnswer: {answer}"
        
        return "Question: {question}\n\nAnswer: {answer}"

class RewardCalculator:
    """Calculate rewards for GRPO training"""
    
    def __init__(self):
        self.format_patterns = {
            'think_block': r'<think>(.*?)</think>',
            'answer_block': r'<answer>(.*?)</answer>'
        }
    
    def calculate_format_reward(self, response: str) -> float:
        """Calculate format compliance reward (0, 0.5, or 1.0)"""
        has_think = bool(re.search(self.format_patterns['think_block'], response, re.DOTALL))
        has_answer = bool(re.search(self.format_patterns['answer_block'], response, re.DOTALL))
        
        if has_think and has_answer:
            return 1.0
        elif has_answer:
            return 0.5
        else:
            return 0.0
    
    def calculate_accuracy_reward(self, response: str, ground_truth: str) -> float:
        """Calculate accuracy reward (0 or 1)"""
        # Extract answer from response
        answer_match = re.search(self.format_patterns['answer_block'], response, re.DOTALL)
        if not answer_match:
            return 0.0
        
        predicted_answer = answer_match.group(1).strip().lower()
        ground_truth_lower = ground_truth.strip().lower()
        
        # Simple accuracy check - can be enhanced with semantic similarity
        if predicted_answer in ground_truth_lower or ground_truth_lower in predicted_answer:
            return 1.0
        return 0.0
    
    def calculate_total_reward(self, response: str, ground_truth: str) -> float:
        """Calculate total reward (format + accuracy)"""
        format_reward = self.calculate_format_reward(response)
        accuracy_reward = self.calculate_accuracy_reward(response, ground_truth)
        return format_reward + accuracy_reward

# Base interfaces for the medical agent system
class LLMInterface(ABC):
    """Abstract interface for language models"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def calculate_logprob(self, text: str, prompt: str) -> float:
        pass

class SpecialistInterface(ABC):
    """Abstract interface for specialist models"""
    
    @abstractmethod
    async def consult(self, query: MedicalQuery) -> SpecialistResponse:
        pass

class RetrieverInterface(ABC):
    """Abstract interface for document retrieval"""
    
    @abstractmethod
    async def retrieve(self, query: str, domain: MedicalDomain, k: int = 10) -> List[RetrievedDocument]:
        pass

if __name__ == "__main__":
    # Test the core components
    difficulty_classifier = MedicalDifficultyClassifier()
    domain_classifier = DomainClassifier()
    scheduler = CurriculumScheduler()
    template_manager = PromptTemplateManager()
    reward_calculator = RewardCalculator()
    
    # Test medical question classification
    test_questions = [
        "What is a headache?",
        "Explain the pathophysiology of myocardial infarction with ST elevation",
        "How do you interpret an ECG?",
        "Differential diagnosis for acute chest pain in emergency department"
    ]
    
    for question in test_questions:
        difficulty = difficulty_classifier.classify_difficulty_level(question)
        domain = domain_classifier.classify_domain(question)
        complexity_score = difficulty_classifier.classify_complexity(question)
        
        print(f"\nQuestion: {question}")
        print(f"Domain: {domain.value}")
        print(f"Difficulty: {difficulty.value}")
        print(f"Complexity Score: {complexity_score:.3f}")
    
    # Test curriculum progression
    print(f"\n=== Curriculum Progression ===")
    for i in range(0, 101, 20):
        scheduler.iteration = i
        difficulty, probs = scheduler.sample_difficulty()
        print(f"Iteration {i}: {difficulty.value}")
        for level, prob in probs.items():
            print(f"  {level.value}: {prob:.3f}")
    
    print("\nCore medical agent components initialized successfully.")
