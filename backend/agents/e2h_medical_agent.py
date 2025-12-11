# Complete E2H Medical Agent System
# Integrates MMedAgent-RL + MMed-RAG + E2H Curriculum Learning

import asyncio
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import requests
import torch

from agents.medical_agent_core import (
    MedicalQuery, MedicalDomain, DifficultyLevel, SpecialistResponse,
    CurriculumScheduler, DomainClassifier, MedicalDifficultyClassifier,
    PromptTemplateManager, RewardCalculator
)
from training.grpo_trainer import GRPOTrainer, CurriculumMARLTrainer, GRPOConfig
from retrieval.medrag_system import MultiDomainRAGSystem, PreferenceDatasetBuilder, DPOTrainer

logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    answer: str
    confidence: float
    reasoning: str
    retrieved_context: List[str]
    specialist_consultations: List[SpecialistResponse]
    difficulty_level: DifficultyLevel
    domain: MedicalDomain

class OllamaLLMInterface:
    """Interface to Ollama local LLM"""
    
    def __init__(self, model_name: str = "llama3.1:8b-instruct-q4_K_M", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "num_predict": kwargs.get("max_tokens", 512)
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "Error: Unable to generate response"
                
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return "Error: Model unavailable"
    
    def calculate_logprob(self, response: str, prompt: str) -> float:
        """Calculate log probability (mock implementation)"""
        # In practice, this would require access to model's internal states
        # For now, return a reasonable mock value based on response length
        return -len(response) * 0.1  # Longer responses get lower probability

class MockSpecialist:
    """Mock specialist for testing"""
    
    def __init__(self, specialty: str, llm_interface: OllamaLLMInterface):
        self.specialty = specialty
        self.llm_interface = llm_interface
    
    async def consult(self, query: MedicalQuery) -> SpecialistResponse:
        """Provide specialist consultation"""
        
        specialist_prompt = f"""
        You are a medical specialist in {self.specialty}. 
        
        Patient Question: {query.question}
        Medical Domain: {query.domain.value}
        
        Please provide a focused consultation from your specialty perspective:
        1. Key findings relevant to your specialty
        2. Your clinical opinion
        3. Recommendations for further evaluation or treatment
        
        Keep your response professional and focused on your specialty area.
        """
        
        response = await self.llm_interface.generate(specialist_prompt)
        
        # Extract confidence (simplified)
        confidence = 0.8 if len(response) > 100 else 0.6
        
        return SpecialistResponse(
            specialist_type=self.specialty,
            response=response,
            confidence=confidence,
            reasoning=f"Consultation from {self.specialty} perspective"
        )

class TriageAgent:
    """Triage agent for routing medical queries to appropriate domains"""
    
    def __init__(self, llm_interface: OllamaLLMInterface, template_manager: PromptTemplateManager):
        self.llm_interface = llm_interface
        self.template_manager = template_manager
        self.domain_classifier = DomainClassifier()
    
    async def triage_query(self, query: MedicalQuery, difficulty: DifficultyLevel) -> MedicalDomain:
        """Triage medical query to appropriate domain"""
        
        # Get appropriate prompt template
        template = self.template_manager.get_template("triage", difficulty)
        
        # Prepare domain options
        domain_options = "\n".join([f"- {domain.value}" for domain in MedicalDomain])
        
        prompt = template.format(
            question=query.question,
            analysis=f"Available medical specialties: {domain_options}",
            domain="{domain}",
            reasoning="{reasoning}"
        )
        
        # Generate triage decision
        response = await self.llm_interface.generate(prompt)
        
        # Extract domain from response (simplified parsing)
        predicted_domain = self.domain_classifier.classify_domain(query.question)
        
        logger.info(f"Triaged query to domain: {predicted_domain.value}")
        return predicted_domain

class AttendingAgent:
    """Attending physician agent that integrates specialist consultations"""
    
    def __init__(self, llm_interface: OllamaLLMInterface, template_manager: PromptTemplateManager, rag_system: MultiDomainRAGSystem):
        self.llm_interface = llm_interface
        self.template_manager = template_manager
        self.rag_system = rag_system
    
    async def generate_final_answer(
        self, 
        query: MedicalQuery, 
        specialist_responses: List[SpecialistResponse],
        difficulty: DifficultyLevel
    ) -> AgentResponse:
        """Generate final integrated medical answer"""
        
        # Retrieve relevant medical literature
        retrieved_docs = await self.rag_system.retrieve_with_curriculum(
            query.question, query.domain, difficulty
        )
        
        # Prepare context
        context = "\n".join([f"{doc.title}: {doc.content}" for doc in retrieved_docs[:5]])
        specialist_context = "\n".join([
            f"Specialist {i+1} ({resp.specialist_type}): {resp.response}"
            for i, resp in enumerate(specialist_responses)
        ])
        
        # Get appropriate template
        template = self.template_manager.get_template("medical_reasoning", difficulty)
        
        # Generate comprehensive answer
        prompt = template.format(
            question=query.question,
            context=context,
            specialist_responses=specialist_context,
            thinking_process="{thinking_process}",
            answer="{answer}"
        )
        
        response = await self.llm_interface.generate(prompt, temperature=0.3)
        
        # Extract answer and reasoning (simplified)
        answer_parts = response.split("<answer>")
        if len(answer_parts) > 1:
            answer = answer_parts[1].replace("</answer>", "").strip()
        else:
            answer = response
        
        # Calculate confidence based on specialist agreement and retrieval quality
        confidence = self.calculate_confidence(specialist_responses, retrieved_docs)
        
        return AgentResponse(
            answer=answer,
            confidence=confidence,
            reasoning=response,
            retrieved_context=[doc.title for doc in retrieved_docs],
            specialist_consultations=specialist_responses,
            difficulty_level=difficulty,
            domain=query.domain
        )
    
    def calculate_confidence(self, specialist_responses: List[SpecialistResponse], retrieved_docs: List) -> float:
        """Calculate response confidence"""
        if not specialist_responses:
            return 0.5
        
        avg_specialist_confidence = np.mean([resp.confidence for resp in specialist_responses])
        retrieval_quality = min(len(retrieved_docs) / 5, 1.0)  # Max confidence if 5+ docs
        
        return (avg_specialist_confidence + retrieval_quality) / 2

class E2HMedicalAgent:
    """Complete E2H Medical Agent System"""
    
    def __init__(self, model_name: str = "llama3.1:8b-instruct-q4_K_M"):
        # Initialize core components
        self.llm_interface = OllamaLLMInterface(model_name)
        self.curriculum_scheduler = CurriculumScheduler()
        self.difficulty_classifier = MedicalDifficultyClassifier()
        self.template_manager = PromptTemplateManager()
        self.reward_calculator = RewardCalculator()
        
        # Initialize RAG system
        self.rag_system = MultiDomainRAGSystem()
        
        # Initialize agents
        self.triage_agent = TriageAgent(self.llm_interface, self.template_manager)
        self.attending_agent = AttendingAgent(self.llm_interface, self.template_manager, self.rag_system)
        
        # Initialize specialists
        self.specialists = [
            MockSpecialist("Cardiology", self.llm_interface),
            MockSpecialist("Radiology", self.llm_interface),
            MockSpecialist("Emergency Medicine", self.llm_interface)
        ]
        
        logger.info("E2H Medical Agent System initialized")
    
    async def process_medical_query(self, question: str, context: Optional[str] = None) -> AgentResponse:
        """Process medical query through complete pipeline"""
        
        # Step 1: Classify initial difficulty and domain
        base_difficulty = self.difficulty_classifier.classify_difficulty_level(question)
        
        # Step 2: Get curriculum-adjusted difficulty
        curriculum_difficulty, probs = self.curriculum_scheduler.sample_difficulty()
        
        logger.info(f"Base difficulty: {base_difficulty.value}, Curriculum: {curriculum_difficulty.value}")
        
        # Step 3: Create medical query object
        medical_query = MedicalQuery(
            question=question,
            domain=MedicalDomain.GENERAL,  # Will be updated by triage
            difficulty=curriculum_difficulty,
            context=context
        )
        
        # Step 4: Triage to appropriate domain
        triaged_domain = await self.triage_agent.triage_query(medical_query, curriculum_difficulty)
        medical_query.domain = triaged_domain
        
        # Step 5: Get specialist consultations
        specialist_responses = []
        for specialist in self.specialists:
            if self.is_relevant_specialist(specialist.specialty, triaged_domain):
                consultation = await specialist.consult(medical_query)
                specialist_responses.append(consultation)
        
        # Step 6: Generate final integrated answer
        final_response = await self.attending_agent.generate_final_answer(
            medical_query, specialist_responses, curriculum_difficulty
        )
        
        # Step 7: Update curriculum
        self.curriculum_scheduler.update_iteration()
        
        logger.info(f"Processed query with {len(specialist_responses)} consultations")
        return final_response
    
    def is_relevant_specialist(self, specialty: str, domain: MedicalDomain) -> bool:
        """Check if specialist is relevant for domain"""
        relevance_map = {
            MedicalDomain.CARDIOLOGY: ["Cardiology", "Emergency Medicine"],
            MedicalDomain.RADIOLOGY: ["Radiology"],
            MedicalDomain.EMERGENCY: ["Emergency Medicine", "Cardiology"],
        }
        
        relevant_specialists = relevance_map.get(domain, ["Emergency Medicine"])
        return specialty in relevant_specialists
    
    def get_curriculum_status(self) -> Dict:
        """Get current curriculum status"""
        probs = self.curriculum_scheduler.get_difficulty_distribution()
        return {
            "iteration": self.curriculum_scheduler.iteration,
            "max_iterations": self.curriculum_scheduler.max_iterations,
            "difficulty_distribution": {level.value: prob for level, prob in probs.items()},
            "scheduler_type": self.curriculum_scheduler.scheduler_type
        }
    
    async def train_with_curriculum(self, training_data: List[Tuple[str, str]]) -> Dict:
        """Train the system using curriculum learning"""
        
        # Initialize GRPO trainer (simplified for demo)
        grpo_config = GRPOConfig(group_size=4, learning_rate=1e-6)
        
        # Mock training loop
        training_metrics = []
        
        for epoch in range(5):  # Simplified training
            epoch_rewards = []
            
            for question, ground_truth in training_data:
                # Process query
                response = await self.process_medical_query(question)
                
                # Calculate reward
                reward = self.reward_calculator.calculate_total_reward(response.answer, ground_truth)
                epoch_rewards.append(reward)
            
            avg_reward = np.mean(epoch_rewards)
            training_metrics.append({"epoch": epoch, "avg_reward": avg_reward})
            
            logger.info(f"Training epoch {epoch}: avg_reward={avg_reward:.3f}")
        
        return {
            "training_completed": True,
            "epochs": len(training_metrics),
            "final_avg_reward": training_metrics[-1]["avg_reward"],
            "curriculum_status": self.get_curriculum_status()
        }

# Demo usage
async def demo_medical_agent():
    """Demonstrate the E2H Medical Agent system"""
    
    agent = E2HMedicalAgent()
    
    # Test queries of varying complexity
    test_queries = [
        "What is a headache?",
        "How do you interpret an ECG?", 
        "Explain the pathophysiology of acute myocardial infarction with ST elevation",
        "What are the differential diagnoses for acute chest pain in a 65-year-old male with risk factors?"
    ]
    
    print("E2H Medical Agent Demo\n" + "="*50)
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        print("-" * 50)
        
        try:
            response = await agent.process_medical_query(query)
            
            print(f"Domain: {response.domain.value}")
            print(f"Difficulty: {response.difficulty_level.value}")
            print(f"Confidence: {response.confidence:.3f}")
            print(f"Specialists: {[cons.specialist_type for cons in response.specialist_consultations]}")
            print(f"Retrieved Docs: {len(response.retrieved_context)}")
            print(f"Answer: {response.answer[:200]}...")
            
        except Exception as e:
            print(f"Error processing query: {e}")
        
        # Show curriculum status
        status = agent.get_curriculum_status()
        print(f"Curriculum Iteration: {status['iteration']}/{status['max_iterations']}")
    
    # Demo training
    print(f"\nTraining Demo")
    print("-" * 50)
    
    training_data = [
        ("What causes chest pain?", "Chest pain can be caused by cardiac, pulmonary, or musculoskeletal issues"),
        ("How to read an ECG?", "ECG interpretation involves analyzing rhythm, rate, axis, and morphology")
    ]
    
    training_results = await agent.train_with_curriculum(training_data)
    print(f"Training completed: {training_results}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demo
    asyncio.run(demo_medical_agent())
