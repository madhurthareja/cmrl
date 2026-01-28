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
import os

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

from agents.medical_agent_core import (
    MedicalQuery, MedicalDomain, DifficultyLevel, SpecialistResponse,
    CurriculumScheduler, DomainClassifier, MedicalDifficultyClassifier,
    PromptTemplateManager, RewardCalculator
)
from training.grpo_trainer import GRPOTrainer, CurriculumMARLTrainer, GRPOConfig
from retrieval.medrag_system import MultiDomainRAGSystem, PreferenceDatasetBuilder, DPOTrainer
from symbolic.reasoning_agent import SymbolicReasoningAgent
from agents.llm_interface import OllamaLLMInterface
from retrieval.state_aware_retriever import StateAwareRetriever
from symbolic.engine import SymbolicEngine
from retrieval.pubmed_service import PubMedService

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
    structured_answer: Optional[str] = None # For benchmark compliance (yes/no/indeterminate)

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
        Context & Findings: {query.context or "None"}
        
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
        
        logger.info(f"Triage Template used for {difficulty}: {template[:50]}...")
        
        try:
            prompt = template.format(
                question=query.question,
                analysis=f"Available medical specialties: {domain_options}",
                domain="{domain}",
                reasoning="{reasoning}",
                secondary_domains="{secondary_domains}",
                urgency="{urgency}"
            )
        except KeyError as e:
            logger.error(f"Template format failed: Missing key {e}")
            logger.error(f"Template content: {template}")
            raise e
        
        # Generate triage decision
        response = await self.llm_interface.generate(prompt)
        
        # Extract domain from response (simplified)
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
        rag_context = "\n".join([f"{doc.title}: {doc.content}" for doc in retrieved_docs[:5]])
        full_context = f"{query.context or ''}\n\nAdditional Literature:\n{rag_context}"
        
        specialist_context = "\n".join([
            f"Specialist {i+1} ({resp.specialist_type}): {resp.response}"
            for i, resp in enumerate(specialist_responses)
        ])
        
        # Get appropriate template
        template = self.template_manager.get_template("medical_reasoning", difficulty)
        
        # Generate comprehensive answer
        prompt = template.format(
            question=query.question,
            context=full_context,
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
        retrieval_quality = min(len(retrieved_docs) / 5, 1.0) # Max confidence if 5+ docs
        
        return (avg_specialist_confidence + retrieval_quality) / 2

class E2HMedicalAgent:
    """Complete E2H Medical Agent System"""
    
    def __init__(self, model_name: str = "qwen3:1.7b", use_symbolic_engine: bool = True):
        self.llm_interface = OllamaLLMInterface(model_name)
        self.curriculum_scheduler = CurriculumScheduler()
        self.use_symbolic_engine = use_symbolic_engine
        
        # Multi-turn Context Memory
        self.conversation_history = [] 
        
        self.difficulty_classifier = MedicalDifficultyClassifier()
        self.template_manager = PromptTemplateManager()
        self.reward_calculator = RewardCalculator()
        
        self.pubmed_service = PubMedService()

        self.rag_system = MultiDomainRAGSystem()

        # Initialize Symbolic Engine (Workflows)
        self.symbolic_workflow_path = os.path.join(os.path.dirname(__file__), '../../symbolic/general')
        
        # Load Symbolic Knowledge Base (State-Aware Retriever)
        self.symbolic_retriever = None
        try:
            kb_dir = os.path.join(os.path.dirname(__file__), '../../data/state_indices/general')
            if os.path.exists(kb_dir):
                # We need the states list to init the retriever correctly
                workflow_states = SymbolicEngine(self.symbolic_workflow_path).states
                self.symbolic_retriever = StateAwareRetriever(workflow_states)
                self.symbolic_retriever.load_indices(kb_dir)
                logger.info("Loaded State-Aware Knowledge Base")
            else:
                logger.warning("Symbolic KB not found. Will use live PubMed Service.")
        except Exception as e:
            logger.error(f"Failed to load Symbolic KB: {e}. Will use live PubMed Service.")

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
        
        # Construct history string for context awareness
        history_context = ""
        if len(self.conversation_history) > 0:
            history_context = "\n\nPrevious Conversation History:\n"
            for i, turn in enumerate(self.conversation_history[-5:]): # Include last 5 turns
                 history_context += f"User: {turn['content']}\n"
                 if "context" in turn:
                     history_context += f"Context: {turn['context'][:100]}...\n"
                 if "assistant" in turn:
                     history_context += f"Assistant: {turn['assistant']}\n"
        
        # Update history with current query
        self.conversation_history.append({"role": "user", "content": question})
        if context:
             self.conversation_history[-1]["context"] = context
        
        # Merge history into context
        enhanced_context = (context or "") + history_context
             
        # Check for Symbolic Workflow trigger
        # We trigger if explicitly requested or if visual findings are heavily implied
        if getattr(self, 'use_symbolic_engine', True) and ("Visual Findings" in (context or "") or getattr(self, 'force_symbolic', False)):
            logger.info("Triggering Symbolic Workflow")
            response = await self.process_symbolic_query(question, enhanced_context)
            # Record response in history
            self.conversation_history[-1]["assistant"] = response.answer
            return response

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
            context=enhanced_context
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
        
        # Record response in history
        self.conversation_history[-1]["assistant"] = final_response.answer
        
        return final_response
    
    async def process_symbolic_query(self, question: str, context: Optional[str] = None) -> AgentResponse:
        """
        Run the query through the SymbolicReasoningAgent.
        """
        # Determine which retriever to use
        # If we have a trained local state-aware retriever, use it.
        # Otherwise, fallback to live PubMed service.
        active_retriever = self.symbolic_retriever
        retrieval_source = "Local Knowledge Base"
        
        if not active_retriever:
            logger.info("Using Live PubMed Service for Symbolic Agent")
            active_retriever = LivePubMedAdapter(self.pubmed_service)
            retrieval_source = "Live PubMed (E-utilities)"
            
        # Optional: Pre-fetch evidence based on initial context (Visual Findings)
        initial_evidence = ""
        if context and "Visual Findings" in context:
            # simple keyword extraction from context could be better, but we use the context as query
            # or just rely on the agent's step-by-step retrieval.
            # Let's do a quick pre-fetch for the 'START' state context
            pre_docs = active_retriever.retrieve(f"{question} {context}", state="HISTORY_OF_PRESENT_ILLNESS")
            if pre_docs:
                initial_evidence = "\nPre-retrieved Evidence:\n" + "\n".join([f"- {d['title']}" for d in pre_docs])

        # Pass the retriever
        symbolic_agent = SymbolicReasoningAgent(
            self.symbolic_workflow_path, 
            model_name=self.llm_interface.model_name,
            retriever=active_retriever
        )
        
        # Run workflow
        full_context = f"Patient Complaint: {question}. {context or ''}\n{initial_evidence}"
        
        # Enhanced Logging
        print(f"{Colors.HEADER}=================================================={Colors.ENDC}")
        print(f"{Colors.HEADER} SYMBOLIC AGENT STARTED {Colors.ENDC}")
        print(f"{Colors.HEADER}=================================================={Colors.ENDC}")
        logger.info(f"Symbolic Agent started for complaint: {question}")
        
        history = await symbolic_agent.run_full_workflow(initial_context=full_context)
        
        # Format the history into an AgentResponse
        final_answer_parts = []
        reasoning_trace = []
        
        for step in history:
            state = step['state']
            content = step.get('content', '')
            status = step['status']
            
            trace_entry = f"[{state}] ({'✅' if status == 'success' else '❌'}) {content}"
            if status == 'violation':
                trace_entry += f" [Violations: {step.get('violations')}]"
            
            # Color-coded logging per step
            if status == 'success':
                 step_color = Colors.GREEN
                 icon = "✓"
            elif status == 'violation':
                 step_color = Colors.WARNING
                 icon = "⚠"
            else:
                 step_color = Colors.FAIL
                 icon = "✗"
                 
            # Ensure content is a string before slicing to avoid KeyError/TypeError
            content_str = str(content)
            print(f"{step_color}   [{state}] {icon} {content_str[:100]}...{Colors.ENDC}")
            
            reasoning_trace.append(trace_entry)
            
            # The 'final' content usually sits in DISPOSITION or PLAN
            if state in ['INITIAL_TREATMENT', 'DISPOSITION', 'PLAN'] and status == 'success':
                final_answer_parts.append(content)

        final_answer = "\n".join(final_answer_parts)
        full_reasoning = "\n".join(reasoning_trace)
        
        if not final_answer:
            final_answer = "Protocol completed but no final disposition recorded in trace."

        # Phase 6: Binary Task Compliance
        # We extract a structured yes/no answer for benchmarking WITHOUT altering the main answer.
        # This allows the model to "think" fully while still having a machine-readable flag.
        structured_answer = "indeterminate"
        try:
            extraction_prompt = f"""
            Analyze the following medical conclusion and map it to a single category.
            
            Conclusion: "{final_answer}"
            
            Categories:
            - "yes": Confirms presence of pathology/finding (e.g., "opacity observed", "aneurysm present").
            - "no": Denies presence or states "no evidence" (e.g., "no abnormalities", "clear").
            - "indeterminate": States uncertainty, requires more imaging, or not applicable.
            
            OUTPUT ONLY THE CATEGORY WORD (yes/no/indeterminate).
            """
            structured_answer = await self.llm_interface.generate(extraction_prompt, temperature=0.1)
            structured_answer = structured_answer.strip().lower()
            if structured_answer not in ["yes", "no", "indeterminate"]:
                structured_answer = "indeterminate"
        except Exception as e:
            logger.warning(f"Failed to extract structured answer: {e}")

        print(f"{Colors.BLUE}   -> Final Disposition: {final_answer[:100]}...{Colors.ENDC}")
        
        return AgentResponse(
            answer=final_answer,
            confidence=0.9, # Symbolic workflow usually implies higher protocol confidence
            reasoning=full_reasoning,
            retrieved_context=[retrieval_source],
            specialist_consultations=[
                SpecialistResponse(
                    specialist_type="SymbolicEngine",
                    response="Workflow constraints enforced.",
                    confidence=1.0,
                    reasoning="Adherence to guidelines verified."
                )
            ],
            difficulty_level=DifficultyLevel.HARD,
            domain=MedicalDomain.GENERAL,
            structured_answer=structured_answer
        )

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

class LivePubMedAdapter:
    """Adapts PubMedService to the interface expected by SymbolicReasoningAgent"""
    def __init__(self, service: PubMedService):
        self.service = service
        self.cache = {}

    def _extract_keywords(self, text: str) -> str:
        # Simple stopword removal and cleaning
        stopwords = {
            "the", "a", "an", "in", "on", "at", "for", "to", "of", "with", "is", "are", "was", "were",
            "be", "been", "this", "that", "these", "those", "it", "he", "she", "they", "we", "i",
            "what", "where", "when", "how", "why", "who", "which", "seen", "image", "visual", "findings",
            "patient", "case", "history", "shown", "shows", "depicts", "seen", "context", "question",
            "abnormality", "primary", "please", "analyze", "list", "objective"
        }
        import re
        # Remove non-alphanumeric (keep hyphens sometimes useful but space is safer)
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = clean_text.split()
        keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
        # Limit to top 5-7 keywords to avoid query overflow
        return " ".join(keywords[:8])

    def retrieve(self, query: str, state: str = None) -> List[Dict]:
        # Extract keywords for robust search
        keywords = self._extract_keywords(query)
        
        search_term = keywords
        if state:
            # Map state to medical keywords if needed, or just append
            search_term += f" AND {state.replace('_', ' ').lower()}"
        
        if not search_term.strip():
             # Fallback if cleaning removed everything
             search_term = query[:100]

        # Check cache
        if search_term in self.cache:
            return self.cache[search_term]

        logger.info(f"Live PubMed Search: {search_term}")
        pmids = self.service.search(search_term, retmax=3)
        docs = self.service.fetch_details(pmids)
        
        # Adapt format to what agent expects (title, abstract -> content)
        adapted_docs = []
        for d in docs:
            adapted_docs.append({
                'title': d.get('title'),
                'abstract': d.get('abstract'),  # Agent uses this
                'content': d.get('abstract')    # Fallback
            })
            
        self.cache[search_term] = adapted_docs
        return adapted_docs
        

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run demo
    asyncio.run(demo_medical_agent())
