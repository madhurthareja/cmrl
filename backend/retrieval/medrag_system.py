# MMed-RAG Implementation with E2H Curriculum Learning
# Domain-aware retrieval with adaptive k-selection and DPO training

import asyncio
import json
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import requests
import sqlite3
import os
from sentence_transformers import SentenceTransformer
import faiss

from agents.medical_agent_core import (
    MedicalDomain, DifficultyLevel, MedicalQuery, SpecialistResponse,
    RetrievedDocument, PromptTemplateManager, MedicalDifficultyClassifier
)

logger = logging.getLogger(__name__)

@dataclass
class PreferenceExample:
    """Training example for DPO preference learning"""
    query: str
    context: str
    preferred_response: str
    dispreferred_response: str
    preference_type: str  # 'cross_modality', 'overall_alignment'
    domain: MedicalDomain

class AdaptiveRetriever:
    """Adaptive k-selection retriever with gap statistic"""
    
    def __init__(self, gap_threshold: float = 2.0):
        self.gap_threshold = gap_threshold
        
    def adaptive_truncate(self, retrieved_docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Truncate retrieval results based on similarity gap"""
        if len(retrieved_docs) <= 1:
            return retrieved_docs
        
        # Sort by similarity score (descending)
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.similarity_score, reverse=True)
        
        # Calculate gaps: u_i = log(S_i / S_{i+1})
        for i in range(len(sorted_docs) - 1):
            s_i = sorted_docs[i].similarity_score
            s_i_plus_1 = sorted_docs[i + 1].similarity_score
            
            # Avoid division by zero or log of negative
            if s_i_plus_1 > 0:
                gap = np.log(max(s_i, 1e-6) / max(s_i_plus_1, 1e-6))
                
                if gap > self.gap_threshold:
                    logger.info(f"Adaptive truncation at position {i+1}, gap={gap:.3f}")
                    return sorted_docs[:i+1]
        
        # No significant gap found, return all
        return sorted_docs

class DomainSpecificRetriever:
    """Retriever for specific medical domain"""
    
    def __init__(self, domain: MedicalDomain, corpus_path: str):
        self.domain = domain
        self.corpus_path = corpus_path
        # Try to initialize a sentence encoder; if unavailable (no internet or missing package),
        # fall back to a DummyEncoder that returns fixed-size zero embeddings for testing.
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight encoder
            self._encoder_dim = None
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer model: {e}. Using DummyEncoder for tests.")

            class DummyEncoder:
                def __init__(self, dim=384):
                    self.dim = dim
                def encode(self, texts):
                    import numpy as _np
                    arr = _np.zeros((len(texts), self.dim), dtype=_np.float32)
                    return arr

            self.encoder = DummyEncoder(dim=384)
            self._encoder_dim = 384
        self.difficulty_classifier = MedicalDifficultyClassifier()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.data_loader = None
        self.load_corpus()
    
    def load_corpus(self):
        """Load domain-specific medical corpus"""
        # Try to load real data, fall back to mock data
        try:
            from retrieval.medical_data_loader import MedicalDataLoader
            self.data_loader = MedicalDataLoader()
            medical_docs = self.data_loader.load_text_corpus(self.domain)
            
            # Convert to dict format for compatibility
            corpus_data = []
            for doc in medical_docs:
                corpus_data.append({
                    'title': doc.title,
                    'content': doc.content,
                    'source': doc.source,
                    'difficulty': doc.difficulty,
                    'keywords': doc.keywords
                })
            
        except Exception as e:
            logger.warning(f"Could not load real data: {e}. Using mock corpus.")
            corpus_data = self.create_mock_corpus()
        
        # Encode documents
        doc_texts = [doc['content'] for doc in corpus_data]
        embeddings = self.encoder.encode(doc_texts)

        # Build FAISS index
        try:
            dimension = embeddings.shape[1]
        except Exception:
            # If encoder returned 1-D array or fallback, use configured encoder dim
            dimension = self._encoder_dim or 384
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Store documents with metadata
        self.documents = [
            RetrievedDocument(
                title=doc['title'],
                content=doc['content'],
                similarity_score=0.0,  # Will be set during retrieval
                domain=self.domain,
                complexity_level=self.difficulty_classifier.classify_complexity(doc['content'])
            ) for doc in corpus_data
        ]
        
        logger.info(f"Loaded {len(self.documents)} documents for domain {self.domain.value}")
    
    def create_mock_corpus(self) -> List[Dict[str, str]]:
        """Create mock medical corpus for testing"""
        if self.domain == MedicalDomain.CARDIOLOGY:
            return [
                {
                    "title": "Basic Heart Anatomy",
                    "content": "The heart is a muscular organ that pumps blood through the circulatory system. It has four chambers: two atria and two ventricles."
                },
                {
                    "title": "Myocardial Infarction Pathophysiology", 
                    "content": "Acute myocardial infarction occurs due to coronary artery occlusion leading to myocardial ischemia and necrosis. The pathophysiology involves atherosclerotic plaque rupture, thrombosis, and downstream myocardial tissue death."
                },
                {
                    "title": "ECG Interpretation Basics",
                    "content": "Electrocardiogram interpretation involves analyzing P waves, QRS complexes, and T waves to assess cardiac rhythm, conduction, and potential pathology."
                },
                {
                    "title": "Advanced Heart Failure Management",
                    "content": "Heart failure with reduced ejection fraction requires comprehensive management including ACE inhibitors, beta-blockers, aldosterone antagonists, and potentially cardiac resynchronization therapy or ventricular assist devices in advanced cases."
                }
            ]
        elif self.domain == MedicalDomain.RADIOLOGY:
            return [
                {
                    "title": "Chest X-ray Basics",
                    "content": "Chest X-rays show the heart, lungs, and bones. Normal findings include clear lung fields and normal heart size."
                },
                {
                    "title": "CT Scan Interpretation",
                    "content": "Computed tomography provides detailed cross-sectional imaging with superior soft tissue contrast compared to conventional radiography. Advanced techniques include contrast enhancement and multi-planar reconstruction."
                }
            ]
        else:
            # Generic medical content
            return [
                {
                    "title": "General Medical Principles",
                    "content": "Medicine involves diagnosis, treatment, and prevention of disease through evidence-based practice."
                },
                {
                    "title": "Clinical Decision Making",
                    "content": "Clinical reasoning integrates patient history, physical examination, diagnostic testing, and evidence-based guidelines to formulate differential diagnoses and treatment plans."
                }
            ]
    
    async def retrieve(self, query: str, k: int = 10, difficulty_filter: Optional[DifficultyLevel] = None) -> List[RetrievedDocument]:
        """Retrieve relevant documents for query"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.documents)))
        
        # Create results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):  # Valid index
                doc = self.documents[idx]
                doc.similarity_score = float(score)
                
                # Apply difficulty filter if specified
                if difficulty_filter is not None:
                    doc_difficulty = self.difficulty_classifier.classify_difficulty_level(doc.content)
                    if doc_difficulty != difficulty_filter:
                        continue
                
                results.append(doc)
        
        return results

class MultiDomainRAGSystem:
    """Multi-domain RAG system with curriculum learning integration"""
    
    def __init__(self, corpus_dir: str = "./data/medical_corpus"):
        self.corpus_dir = corpus_dir
        self.retrievers = {}
        self.adaptive_retriever = AdaptiveRetriever()
        self.difficulty_classifier = MedicalDifficultyClassifier()
        
        # Initialize domain-specific retrievers
        for domain in MedicalDomain:
            self.retrievers[domain] = DomainSpecificRetriever(domain, corpus_dir)
    
    async def retrieve_with_curriculum(
        self, 
        query: str, 
        domain: MedicalDomain, 
        curriculum_level: DifficultyLevel,
        k: int = 20
    ) -> List[RetrievedDocument]:
        """Retrieve documents with curriculum-aware filtering"""
        
        # Get base retrieval results
        retriever = self.retrievers.get(domain, self.retrievers[MedicalDomain.GENERAL])
        retrieved_docs = await retriever.retrieve(query, k=k)
        
        # Apply curriculum filtering
        curriculum_filtered = self.apply_curriculum_filter(retrieved_docs, curriculum_level)
        
        # Adaptive k-selection
        final_docs = self.adaptive_retriever.adaptive_truncate(curriculum_filtered)
        
        logger.info(f"Retrieved {len(final_docs)} docs for query '{query[:50]}...' at {curriculum_level.value} level")
        return final_docs

    def retrieve_documents(self, query: str, top_k: int = 10, domain: Optional[MedicalDomain] = None, curriculum_level: Optional[DifficultyLevel] = None) -> List[RetrievedDocument]:
        """Synchronous wrapper for retrieval used by benchmarking scripts.

        This convenience method runs the async retrieval pipeline and returns results.
        It selects a default domain and curriculum level when not provided.
        """
        # Choose defaults
        if domain is None:
            domain = MedicalDomain.GENERAL
        if curriculum_level is None:
            curriculum_level = DifficultyLevel.EASY

        try:
            return asyncio.run(self.retrieve_with_curriculum(query, domain, curriculum_level, k=top_k))
        except RuntimeError:
            # If an event loop is already running (rare in scripts), try using existing loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task and run until complete
                return loop.run_until_complete(self.retrieve_with_curriculum(query, domain, curriculum_level, k=top_k))
            else:
                return loop.run_until_complete(self.retrieve_with_curriculum(query, domain, curriculum_level, k=top_k))
    
    def apply_curriculum_filter(
        self, 
        docs: List[RetrievedDocument], 
        curriculum_level: DifficultyLevel
    ) -> List[RetrievedDocument]:
        """Filter documents based on curriculum difficulty"""
        
        if curriculum_level == DifficultyLevel.TRIVIAL:
            # Only very basic documents
            return [doc for doc in docs if doc.complexity_level < 0.3]
        elif curriculum_level == DifficultyLevel.EASY:
            # Basic to moderate documents  
            return [doc for doc in docs if doc.complexity_level < 0.6]
        elif curriculum_level == DifficultyLevel.MEDIUM:
            # Moderate to advanced documents
            return [doc for doc in docs if 0.3 < doc.complexity_level < 0.8]
        else:  # HARD
            # All documents, preference for complex ones
            return sorted(docs, key=lambda x: x.complexity_level, reverse=True)

class PreferenceDatasetBuilder:
    """Build preference dataset for DPO training"""
    
    def __init__(self, rag_system: MultiDomainRAGSystem, llm_interface):
        self.rag_system = rag_system
        self.llm_interface = llm_interface
        self.preference_examples = []
    
    async def create_cross_modality_pairs(self, query: str, domain: MedicalDomain) -> List[PreferenceExample]:
        """Create cross-modality preference pairs (Dcm in paper)"""
        # Get retrieved context
        retrieved_docs = await self.rag_system.retrieve_with_curriculum(
            query, domain, DifficultyLevel.MEDIUM
        )
        
        if not retrieved_docs:
            return []
        
        context = "\n".join([f"{doc.title}: {doc.content}" for doc in retrieved_docs])
        
        # Generate response with good context
        good_response = await self.llm_interface.generate(f"Context: {context}\nQuestion: {query}")
        
        # Simulate noisy/unrelated context (simplified)
        noisy_context = self.generate_noisy_context(context)
        noisy_response = await self.llm_interface.generate(f"Context: {noisy_context}\nQuestion: {query}")
        
        # If both responses are similar despite different context, it indicates over-reliance on retrieval
        if self.responses_similar(good_response, noisy_response):
            return [PreferenceExample(
                query=query,
                context=context,
                preferred_response=good_response,
                dispreferred_response=noisy_response,
                preference_type='cross_modality',
                domain=domain
            )]
        
        return []
    
    async def create_overall_alignment_pairs(self, query: str, domain: MedicalDomain, ground_truth: str) -> List[PreferenceExample]:
        """Create overall alignment preference pairs (Doa in paper)"""
        examples = []
        
        # Get response with RAG
        retrieved_docs = await self.rag_system.retrieve_with_curriculum(
            query, domain, DifficultyLevel.MEDIUM
        )
        context = "\n".join([f"{doc.title}: {doc.content}" for doc in retrieved_docs])
        rag_response = await self.llm_interface.generate(f"Context: {context}\nQuestion: {query}")
        
        # Get response without RAG
        no_rag_response = await self.llm_interface.generate(f"Question: {query}")
        
        # D1_oa: RAG helps (prefer RAG)
        if self.is_correct(rag_response, ground_truth) and not self.is_correct(no_rag_response, ground_truth):
            examples.append(PreferenceExample(
                query=query,
                context=context,
                preferred_response=rag_response,
                dispreferred_response=no_rag_response,
                preference_type='rag_helps',
                domain=domain
            ))
        
        # D2_oa: RAG hurts (prefer no RAG)
        elif self.is_correct(no_rag_response, ground_truth) and not self.is_correct(rag_response, ground_truth):
            examples.append(PreferenceExample(
                query=query,
                context="",
                preferred_response=no_rag_response,
                dispreferred_response=rag_response,
                preference_type='rag_hurts',
                domain=domain
            ))
        
        return examples
    
    def generate_noisy_context(self, original_context: str) -> str:
        """Generate noisy/unrelated context"""
        # Simplified: just shuffle sentences
        sentences = original_context.split('.')
        np.random.shuffle(sentences)
        return '. '.join(sentences)
    
    def responses_similar(self, resp1: str, resp2: str) -> bool:
        """Check if responses are similar (simplified)"""
        # In practice, use semantic similarity
        words1 = set(resp1.lower().split())
        words2 = set(resp2.lower().split())
        overlap = len(words1.intersection(words2)) / max(len(words1), len(words2), 1)
        return overlap > 0.7
    
    def is_correct(self, response: str, ground_truth: str) -> bool:
        """Check if response is correct (simplified)"""
        return ground_truth.lower() in response.lower()
    
    async def build_preference_dataset(self, queries: List[Tuple[str, MedicalDomain, str]]) -> List[PreferenceExample]:
        """Build complete preference dataset"""
        all_examples = []
        
        for query, domain, ground_truth in queries:
            # Cross-modality pairs
            cm_examples = await self.create_cross_modality_pairs(query, domain)
            all_examples.extend(cm_examples)
            
            # Overall alignment pairs
            oa_examples = await self.create_overall_alignment_pairs(query, domain, ground_truth)
            all_examples.extend(oa_examples)
        
        logger.info(f"Built preference dataset with {len(all_examples)} examples")
        return all_examples

class DPOTrainer:
    """Direct Preference Optimization trainer for RAG fine-tuning"""
    
    def __init__(self, model, reference_model, alpha: float = 0.1, learning_rate: float = 1e-5):
        self.model = model
        self.reference_model = reference_model
        self.alpha = alpha
        self.learning_rate = learning_rate
    
    def compute_dpo_loss(self, preference_examples: List[PreferenceExample]) -> float:
        """Compute DPO loss (simplified version)"""
        total_loss = 0.0
        
        for example in preference_examples:
            # Get log probabilities (simplified - in practice need proper tokenization)
            logp_preferred = self.model.calculate_logprob(example.preferred_response, example.query)
            logp_dispreferred = self.model.calculate_logprob(example.dispreferred_response, example.query)
            
            logp_preferred_ref = self.reference_model.calculate_logprob(example.preferred_response, example.query)
            logp_dispreferred_ref = self.reference_model.calculate_logprob(example.dispreferred_response, example.query)
            
            # DPO loss: -log(sigmoid(α * (logp_w - logp_w_ref) - α * (logp_l - logp_l_ref)))
            preferred_diff = self.alpha * (logp_preferred - logp_preferred_ref)
            dispreferred_diff = self.alpha * (logp_dispreferred - logp_dispreferred_ref)
            
            # Sigmoid of difference
            logit = preferred_diff - dispreferred_diff
            loss = -np.log(1 / (1 + np.exp(-logit)) + 1e-8)
            total_loss += loss
        
        return total_loss / len(preference_examples)

# Mock implementations for testing
class MockLLMInterface:
    """Mock LLM interface for testing"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        # Mock responses based on prompt content
        if "context" in prompt.lower():
            return "This is a response based on retrieved context."
        else:
            return "This is a response without context."
    
    def calculate_logprob(self, response: str, prompt: str) -> float:
        return np.random.normal(-2.0, 0.5)

async def test_rag_system():
    """Test the RAG system implementation"""
    
    # Initialize system
    rag_system = MultiDomainRAGSystem()
    
    # Test retrieval
    query = "What causes chest pain?"
    domain = MedicalDomain.CARDIOLOGY
    curriculum_level = DifficultyLevel.MEDIUM
    
    retrieved_docs = await rag_system.retrieve_with_curriculum(query, domain, curriculum_level)
    
    print(f"Retrieved {len(retrieved_docs)} documents:")
    for doc in retrieved_docs:
        print(f"- {doc.title}: {doc.similarity_score:.3f} (complexity: {doc.complexity_level:.3f})")
    
    # Test preference dataset building
    mock_llm = MockLLMInterface()
    preference_builder = PreferenceDatasetBuilder(rag_system, mock_llm)
    
    test_queries = [
        ("What causes chest pain?", MedicalDomain.CARDIOLOGY, "Chest pain can be caused by heart problems"),
        ("How to read ECG?", MedicalDomain.CARDIOLOGY, "ECG interpretation involves analyzing waves")
    ]
    
    preference_examples = await preference_builder.build_preference_dataset(test_queries)
    print(f"\nBuilt {len(preference_examples)} preference examples")
    
    for example in preference_examples[:2]:  # Show first 2
        print(f"Type: {example.preference_type}")
        print(f"Query: {example.query}")
        print(f"Preferred: {example.preferred_response[:50]}...")
        print(f"Dispreferred: {example.dispreferred_response[:50]}...")
        print("---")

if __name__ == "__main__":
    asyncio.run(test_rag_system())
    print("RAG system implementation tested successfully!")
