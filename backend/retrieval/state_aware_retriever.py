import faiss
import numpy as np
import logging
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from retrieval.pubmed_service import PubMedService

logger = logging.getLogger(__name__)

class StateAwareRetriever:
    """
    Retriever that maintains separate indices for different reasoning states.
    Corresponds to Step 5: Index with Intent.
    """
    def __init__(self, states: List[str]):
        self.states = states
        # Initialize encoder
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer: {e}. using random embeddings for demo.")
            self.encoder = None
            self.embedding_dim = 384

        # Create an index for each state
        self.indices: Dict[str, faiss.IndexFlatL2] = {}
        self.documents: Dict[str, List[Dict]] = {}
        
        for state in self.states:
            self.indices[state] = faiss.IndexFlatL2(self.embedding_dim)
            self.documents[state] = []

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.encoder:
            return self.encoder.encode(texts)
        # Mock encoding
        return np.random.rand(len(texts), self.embedding_dim).astype('float32')

    def add_documents(self, state: str, docs: List[Dict]):
        """
        Add documents to a specific state's index.
        docs should be dicts with 'content' key (e.g. abstract).
        """
        if state not in self.indices:
            logger.warning(f"State {state} not known, skipping indexing.")
            return

        valid_docs = [d for d in docs if d.get('abstract')]
        if not valid_docs:
            return
            
        texts = [d['abstract'] for d in valid_docs]
        embeddings = self._encode(texts)
        
        self.indices[state].add(embeddings)
        self.documents[state].extend(valid_docs)
        logger.info(f"Added {len(valid_docs)} documents to index for {state}")

    def retrieve(self, query: str, state: str, k: int = 3) -> List[Dict]:
        """
        Retrieve documents relevant to the query from the specific state's index.
        """
        if state not in self.indices:
            logger.warning(f"No index for state {state}")
            return []
            
        index = self.indices[state]
        if index.ntotal == 0:
            return []
            
        query_vec = self._encode([query])
        distances, doc_indices = index.search(query_vec, k)
        
        results = []
        for idx in doc_indices[0]:
            if idx != -1 and idx < len(self.documents[state]):
                results.append(self.documents[state][idx])
                
        return results

    def populate_from_pubmed(self, pubmed_service: PubMedService, queries_per_state: Dict[str, str]):
        """
        Populate indices by searching PubMed for intended topics per state.
        Example: 
        {
            'HISTORY_OF_PRESENT_ILLNESS': 'chest pain history taking guidelines',
            'DIFFERENTIAL_GENERATION': 'acute chest pain differential diagnosis review'
        }
        """
        for state, query in queries_per_state.items():
            logger.info(f"Fetching PubMed data for state: {state} with query: '{query}'")
            docs = pubmed_service.search_and_fetch(query, retmax=5)
            self.add_documents(state, docs)

    def save_indices(self, directory: str):
        """
        Save all FAISS indices and document stores to disk.
        """
        import pickle
        import os
        
        os.makedirs(directory, exist_ok=True)
        
        # Save indices
        for state, index in self.indices.items():
            if index.ntotal > 0:
                faiss.write_index(index, os.path.join(directory, f"{state}.index"))
        
        # Save documents
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
            
        logger.info(f"Saved {len(self.indices)} indices to {directory}")

    def load_indices(self, directory: str):
        """
        Load indices and documents from disk.
        """
        import pickle
        import os
        
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist.")
            return

        # Load documents
        doc_path = os.path.join(directory, "documents.pkl")
        if os.path.exists(doc_path):
            with open(doc_path, "rb") as f:
                self.documents = pickle.load(f)
        
        # Load indices
        for state in self.states:
            index_path = os.path.join(directory, f"{state}.index")
            if os.path.exists(index_path):
                self.indices[state] = faiss.read_index(index_path)
                logger.info(f"Loaded index for {state} with {self.indices[state].ntotal} vectors")
            else:
                # Re-init empty if not found
                self.indices[state] = faiss.IndexFlatL2(self.embedding_dim)
