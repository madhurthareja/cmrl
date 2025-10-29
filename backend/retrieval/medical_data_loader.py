"""
Medical Data Loader - Loads medical text and vision-language data
Replaces mock data with real medical datasets
"""

import json
import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from PIL import Image

from agents.medical_agent_core import MedicalDomain

logger = logging.getLogger(__name__)

@dataclass
class MedicalDocument:
    """Medical document with metadata"""
    title: str
    content: str
    source: str
    difficulty: str
    keywords: List[str]
    domain: MedicalDomain

@dataclass
class MedicalVQA:
    """Medical Visual Question Answering item"""
    image_path: str
    question: str
    answer: str
    domain: str
    difficulty: str
    mask_path: Optional[str] = None

class MedicalDataLoader:
    """Loads medical text and vision-language datasets"""
    
    def __init__(self, corpus_dir: str = "./data/medical_corpus", medvlm_dir: str = "./data/medvlm_data"):
        self.corpus_dir = corpus_dir
        self.medvlm_dir = medvlm_dir
        
    def load_text_corpus(self, domain: MedicalDomain) -> List[MedicalDocument]:
        """Load medical text corpus for a specific domain"""
        domain_file = os.path.join(self.corpus_dir, f"{domain.value}.json")
        
        if not os.path.exists(domain_file):
            logger.warning(f"No data file found for {domain.value}, using mock data")
            return self._create_mock_documents(domain)
        
        try:
            with open(domain_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            for item in data:
                doc = MedicalDocument(
                    title=item.get('title', ''),
                    content=item.get('content', ''),
                    source=item.get('source', 'Unknown'),
                    difficulty=item.get('difficulty', 'medium'),
                    keywords=item.get('keywords', []),
                    domain=domain
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents for {domain.value}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {domain_file}: {e}")
            return self._create_mock_documents(domain)
    
    def load_medvlm_data(self) -> List[MedicalVQA]:
        """Load medical vision-language QA data"""
        annotations_file = os.path.join(self.medvlm_dir, "annotations.json")
        
        if not os.path.exists(annotations_file):
            logger.warning("No vision-language annotations found, creating mock data")
            return self._create_mock_vqa_data()
        
        try:
            with open(annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            vqa_items = []
            for item in data:
                vqa = MedicalVQA(
                    image_path=item.get('image_path', ''),
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    domain=item.get('domain', 'general'),
                    difficulty=item.get('difficulty', 'medium')
                )
                vqa_items.append(vqa)
            
            logger.info(f"Loaded {len(vqa_items)} VQA items")
            return vqa_items
            
        except Exception as e:
            logger.error(f"Error loading VQA data: {e}")
            return self._create_mock_vqa_data()
    
    def _create_mock_documents(self, domain: MedicalDomain) -> List[MedicalDocument]:
        """Create mock documents when real data is unavailable"""
        mock_data = {
            MedicalDomain.CARDIOLOGY: [
                {
                    "title": "Myocardial Infarction Diagnosis",
                    "content": "Acute myocardial infarction requires immediate recognition and treatment...",
                    "difficulty": "hard"
                },
                {
                    "title": "Basic ECG Interpretation",
                    "content": "Normal sinus rhythm shows regular P waves followed by QRS complexes...",
                    "difficulty": "easy"
                }
            ],
            MedicalDomain.RADIOLOGY: [
                {
                    "title": "Chest X-ray Normal Anatomy",
                    "content": "Normal chest X-ray shows clear lung fields with visible cardiac silhouette...",
                    "difficulty": "easy"
                },
                {
                    "title": "CT Angiography Interpretation",
                    "content": "CT angiography provides detailed vascular imaging for diagnosis...",
                    "difficulty": "hard"
                }
            ]
        }
        
        domain_data = mock_data.get(domain, [
            {
                "title": f"General {domain.value} Knowledge",
                "content": f"Basic medical knowledge in {domain.value} domain...",
                "difficulty": "medium"
            }
        ])
        
        documents = []
        for item in domain_data:
            doc = MedicalDocument(
                title=item["title"],
                content=item["content"],
                source="Mock Data",
                difficulty=item["difficulty"],
                keywords=[domain.value],
                domain=domain
            )
            documents.append(doc)
        
        return documents
    
    def _create_mock_vqa_data(self) -> List[MedicalVQA]:
        """Create mock VQA data when real data is unavailable"""
        mock_vqa = [
            MedicalVQA(
                image_path="images/radiology/chest_xray_001.jpg",
                question="What abnormality is visible in this chest X-ray?",
                answer="No acute abnormalities detected. Normal cardiac and pulmonary structures.",
                domain="radiology",
                difficulty="medium"
            ),
            MedicalVQA(
                image_path="images/pathology/tissue_001.jpg",
                question="Describe the pathological findings in this tissue sample.",
                answer="Normal tissue architecture with no signs of malignancy or inflammation.",
                domain="pathology",
                difficulty="hard"
            )
        ]
        
        return mock_vqa
    
    def validate_data_structure(self) -> Dict[str, bool]:
        """Validate that required data directories and files exist"""
        validation = {}
        
        # Check text corpus directory
        validation['corpus_dir_exists'] = os.path.exists(self.corpus_dir)
        
        # Check for domain-specific files
        for domain in MedicalDomain:
            domain_file = os.path.join(self.corpus_dir, f"{domain.value}.json")
            validation[f'{domain.value}_file'] = os.path.exists(domain_file)
        
        # Check vision-language data
        validation['medvlm_dir_exists'] = os.path.exists(self.medvlm_dir)
        validation['annotations_file'] = os.path.exists(os.path.join(self.medvlm_dir, "annotations.json"))
        
        return validation
