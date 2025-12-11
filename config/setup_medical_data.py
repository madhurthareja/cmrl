#!/usr/bin/env python3
"""
Medical Data Setup - Creates directory structure and shows what data to upload
"""

import os
import json

def create_directories():
    """Create required directory structure"""
    directories = [
        "../data/medical_corpus",
        "../data/medvlm_data",
        "../data/medvlm_data/images",
        "../data/medvlm_data/images/radiology",
        "../data/medvlm_data/images/pathology",
        "../data/medvlm_data/images/general"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

def create_example_text_data():
    """Create example medical text data files"""
    domains = ["cardiology", "radiology", "neurology", "pathology", "general"]
    
    for domain in domains:
        example_data = [
            {
                "title": f"Example {domain.title()} Document 1",
                "content": f"This is example medical content for {domain}. Replace with real medical literature.",
                "source": "Example Source",
                "difficulty": "medium",
                "keywords": [domain, "medicine", "healthcare"]
            }
        ]
        
        filename = f"../data/medical_corpus/{domain}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(example_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created example: {filename}")

def create_example_vqa_data():
    """Create example vision-language data structure"""
    example_annotations = [
        {
            "image_path": "images/radiology/example_xray.jpg",
            "question": "What abnormality is visible in this X-ray?",
            "answer": "Example answer - replace with real medical annotations",
            "domain": "radiology",
            "difficulty": "medium"
        },
        {
            "image_path": "images/pathology/example_tissue.jpg", 
            "question": "Describe the pathological findings.",
            "answer": "Example pathology answer - replace with real data",
            "domain": "pathology",
            "difficulty": "hard"
        }
    ]
    
    filename = "../data/medvlm_data/annotations.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(example_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"Created example: {filename}")

def show_data_requirements():
    """Display data requirements to user"""
    print("\n" + "="*60)
    print("MEDICAL DATA REQUIREMENTS")
    print("="*60)
    
    print("\n1. MEDICAL TEXT DATA")
    print("   Location: ./medical_corpus/")
    print("   Files needed:")
    print("   - cardiology.json")
    print("   - radiology.json") 
    print("   - neurology.json")
    print("   - pathology.json")
    print("   - general.json")
    
    print("\n   Format: JSON array with objects containing:")
    print("   - title: Document title")
    print("   - content: Medical text content")
    print("   - source: Source reference")
    print("   - difficulty: easy|medium|hard")
    print("   - keywords: Array of relevant keywords")
    
    print("\n2. MEDICAL IMAGES (Optional)")
    print("   Location: ./medvlm_data/images/")
    print("   Subdirectories:")
    print("   - radiology/ (X-rays, CT scans, MRIs)")
    print("   - pathology/ (Tissue samples, microscopy)")
    print("   - general/ (Other medical images)")
    
    print("\n3. VISION-LANGUAGE ANNOTATIONS (Optional)")
    print("   Location: ./medvlm_data/annotations.json")
    print("   Format: JSON array with objects containing:")
    print("   - image_path: Path to image file")
    print("   - question: Medical question about image")
    print("   - answer: Correct medical answer")
    print("   - domain: Medical domain")
    print("   - difficulty: easy|medium|hard")
    
    print("\n4. DATA SOURCES TO USE:")
    print("   Text Data:")
    print("   - PubMed articles (download via NCBI API)")
    print("   - Medical textbooks (Harrison's, Cecil Medicine)")
    print("   - Clinical guidelines (WHO, CDC, medical societies)")
    print("   - Medical Q&A datasets (MedQuAD, HealthTap)")
    
    print("\n   Vision Data:")
    print("   - VQA-RAD: https://osf.io/89kps/")
    print("   - PathVQA: https://github.com/UCSD-AI4H/PathVQA")
    print("   - MIMIC-CXR: https://physionet.org/content/mimic-cxr/")
    print("   - SLAKE: https://www.med-vqa.com/slake/")
    
    print("\n5. NEXT STEPS:")
    print("   1. Replace example files with your real medical data")
    print("   2. Run: python medical_agent_app.py")
    print("   3. The system will automatically load your data")
    print("   4. Test with medical queries through the web interface")
    
    print("\n" + "="*60)

def main():
    print("Setting up Medical AI System...")
    
    create_directories()
    create_example_text_data()
    create_example_vqa_data()
    show_data_requirements()
    
    print("\nSetup complete! Replace example files with your real medical data.")

if __name__ == "__main__":
    main()
