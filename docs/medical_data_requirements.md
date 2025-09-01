# Medical Data Requirements

## Required Data Format and Structure

### 1. Medical Text Corpus
**Location**: `./medical_corpus/`
**Format**: JSON files organized by medical domain

```
medical_corpus/
├── cardiology.json
├── radiology.json  
├── neurology.json
├── pathology.json
├── general.json
└── ...
```

**JSON Structure**:
```json
[
  {
    "title": "Document title",
    "content": "Medical text content",
    "source": "PubMed/Textbook/Guidelines",
    "difficulty": "easy|medium|hard",
    "keywords": ["keyword1", "keyword2"]
  }
]
```

### 2. Medical Vision-Language Data (Optional)
**Location**: `./medvlm_data/`
**Format**: JSON + Images

```
medvlm_data/
├── images/
│   ├── radiology/
│   ├── pathology/
│   └── general/
└── annotations.json
```

**Annotations Structure**:
```json
[
  {
    "image_path": "images/radiology/chest_xray_001.jpg",
    "question": "What abnormality is visible in this chest X-ray?",
    "answer": "Pneumonia in the right lower lobe",
    "domain": "radiology",
    "difficulty": "medium"
  }
]
```

## Data Sources to Integrate

### Text Data
1. **PubMed Articles**: Download via NCBI API
2. **Medical Textbooks**: Harrison's, Cecil Medicine (if available)
3. **Clinical Guidelines**: WHO, CDC, medical society guidelines
4. **Medical Q&A**: MedQuAD, HealthTap datasets

### Vision-Language Data
1. **VQA-RAD**: 315 radiology images with 3,515 QA pairs
2. **PathVQA**: 32,799 pathology images with 234,775 QA pairs
3. **MIMIC-CXR**: 377,110 chest X-rays with reports
4. **SLAKE**: 642 medical images with 14,028 bilingual QA pairs

## Integration Steps

1. Create the directory structure above
2. Place your medical texts in the appropriate domain JSON files
3. Run the system - it will automatically load from these files
4. For vision data, place images in the images/ subdirectories
5. Create the annotations.json file with questions and answers

## Current System Status
- Currently uses mock data generated in `medrag_system.py`
- Replace `create_mock_corpus()` method to load from your JSON files
- Vision capabilities available in `medvlm_extension.py`
