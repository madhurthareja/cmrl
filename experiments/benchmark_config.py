"""
Benchmark configurations for different testing scenarios
"""

# Sample test queries for comprehensive evaluation
BENCHMARK_QUERIES = [
    # General Medicine
    {
        "query": "What are the common symptoms of diabetes?",
        "category": "general_medicine",
        "difficulty": "easy",
        "expected_topics": ["diabetes", "symptoms", "blood sugar", "thirst", "urination"]
    },
    {
        "query": "Explain the pathophysiology of Type 2 diabetes mellitus",
        "category": "endocrinology",
        "difficulty": "hard",
        "expected_topics": ["insulin resistance", "beta cells", "glucose metabolism"]
    },
    
    # Cardiology
    {
        "query": "What is the normal range for blood pressure in adults?",
        "category": "cardiology",
        "difficulty": "easy",
        "expected_topics": ["blood pressure", "systolic", "diastolic", "mmHg"]
    },
    {
        "query": "Describe the ECG changes in acute myocardial infarction",
        "category": "cardiology",
        "difficulty": "hard",
        "expected_topics": ["ECG", "ST elevation", "myocardial infarction", "Q waves"]
    },
    
    # Neurology
    {
        "query": "What are the early signs of Alzheimer's disease?",
        "category": "neurology",
        "difficulty": "medium",
        "expected_topics": ["Alzheimer's", "memory loss", "cognitive decline", "dementia"]
    },
    {
        "query": "Explain the Glasgow Coma Scale scoring system",
        "category": "neurology",
        "difficulty": "medium",
        "expected_topics": ["Glasgow Coma Scale", "consciousness", "neurological assessment"]
    },
    
    # Radiology
    {
        "query": "What does pneumonia look like on a chest X-ray?",
        "category": "radiology",
        "difficulty": "medium",
        "expected_topics": ["pneumonia", "chest X-ray", "consolidation", "opacity"]
    },
    {
        "query": "Describe the radiological features of knee osteoarthritis",
        "category": "radiology",
        "difficulty": "medium",
        "expected_topics": ["osteoarthritis", "joint space narrowing", "osteophytes", "sclerosis"]
    },
    
    # Pharmacology
    {
        "query": "What are the common side effects of aspirin?",
        "category": "pharmacology",
        "difficulty": "easy",
        "expected_topics": ["aspirin", "side effects", "bleeding", "stomach"]
    },
    {
        "query": "Explain the mechanism of action of ACE inhibitors",
        "category": "pharmacology",
        "difficulty": "hard",
        "expected_topics": ["ACE inhibitors", "angiotensin", "blood pressure", "mechanism"]
    },
    
    # Emergency Medicine
    {
        "query": "How do you manage anaphylactic shock?",
        "category": "emergency_medicine",
        "difficulty": "hard",
        "expected_topics": ["anaphylaxis", "epinephrine", "airway", "emergency"]
    },
    {
        "query": "What are the signs of dehydration in children?",
        "category": "pediatrics",
        "difficulty": "medium",
        "expected_topics": ["dehydration", "children", "signs", "fluid loss"]
    },
    
    # Complex Clinical Scenarios
    {
        "query": "A 65-year-old patient presents with chest pain and shortness of breath. What is your differential diagnosis?",
        "category": "clinical_reasoning",
        "difficulty": "hard",
        "expected_topics": ["chest pain", "shortness of breath", "differential diagnosis", "cardiac", "pulmonary"]
    },
    {
        "query": "Interpret these lab values: Glucose 250 mg/dL, HbA1c 9.5%, Creatinine 2.1 mg/dL",
        "category": "laboratory_interpretation",
        "difficulty": "hard",
        "expected_topics": ["glucose", "HbA1c", "creatinine", "diabetes", "kidney function"]
    },
    
    # Medical Procedures
    {
        "query": "Describe the steps for performing CPR",
        "category": "procedures",
        "difficulty": "medium",
        "expected_topics": ["CPR", "chest compressions", "rescue breathing", "AED"]
    },
    {
        "query": "What are the indications for lumbar puncture?",
        "category": "procedures",
        "difficulty": "medium",
        "expected_topics": ["lumbar puncture", "CSF", "meningitis", "contraindications"]
    }
]

# Benchmark configurations for different evaluation modes
BENCHMARK_CONFIGS = {
    "quick": {
        "description": "Quick evaluation with basic queries",
        "queries": BENCHMARK_QUERIES[:5],
        "timeout": 30,
        "retrieval_top_k": 3
    },
    
    "comprehensive": {
        "description": "Comprehensive evaluation across all medical domains",
        "queries": BENCHMARK_QUERIES,
        "timeout": 60,
        "retrieval_top_k": 5
    },
    
    "retrieval_focus": {
        "description": "Focus on retrieval system performance",
        "queries": BENCHMARK_QUERIES,
        "timeout": 15,
        "retrieval_top_k": 10,
        "mode": "retrieval"
    },
    
    "clinical_reasoning": {
        "description": "Focus on complex clinical reasoning scenarios",
        "queries": [q for q in BENCHMARK_QUERIES if q["difficulty"] == "hard"],
        "timeout": 90,
        "retrieval_top_k": 7
    }
}

# Evaluation criteria for different aspects
EVALUATION_CRITERIA = {
    "accuracy": {
        "description": "How factually correct is the response?",
        "scale": "1-5 (1=incorrect, 5=highly accurate)"
    },
    
    "completeness": {
        "description": "Does the response cover all important aspects?",
        "scale": "1-5 (1=incomplete, 5=comprehensive)"
    },
    
    "clarity": {
        "description": "How clear and understandable is the response?",
        "scale": "1-5 (1=confusing, 5=very clear)"
    },
    
    "clinical_relevance": {
        "description": "How relevant is the response for clinical practice?",
        "scale": "1-5 (1=not relevant, 5=highly relevant)"
    },
    
    "safety": {
        "description": "Does the response avoid potentially harmful advice?",
        "scale": "1-5 (1=potentially harmful, 5=safe)"
    }
}

# Expected performance benchmarks
PERFORMANCE_TARGETS = {
    "response_time": {
        "excellent": "< 2 seconds",
        "good": "< 5 seconds", 
        "acceptable": "< 10 seconds",
        "poor": "> 10 seconds"
    },
    
    "success_rate": {
        "excellent": "> 95%",
        "good": "> 90%",
        "acceptable": "> 80%",
        "poor": "< 80%"
    },
    
    "retrieval_precision": {
        "excellent": "> 0.8",
        "good": "> 0.6",
        "acceptable": "> 0.4",
        "poor": "< 0.4"
    }
}