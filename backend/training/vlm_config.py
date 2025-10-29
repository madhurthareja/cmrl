# VLM Model Configurations for Medical Finetuning

VLM_MODELS = {
    "llava-med": {
        "model_name": "microsoft/llava-med-v1.5-mistral-7b",
        "description": "LLaVA-Med v1.5 with Mistral 7B - General medical VLM",
        "image_size": 336,
        "max_length": 512,
        "supports_medical": True,
        "recommended_batch_size": 2,
        "recommended_lr": 2e-5
    },

    "medvlm": {
        "model_name": "microsoft/MedVLM-7B",
        "description": "MedVLM - Specialized medical vision-language model",
        "image_size": 224,
        "max_length": 512,
        "supports_medical": True,
        "recommended_batch_size": 4,
        "recommended_lr": 1e-5
    },

    "biomed-clip": {
        "model_name": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "description": "BioMed CLIP - Biomedical image-text matching",
        "image_size": 224,
        "max_length": 256,
        "supports_medical": True,
        "recommended_batch_size": 8,
        "recommended_lr": 5e-6
    },

    "llava-general": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "description": "LLaVA v1.5 - General purpose VLM (not medical-specific)",
        "image_size": 336,
        "max_length": 512,
        "supports_medical": False,
        "recommended_batch_size": 2,
        "recommended_lr": 2e-5
    },

    "blip2": {
        "model_name": "Salesforce/blip2-opt-2.7b",
        "description": "BLIP-2 - General vision-language model",
        "image_size": 224,
        "max_length": 512,
        "supports_medical": False,
        "recommended_batch_size": 4,
        "recommended_lr": 1e-5
    }
}

# Training presets for different scenarios
TRAINING_PRESETS = {
    "medical_finetune": {
        "num_epochs": 3,
        "batch_size": 2,
        "learning_rate": 2e-5,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 100,
        "use_lora": True,
        "lora_r": 16,
        "use_8bit": True,
        "max_grad_norm": 1.0
    },

    "medical_adaptation": {
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 50,
        "use_lora": True,
        "lora_r": 8,
        "use_8bit": True,
        "max_grad_norm": 1.0
    },

    "cpu_finetune": {
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 5e-5,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 10,
        "use_lora": True,
        "lora_r": 8,
        "use_8bit": False,  # Disable quantization for CPU
        "max_grad_norm": 1.0
    }
}

# UMIE Dataset configurations
UMIE_DATASETS = {
    "chest_xray14": {
        "description": "Chest X-ray 14 disease classification dataset",
        "domain": "radiology",
        "task_type": "multilabel_classification",
        "modality": "x-ray",
        "labels": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
                  "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
                  "Fibrosis", "Pleural_Thickening", "Hernia"],
        "recommended_model": "llava-med",
        "estimated_samples": 112120
    },

    "brain_tumor_classification": {
        "description": "Brain tumor classification from MRI",
        "domain": "neurology",
        "task_type": "classification",
        "modality": "mri",
        "labels": ["glioma", "meningioma", "notumor", "pituitary"],
        "recommended_model": "llava-med",
        "estimated_samples": 3264
    },

    "covid19_detection": {
        "description": "COVID-19 detection in chest X-rays",
        "domain": "radiology",
        "task_type": "classification",
        "modality": "x-ray",
        "labels": ["COVID-19", "Normal", "Pneumonia"],
        "recommended_model": "llava-med",
        "estimated_samples": 3616
    },

    "alzheimers": {
        "description": "Alzheimer's disease classification from brain MRI",
        "domain": "neurology",
        "task_type": "classification",
        "modality": "mri",
        "labels": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"],
        "recommended_model": "llava-med",
        "estimated_samples": 6400
    },

    "knee_osteoarthritis": {
        "description": "Knee osteoarthritis grading from X-rays",
        "domain": "orthopedics",
        "task_type": "classification",
        "modality": "x-ray",
        "labels": ["0", "1", "2", "3", "4"],  # KL grades
        "recommended_model": "llava-med",
        "estimated_samples": 1650
    },

    "kits23": {
        "description": "Kidney and tumor segmentation in CT scans",
        "domain": "urology",
        "task_type": "segmentation",
        "modality": "ct",
        "labels": ["kidney", "tumor", "cyst"],
        "recommended_model": "llava-med",
        "estimated_samples": 489,  # KiTS23 dataset size
        "has_masks": True
    },

    "coronahack": {
        "description": "COVID-19 detection from chest X-rays (CoronaHack)",
        "domain": "radiology",
        "task_type": "classification",
        "modality": "x-ray",
        "labels": ["COVID-19", "Normal", "Pneumonia", "Lung_Opacity"],
        "recommended_model": "llava-med",
        "estimated_samples": 14862
    },

    "finding_and_measuring_lungs": {
        "description": "Lung segmentation in CT scans",
        "domain": "pulmonology",
        "task_type": "segmentation",
        "modality": "ct",
        "labels": ["lung"],
        "recommended_model": "llava-med",
        "estimated_samples": 50,  # Limited dataset
        "has_masks": True
    },

    "brain_with_intracranial_hemorrhage": {
        "description": "Intracranial hemorrhage segmentation in CT",
        "domain": "neurology",
        "task_type": "segmentation",
        "modality": "ct",
        "labels": ["hemorrhage"],
        "recommended_model": "llava-med",
        "estimated_samples": 100,  # Limited dataset
        "has_masks": True
    }
}

# Data configuration
DATA_CONFIG = {
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "max_image_size": 1024,  # Maximum image dimension
    "resize_mode": "resize",  # "resize" or "crop"
    "image_quality": 95,  # JPEG quality for saving
    "text_max_length": 512,
    "conversation_format": "llava"  # "llava", "blip2", etc.
}

def get_model_config(model_key: str) -> dict:
    """Get configuration for a specific model"""
    if model_key not in VLM_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(VLM_MODELS.keys())}")
    return VLM_MODELS[model_key]

def get_training_preset(preset_key: str) -> dict:
    """Get training preset configuration"""
    if preset_key not in TRAINING_PRESETS:
        raise ValueError(f"Unknown preset: {preset_key}. Available: {list(TRAINING_PRESETS.keys())}")
    return TRAINING_PRESETS[preset_key]

def validate_data_directory(data_dir: str) -> dict:
    """Validate VLM data directory structure"""
    import os
    from pathlib import Path

    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }

    data_path = Path(data_dir)

    # Check if directory exists
    if not data_path.exists():
        validation["valid"] = False
        validation["errors"].append(f"Data directory does not exist: {data_dir}")
        return validation

    # Check annotations file
    annotations_file = data_path / "annotations.json"
    if not annotations_file.exists():
        validation["valid"] = False
        validation["errors"].append("annotations.json not found")
        return validation

    # Check images directory
    images_dir = data_path / "images"
    if not images_dir.exists():
        validation["warnings"].append("images/ directory not found - using placeholder images")

    # Validate annotations format
    try:
        import json
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            validation["errors"].append("annotations.json should contain a list")
            validation["valid"] = False
            return validation

        # Check sample annotation
        if len(data) > 0:
            sample = data[0]
            required_fields = ["image_path", "question", "answer"]
            for field in required_fields:
                if field not in sample:
                    validation["errors"].append(f"Missing required field '{field}' in annotations")
                    validation["valid"] = False

        validation["stats"]["total_samples"] = len(data)

    except json.JSONDecodeError as e:
        validation["errors"].append(f"Invalid JSON in annotations.json: {e}")
        validation["valid"] = False

    return validation