# UMIE Dataset Loader for VLM Finetuning
# Handles lion-ai/umie_datasets with multiple medical imaging tasks

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np
from datasets import load_dataset, Dataset

from retrieval.medical_data_loader import MedicalVQA, MedicalDocument

logger = logging.getLogger(__name__)

@dataclass
class UMIEConfig:
    """Configuration for UMIE dataset loading"""
    dataset_name: str = "lion-ai/umie_datasets"
    config_name: str = "chest_xray14"  # Default config
    split: str = "train"
    cache_dir: str = "./data/umie_cache"
    max_samples: Optional[int] = None
    image_size: Tuple[int, int] = (224, 224)

class UMIEDataLoader:
    """Loader for UMIE medical imaging datasets"""

    # Available UMIE configs and their properties
    UMIE_CONFIGS = {
        "chest_xray14": {
            "task_type": "classification",
            "description": "Chest X-ray 14 disease classification",
            "domain": "radiology",
            "modality": "x-ray",
            "labels": ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
                      "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
                      "Fibrosis", "Pleural_Thickening", "Hernia"],
            "multilabel": True
        },
        "brain_tumor_classification": {
            "task_type": "classification",
            "description": "Brain tumor classification from MRI",
            "domain": "neurology",
            "modality": "mri",
            "labels": ["glioma", "meningioma", "notumor", "pituitary"],
            "multilabel": False
        },
        "covid19_detection": {
            "task_type": "classification",
            "description": "COVID-19 detection in chest X-rays",
            "domain": "radiology",
            "modality": "x-ray",
            "labels": ["COVID-19", "Normal", "Pneumonia"],
            "multilabel": False
        },
        "alzheimers": {
            "task_type": "classification",
            "description": "Alzheimer's disease classification from brain MRI",
            "domain": "neurology",
            "modality": "mri",
            "labels": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"],
            "multilabel": False
        },
        "knee_osteoarthritis": {
            "task_type": "classification",
            "description": "Knee osteoarthritis grading from X-rays",
            "domain": "orthopedics",
            "modality": "x-ray",
            "labels": ["0", "1", "2", "3", "4"],  # KL grades
            "multilabel": False
        },
        "kits23": {
            "task_type": "segmentation",
            "description": "Kidney and tumor segmentation in CT",
            "domain": "urology",
            "modality": "ct",
            "labels": ["kidney", "tumor", "cyst"],
            "multilabel": False,
            "has_masks": True
        },
        "coronahack": {
            "task_type": "classification",
            "description": "COVID-19 detection from chest X-rays",
            "domain": "radiology",
            "modality": "x-ray",
            "labels": ["COVID-19", "Normal", "Pneumonia", "Lung_Opacity"],
            "multilabel": False
        },
        "finding_and_measuring_lungs": {
            "task_type": "segmentation",
            "description": "Lung segmentation in CT scans",
            "domain": "pulmonology",
            "modality": "ct",
            "labels": ["lung"],
            "multilabel": False,
            "has_masks": True
        },
        "brain_with_intracranial_hemorrhage": {
            "task_type": "segmentation",
            "description": "Intracranial hemorrhage segmentation in CT",
            "domain": "neurology",
            "modality": "ct",
            "labels": ["hemorrhage"],
            "multilabel": False,
            "has_masks": True
        }
    }

    def __init__(self, config: UMIEConfig):
        self.config = config
        os.makedirs(config.cache_dir, exist_ok=True)

    def load_dataset(self) -> Dataset:
        """Load UMIE dataset from HuggingFace"""
        logger.info(f"Loading UMIE dataset: {self.config.config_name}")

        try:
            # Load the dataset
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.config_name,
                split=self.config.split,
                cache_dir=self.config.cache_dir
            )

            # Limit samples if specified
            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

            logger.info(f"Loaded {len(dataset)} samples from {self.config.config_name}")
            return dataset

        except Exception as e:
            logger.error(f"Failed to load UMIE dataset: {e}")
            raise

    def convert_to_vqa_format(self, dataset: Dataset) -> List[MedicalVQA]:
        """Convert UMIE dataset to VQA format for VLM training"""
        logger.info("Converting UMIE dataset to VQA format")

        vqa_items = []
        config_info = self.UMIE_CONFIGS.get(self.config.config_name, {})

        for idx, sample in enumerate(dataset):
            try:
                # Extract image
                if 'image' not in sample:
                    logger.warning(f"Sample {idx} missing image, skipping")
                    continue

                image = sample['image']
                if not isinstance(image, Image.Image):
                    logger.warning(f"Sample {idx} image is not PIL Image, skipping")
                    continue

                # Resize image if needed
                if image.size != self.config.image_size:
                    image = image.resize(self.config.image_size, Image.Resampling.LANCZOS)

                # Save image to local directory
                image_filename = f"{self.config.config_name}_{idx}.jpg"
                image_path = os.path.join(self.config.cache_dir, "images", image_filename)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path, "JPEG", quality=95)

                # Handle segmentation masks if available
                mask_path = None
                if config_info.get("has_masks", False) and 'mask' in sample:
                    mask = sample['mask']
                    if isinstance(mask, (np.ndarray, Image.Image)):
                        # Convert to PIL Image if numpy array
                        if isinstance(mask, np.ndarray):
                            # Assume mask is in HWC format, convert to HW if needed
                            if mask.ndim == 3 and mask.shape[-1] == 1:
                                mask = mask.squeeze(-1)
                            mask = Image.fromarray(mask.astype(np.uint8))

                        mask_filename = f"{self.config.config_name}_{idx}_mask.png"
                        mask_path = os.path.join(self.config.cache_dir, "masks", mask_filename)
                        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                        mask.save(mask_path, "PNG")

                # Create question and answer based on task type
                question, answer = self._create_qa_pair(sample, config_info)

                # Create VQA item
                vqa_item = MedicalVQA(
                    image_path=image_path,
                    question=question,
                    answer=answer,
                    domain=config_info.get("domain", "radiology"),
                    difficulty=self._estimate_difficulty(sample)
                )

                # Add mask path if available
                if mask_path:
                    vqa_item.mask_path = mask_path

                vqa_items.append(vqa_item)

            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue

        logger.info(f"Converted {len(vqa_items)} samples to VQA format")
        return vqa_items

    def _create_qa_pair(self, sample: Dict, config_info: Dict) -> Tuple[str, str]:
        """Create question-answer pair based on dataset type"""
        task_type = config_info.get("task_type", "classification")
        labels = config_info.get("labels", [])
        multilabel = config_info.get("multilabel", False)
        has_masks = config_info.get("has_masks", False)

        if task_type == "classification":
            if multilabel:
                # Multi-label classification (e.g., Chest X-ray 14)
                # Labels are stored as a JSON string in the dataset
                try:
                    if 'labels' in sample and isinstance(sample['labels'], str):
                        # Parse the JSON string to get label dictionary
                        import json
                        label_dict = json.loads(sample['labels'])
                        # Now label_dict is a dictionary, so we can call .items()
                        active_labels = [label for label, value in label_dict.items() if value == 1]
                    else:
                        active_labels = []

                    if active_labels:
                        question = f"What abnormalities are visible in this {config_info.get('domain', 'medical')} image?"
                        answer = f"The image shows: {', '.join(active_labels)}."
                    else:
                        question = f"Are there any abnormalities in this {config_info.get('domain', 'medical')} image?"
                        answer = "The image appears normal with no significant abnormalities detected."
                except (json.JSONDecodeError, KeyError, AttributeError):
                    question = f"Describe the findings in this {config_info.get('domain', 'medical')} image."
                    answer = "This is a medical image for diagnostic analysis."
            else:
                # Single-label classification
                if 'label' in sample and isinstance(sample['label'], int) and sample['label'] < len(labels):
                    predicted_class = labels[sample['label']]
                    question = f"What is shown in this {config_info.get('domain', 'medical')} image?"
                    answer = f"This image shows {predicted_class}."
                else:
                    question = f"Describe this {config_info.get('domain', 'medical')} image."
                    answer = "This is a medical image for diagnostic analysis."

        elif task_type == "segmentation":
            # Segmentation tasks (e.g., KiTS23, lung segmentation)
            if has_masks and 'mask' in sample:
                # If we have segmentation masks, create questions about identifying structures
                question = f"Can you identify and segment the {', '.join(labels)} in this {config_info.get('modality', 'medical')} image?"
                answer = f"This {config_info.get('modality', 'medical')} image contains {', '.join(labels)} that can be segmented. The segmentation mask shows the precise boundaries of these structures."
            else:
                # Fallback for segmentation without masks
                question = f"What anatomical structures are visible in this {config_info.get('modality', 'medical')} image?"
                answer = f"This image shows {', '.join(labels)} that would typically be segmented for analysis."

        else:
            # Default fallback
            question = f"What do you see in this {config_info.get('domain', 'medical')} image?"
            answer = "This appears to be a medical image requiring professional interpretation."

        return question, answer

    def _estimate_difficulty(self, sample: Dict) -> str:
        """Estimate difficulty level of the sample"""
        # Simple heuristic based on available metadata
        if 'label' in sample:
            # For classification tasks, assume varying difficulty
            label = sample['label']
            if isinstance(label, int):
                # Alternate between difficulty levels
                difficulties = ['easy', 'medium', 'hard']
                return difficulties[label % len(difficulties)]

        return 'medium'  # Default difficulty

    def save_vqa_dataset(self, vqa_items: List[MedicalVQA], output_file: str):
        """Save VQA items to JSON file compatible with existing VLM training"""
        logger.info(f"Saving {len(vqa_items)} VQA items to {output_file}")

        # Convert to dictionary format
        data = []
        for item in vqa_items:
            item_dict = {
                "image_path": item.image_path,
                "question": item.question,
                "answer": item.answer,
                "domain": item.domain,
                "difficulty": item.difficulty
            }
            # Add mask path if available
            if item.mask_path:
                item_dict["mask_path"] = item.mask_path

            data.append(item_dict)

        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved VQA dataset to {output_file}")

def create_umie_vqa_dataset(config_name: str = "chest_xray14",
                           max_samples: int = 1000,
                           output_file: str = "./data/medvlm_data/umie_annotations.json"):
    """Convenience function to create VQA dataset from UMIE"""
    config = UMIEConfig(
        config_name=config_name,
        max_samples=max_samples
    )

    loader = UMIEDataLoader(config)
    dataset = loader.load_dataset()
    vqa_items = loader.convert_to_vqa_format(dataset)
    loader.save_vqa_dataset(vqa_items, output_file)

    return vqa_items

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Convert UMIE dataset to VQA format")
    parser.add_argument("--config", type=str, default="chest_xray14",
                       choices=list(UMIEDataLoader.UMIE_CONFIGS.keys()),
                       help="UMIE dataset configuration")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of samples to process")
    parser.add_argument("--output", type=str, default="./data/medvlm_data/umie_annotations.json",
                       help="Output JSON file")

    args = parser.parse_args()

    print(f"Converting UMIE {args.config} dataset to VQA format...")
    vqa_items = create_umie_vqa_dataset(args.config, args.max_samples, args.output)
    print(f"Created {len(vqa_items)} VQA items in {args.output}")