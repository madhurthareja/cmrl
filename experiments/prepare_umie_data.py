#!/usr/bin/env python3
"""
UMIE Dataset Preparation Script
Download and convert UMIE medical imaging datasets for VLM finetuning
"""

import os
import sys
import argparse

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from retrieval.umie_data_loader import create_umie_vqa_dataset, UMIEDataLoader
from training.vlm_config import UMIE_DATASETS

def main():
    parser = argparse.ArgumentParser(description="Prepare UMIE datasets for VLM training")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(UMIE_DATASETS.keys()),
                       help="UMIE dataset to download")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to download (None for all)")
    parser.add_argument("--output_dir", type=str, default="./data/medvlm_data",
                       help="Output directory for prepared data")
    parser.add_argument("--cache_dir", type=str, default="./data/umie_cache",
                       help="Cache directory for downloaded data")
    parser.add_argument("--list", action="store_true",
                       help="List available UMIE datasets")

    args = parser.parse_args()

    if args.list:
        print("Available UMIE Datasets:")
        print("=" * 50)
        for name, info in UMIE_DATASETS.items():
            print(f"\n{name}:")
            print(f"  Description: {info['description']}")
            print(f"  Domain: {info['domain']}")
            print(f"  Task: {info['task_type']}")
            print(f"  Labels: {', '.join(info['labels'])}")
            print(f"  Estimated samples: {info['estimated_samples']}")
            print(f"  Recommended model: {info['recommended_model']}")
        return

    # Get dataset info
    dataset_info = UMIE_DATASETS[args.dataset]
    print(f"Preparing UMIE dataset: {args.dataset}")
    print(f"Description: {dataset_info['description']}")
    print(f"Domain: {dataset_info['domain']}")
    print(f"Estimated samples: {dataset_info['estimated_samples']}")

    if args.max_samples:
        print(f"Limiting to {args.max_samples} samples")
    else:
        print("Downloading all available samples (may take time)")

    # Prepare output file
    output_file = os.path.join(args.output_dir, f"umie_{args.dataset}_annotations.json")

    # Create the VQA dataset
    try:
        print("\nDownloading and processing dataset...")
        vqa_items = create_umie_vqa_dataset(
            config_name=args.dataset,
            max_samples=args.max_samples,
            output_file=output_file
        )

        print(f"\n✓ Successfully prepared {len(vqa_items)} VQA samples")
        print(f"✓ Annotations saved to: {output_file}")

        # Show sample
        if vqa_items:
            sample = vqa_items[0]
            print("\nSample VQA item:")
            print(f"  Question: {sample.question}")
            print(f"  Answer: {sample.answer}")
            print(f"  Domain: {sample.domain}")
            print(f"  Image: {sample.image_path}")

        # Print next steps
        print("\nNext steps:")
        print(f"1. Review the prepared data in {output_file}")
        print(f"2. Check images in {args.cache_dir}/images/")
        print("3. Run VLM finetuning:")
        print(f"   python finetune_vlm.py --model {dataset_info['recommended_model']} --data_dir {args.output_dir}")

    except Exception as e:
        print(f"\n✗ Failed to prepare dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()