# SLAKE Download Instructions

**Description:** Bilingual Medical VQA Dataset

**Size:** 642 images, 14,028 QA pairs

**License:** CC BY-NC 4.0

## Download Steps

1. Visit: https://www.med-vqa.com/slake/
2. Register and download dataset
3. Extract to ./medvlm_data/slake/

## Citation

Paper: https://arxiv.org/abs/2102.09542

Please cite the original paper if you use this dataset.

## Integration with Benchmark

Once downloaded, the dataset will be automatically detected by:
```python
from medvlm_evaluation_suite import MedVLMBenchmarkSuite
# Dataset 'slake' will be available for benchmarking
```
