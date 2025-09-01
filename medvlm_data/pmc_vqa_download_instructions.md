# PMC-VQA Download Instructions

**Description:** PubMed Central Medical VQA

**Size:** 227,000+ image-text pairs

**License:** Apache 2.0

## Download Steps

1. Visit: https://github.com/xiaoman-zhang/PMC-VQA
2. Clone repository
3. Follow data preparation instructions
4. Extract to ./medvlm_data/pmc_vqa/

## Citation

Paper: https://arxiv.org/abs/2305.10415

Please cite the original paper if you use this dataset.

## Integration with Benchmark

Once downloaded, the dataset will be automatically detected by:
```python
from medvlm_evaluation_suite import MedVLMBenchmarkSuite
# Dataset 'pmc_vqa' will be available for benchmarking
```
