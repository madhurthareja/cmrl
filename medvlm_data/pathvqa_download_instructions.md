# PathVQA Download Instructions

**Description:** Pathology Visual Question Answering Dataset

**Size:** 32,799 images, 234,775 QA pairs

**License:** MIT

## Download Steps

1. Visit: https://github.com/UCSD-AI4H/PathVQA
2. Follow repository instructions
3. Download images and QA pairs
4. Extract to ./medvlm_data/pathvqa/

## Citation

Paper: https://arxiv.org/abs/2003.10286

Please cite the original paper if you use this dataset.

## Integration with Benchmark

Once downloaded, the dataset will be automatically detected by:
```python
from medvlm_evaluation_suite import MedVLMBenchmarkSuite
# Dataset 'pathvqa' will be available for benchmarking
```
