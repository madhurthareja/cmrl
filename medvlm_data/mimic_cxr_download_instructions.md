# MIMIC-CXR Download Instructions

**Description:** Chest X-ray Database with Reports

**Size:** 377,110 images, 227,835 reports

**License:** PhysioNet Credentialed Health Data License

## Download Steps

1. Visit: https://physionet.org/content/mimic-cxr/2.0.0/
2. Complete credentialing process
3. Download dataset (Large: ~5TB)
4. Extract to ./medvlm_data/mimic_cxr/

## Citation

Paper: https://arxiv.org/abs/1901.07042

Please cite the original paper if you use this dataset.

## Integration with Benchmark

Once downloaded, the dataset will be automatically detected by:
```python
from medvlm_evaluation_suite import MedVLMBenchmarkSuite
# Dataset 'mimic_cxr' will be available for benchmarking
```
