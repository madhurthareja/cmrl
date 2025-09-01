# VQA-RAD Download Instructions

**Description:** Radiology Visual Question Answering Dataset

**Size:** 315 images, 3,515 QA pairs

**License:** CC BY 4.0

## Download Steps

1. Visit: https://osf.io/89kps/
2. Download VQA_RAD_Dataset.zip
3. Extract to ./medvlm_data/vqa_rad/
4. Structure should be:
   ```
   vqa_rad/
   ├── images/
   ├── trainset.json
   └── testset.json
   ```

## Citation

Paper: https://arxiv.org/abs/1811.02629

Please cite the original paper if you use this dataset.

## Integration with Benchmark

Once downloaded, the dataset will be automatically detected by:
```python
from medvlm_evaluation_suite import MedVLMBenchmarkSuite
# Dataset 'vqa_rad' will be available for benchmarking
```
