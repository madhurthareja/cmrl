# Experiments

Finetuning, evaluation, and benchmarking utilities have moved here to keep the runtime application lightweight. Each script reuses the shared curriculum core and can be run independently of the production API.

- `benchmark_config.py`
- `benchmark_pipeline.py`
- `evaluate_vlm.py`
- `finetune_vlm.py`
- `prepare_umie_data.py`
- `run_quick_benchmark.py`

> Tip: activate the project virtual environment and run scripts with `python -m experiments.finetune_vlm` style invocations so relative imports resolve cleanly.
