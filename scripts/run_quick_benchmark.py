"""Runner for a quick pipeline benchmark using mock LLM to test model outputs vs golden answers."""
import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from benchmark_pipeline import PipelineBenchmark
from benchmark_config import BENCHMARK_CONFIGS

if __name__ == '__main__':
    bench = PipelineBenchmark(use_mock_llm=True)
    config = BENCHMARK_CONFIGS['quick']
    queries = config['queries']

    metrics = bench.run_benchmark(test_queries=queries, mode='full', output_file='quick_benchmark_results.json')
    print('Quick benchmark finished. Results saved to quick_benchmark_results.json')
