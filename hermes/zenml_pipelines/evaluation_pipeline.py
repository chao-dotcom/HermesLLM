"""
Evaluation Pipeline

ZenML pipeline for model evaluation and benchmarking.
"""

from typing import List

from zenml import pipeline

from hermes.zenml_steps.evaluation_steps import evaluate_model, run_benchmarks


@pipeline(name="model_evaluation_pipeline")
def model_evaluation_pipeline(
    model_id: str,
    eval_dataset_id: str = None,
    metrics: List[str] = None,
    benchmarks: List[str] = None,
    num_eval_samples: int = 100,
    num_benchmark_samples: int = 100,
    run_benchmarks_flag: bool = True,
) -> str:
    """
    Pipeline for comprehensive model evaluation.
    
    This pipeline:
    1. Evaluates model on custom metrics
    2. Optionally runs standard benchmarks
    
    Args:
        model_id: Model identifier to evaluate
        eval_dataset_id: Evaluation dataset ID
        metrics: List of metrics to compute
        benchmarks: List of benchmarks to run
        num_eval_samples: Number of samples for evaluation
        num_benchmark_samples: Number of samples per benchmark
        run_benchmarks_flag: Whether to run benchmarks
        
    Returns:
        Step invocation ID
    """
    # Step 1: Evaluate with custom metrics
    eval_results = evaluate_model(
        model_id=model_id,
        eval_dataset_id=eval_dataset_id,
        metrics=metrics,
        num_samples=num_eval_samples,
    )
    
    # Step 2: Optionally run benchmarks
    if run_benchmarks_flag:
        benchmark_results = run_benchmarks(
            model_id=model_id,
            benchmarks=benchmarks,
            num_samples=num_benchmark_samples,
        )
        return benchmark_results.invocation_id
    
    return eval_results.invocation_id
