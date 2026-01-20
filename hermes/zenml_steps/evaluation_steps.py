"""
Evaluation Steps for ZenML Pipelines

This module contains steps for model evaluation.
"""

from typing import Dict, Any, List
from typing_extensions import Annotated

from loguru import logger
from zenml import get_step_context, step

from hermes.evaluation.evaluator import ModelEvaluator
from hermes.evaluation.benchmarks import BenchmarkRunner


@step
def evaluate_model(
    model_id: str,
    eval_dataset_id: str = None,
    metrics: List[str] = None,
    num_samples: int = 100,
) -> Annotated[Dict[str, Any], "evaluation_results"]:
    """
    Evaluate a language model.
    
    Args:
        model_id: Model identifier to evaluate
        eval_dataset_id: Optional evaluation dataset ID
        metrics: List of metrics to compute (e.g., ["g-eval", "accuracy"])
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating model: {model_id}")
    
    metrics = metrics or ["g-eval", "accuracy"]
    
    try:
        evaluator = ModelEvaluator(model_id=model_id)
        
        results = evaluator.evaluate(
            dataset_id=eval_dataset_id,
            metrics=metrics,
            num_samples=num_samples,
        )
        
        logger.success(f"Evaluation complete: {results.get('overall_score', 'N/A')}")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="evaluation_results",
            metadata={
                "model_id": model_id,
                "metrics": metrics,
                "num_samples": num_samples,
                **results,
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
        }


@step
def run_benchmarks(
    model_id: str,
    benchmarks: List[str] = None,
    num_samples: int = 100,
) -> Annotated[Dict[str, Dict[str, Any]], "benchmark_results"]:
    """
    Run standard benchmarks on a model.
    
    Args:
        model_id: Model identifier
        benchmarks: List of benchmarks (e.g., ["mmlu", "gsm8k"])
        num_samples: Number of samples per benchmark
        
    Returns:
        Dictionary mapping benchmark names to results
    """
    logger.info(f"Running benchmarks on model: {model_id}")
    
    benchmarks = benchmarks or ["mmlu", "gsm8k"]
    
    try:
        runner = BenchmarkRunner(model_id=model_id)
        
        all_results = {}
        for benchmark in benchmarks:
            logger.info(f"Running benchmark: {benchmark}")
            results = runner.run_benchmark(
                benchmark_name=benchmark,
                num_samples=num_samples,
            )
            all_results[benchmark] = results
            logger.info(f"{benchmark} score: {results.get('score', 'N/A')}")
        
        logger.success(f"All benchmarks complete")
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="benchmark_results",
            metadata={
                "model_id": model_id,
                "benchmarks": benchmarks,
                "num_samples": num_samples,
                "results": all_results,
            }
        )
        
        return all_results
        
    except Exception as e:
        logger.error(f"Benchmarks failed: {e}")
        return {"error": str(e)}
