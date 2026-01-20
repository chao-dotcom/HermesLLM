"""Model evaluation pipeline."""

from typing import Optional, List
from zenml import pipeline

from tasks.evaluation import (
    load_evaluation_dataset,
    generate_answers_vllm,
    evaluate_with_accuracy_style,
    save_evaluation_results,
    push_results_to_hub,
)


@pipeline(name="model_evaluation_pipeline")
def model_evaluation_pipeline(
    # Dataset parameters
    dataset_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    split: str = "test",
    num_samples: Optional[int] = None,
    
    # Generation parameters
    model_path: str = None,
    tensor_parallel_size: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    batch_size: int = 32,
    
    # Evaluation parameters
    openai_api_key: str = None,
    evaluation_model: str = "gpt-4o-mini",
    max_workers: int = 10,
    
    # Output parameters
    output_path: str = "data/evaluation_results.json",
    push_to_hub: bool = False,
    hub_dataset_name: Optional[str] = None,
    hub_token: Optional[str] = None,
) -> None:
    """
    End-to-end model evaluation pipeline.
    
    Steps:
    1. Load evaluation dataset
    2. Generate answers using vLLM
    3. Evaluate with accuracy & style criteria
    4. Save results
    5. (Optional) Push to HuggingFace Hub
    
    Args:
        dataset_path: Path to local dataset
        dataset_name: HuggingFace dataset name
        split: Dataset split
        num_samples: Max samples
        model_path: Path to model for evaluation
        tensor_parallel_size: Tensor parallelism for vLLM
        max_tokens: Max generation tokens
        temperature: Sampling temperature
        top_p: Nucleus sampling
        batch_size: Generation batch size
        openai_api_key: OpenAI API key for evaluation
        evaluation_model: OpenAI model for evaluation
        max_workers: Parallel workers for evaluation
        output_path: Path to save results
        push_to_hub: Whether to push results to Hub
        hub_dataset_name: HuggingFace dataset name for results
        hub_token: HuggingFace token
    """
    # Load dataset
    dataset = load_evaluation_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        split=split,
        num_samples=num_samples,
    )
    
    # Generate answers
    answers = generate_answers_vllm(
        instructions=dataset["instructions"],
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )
    
    # Evaluate
    results = evaluate_with_accuracy_style(
        instructions=dataset["instructions"],
        answers=answers,
        api_key=openai_api_key,
        model=evaluation_model,
        max_workers=max_workers,
    )
    
    # Save results
    result_path = save_evaluation_results(
        result=results,
        output_path=output_path,
    )
    
    # Push to hub if requested
    if push_to_hub and hub_dataset_name and hub_token:
        push_results_to_hub(
            result=results,
            dataset_name=hub_dataset_name,
            token=hub_token,
        )


@pipeline(name="benchmark_evaluation_pipeline")
def benchmark_evaluation_pipeline(
    # Benchmark parameters
    benchmark_name: str = "mmlu",
    subjects: Optional[List[str]] = None,
    split: str = "test",
    num_samples: Optional[int] = None,
    
    # Generation parameters
    model_path: str = None,
    tensor_parallel_size: int = 1,
    max_tokens: int = 128,
    temperature: float = 0.0,  # Deterministic for benchmarks
    batch_size: int = 32,
    
    # Output parameters
    output_path: str = "data/benchmark_results.json",
) -> None:
    """
    Benchmark evaluation pipeline (MMLU, GSM8K, etc.).
    
    Args:
        benchmark_name: Name of benchmark (mmlu, gsm8k)
        subjects: MMLU subjects (optional)
        split: Dataset split
        num_samples: Max samples per subject
        model_path: Model path
        tensor_parallel_size: Tensor parallelism
        max_tokens: Max tokens
        temperature: Temperature (0.0 for deterministic)
        batch_size: Batch size
        output_path: Results path
    """
    from hermes.evaluation.benchmarks import run_benchmark, MMLUBenchmark, GSM8KBenchmark
    from tasks.evaluation.steps import generate_answers_vllm
    from pathlib import Path
    
    # Initialize benchmark
    if benchmark_name.lower() == "mmlu":
        benchmark = MMLUBenchmark(
            subjects=subjects,
            split=split,
            num_samples=num_samples,
        )
    elif benchmark_name.lower() == "gsm8k":
        benchmark = GSM8KBenchmark(
            split=split,
            num_samples=num_samples,
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    # Run benchmark
    accuracy, results = run_benchmark(
        benchmark=benchmark,
        generate_fn=lambda prompts: generate_answers_vllm(
            instructions=prompts,
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_tokens=max_tokens,
            temperature=temperature,
            batch_size=batch_size,
        ),
        save_results=Path(output_path),
    )


if __name__ == "__main__":
    # Example: Run evaluation pipeline
    model_evaluation_pipeline(
        dataset_name="your-dataset",
        model_path="path/to/model",
        openai_api_key="your-openai-key",
        output_path="data/evaluation_results.json",
    )
