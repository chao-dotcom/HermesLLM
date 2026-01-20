"""ZenML steps for model evaluation."""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from zenml import step
from loguru import logger

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from hermes.evaluation import (
    EvaluationMetrics,
    EvaluationResult,
    OPENAI_AVAILABLE,
)

if OPENAI_AVAILABLE:
    from hermes.evaluation import GEval, AccuracyStyleEvaluator


@step
def load_evaluation_dataset(
    dataset_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    split: str = "test",
    num_samples: Optional[int] = None,
) -> Dict[str, List[str]]:
    """
    Load evaluation dataset.
    
    Args:
        dataset_path: Path to local dataset file (JSON)
        dataset_name: HuggingFace dataset name
        split: Dataset split
        num_samples: Maximum samples to load
        
    Returns:
        Dictionary with instructions and references
    """
    if dataset_path:
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path) as f:
            data = json.load(f)
        
        instructions = [item["instruction"] for item in data]
        references = [item.get("output", "") for item in data]
        
    elif dataset_name:
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets required. Install with: pip install datasets")
        
        logger.info(f"Loading dataset: {dataset_name}")
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        instructions = [item["instruction"] for item in dataset]
        references = [item.get("output", "") for item in dataset]
    
    else:
        raise ValueError("Either dataset_path or dataset_name must be provided")
    
    logger.info(f"Loaded {len(instructions)} evaluation samples")
    
    return {
        "instructions": instructions,
        "references": references,
    }


@step
def generate_answers_vllm(
    instructions: List[str],
    model_path: str,
    tensor_parallel_size: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    batch_size: int = 32,
) -> List[str]:
    """
    Generate answers using vLLM.
    
    Args:
        instructions: List of instructions
        model_path: Path to model
        tensor_parallel_size: Tensor parallelism
        max_tokens: Maximum generation tokens
        temperature: Sampling temperature
        top_p: Nucleus sampling
        batch_size: Batch size
        
    Returns:
        Generated answers
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM required. Install with: pip install vllm")
    
    logger.info(f"Loading model: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    logger.info(f"Generating answers for {len(instructions)} samples")
    
    # Format prompts
    prompts = [f"<|user|>\n{inst}<|end|>\n<|assistant|>\n" for inst in instructions]
    
    # Generate in batches
    all_answers = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        answers = [output.outputs[0].text for output in outputs]
        all_answers.extend(answers)
        
        logger.info(f"Generated {len(all_answers)}/{len(prompts)} answers")
    
    return all_answers


@step
def evaluate_with_g_eval(
    instructions: List[str],
    answers: List[str],
    references: Optional[List[str]],
    api_key: str,
    model: str = "gpt-4o-mini",
    criteria: Optional[List[str]] = None,
    max_workers: int = 10,
) -> EvaluationResult:
    """
    Evaluate answers using G-Eval.
    
    Args:
        instructions: Instructions
        answers: Generated answers
        references: Reference answers (optional)
        api_key: OpenAI API key
        model: OpenAI model
        criteria: Evaluation criteria
        max_workers: Parallel workers
        
    Returns:
        Aggregated evaluation results
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI required. Install with: pip install openai")
    
    logger.info("Initializing G-Eval evaluator")
    evaluator = GEval(
        api_key=api_key,
        model=model,
        criteria=criteria,
        max_workers=max_workers,
    )
    
    logger.info(f"Evaluating {len(instructions)} samples")
    metrics_list = evaluator.evaluate_batch(
        instructions=instructions,
        answers=answers,
        references=references,
    )
    
    # Aggregate results
    result = EvaluationResult(
        metrics=metrics_list,
        evaluator="g_eval",
    )
    
    logger.info(f"Evaluation complete. Average score: {result.average_overall_score:.3f}")
    
    return result


@step
def evaluate_with_accuracy_style(
    instructions: List[str],
    answers: List[str],
    api_key: str,
    model: str = "gpt-4o-mini",
    max_workers: int = 10,
) -> EvaluationResult:
    """
    Evaluate answers for accuracy and style (blog/social media content).
    
    Args:
        instructions: Instructions
        answers: Generated answers
        api_key: OpenAI API key
        model: OpenAI model
        max_workers: Parallel workers
        
    Returns:
        Aggregated evaluation results
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI required. Install with: pip install openai")
    
    logger.info("Initializing Accuracy-Style evaluator")
    evaluator = AccuracyStyleEvaluator(
        api_key=api_key,
        model=model,
        max_workers=max_workers,
    )
    
    logger.info(f"Evaluating {len(instructions)} samples")
    metrics_list = evaluator.evaluate_batch(
        instructions=instructions,
        answers=answers,
    )
    
    # Aggregate results
    result = EvaluationResult(
        metrics=metrics_list,
        evaluator="accuracy_style",
    )
    
    logger.info(f"Evaluation complete. Average score: {result.average_overall_score:.3f}")
    logger.info(f"  Accuracy: {result.average_accuracy:.3f}")
    logger.info(f"  Style: {result.average_style:.3f}")
    
    return result


@step
def save_evaluation_results(
    result: EvaluationResult,
    output_path: str,
) -> str:
    """
    Save evaluation results to file.
    
    Args:
        result: Evaluation results
        output_path: Output file path
        
    Returns:
        Path to saved file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    data = {
        "evaluator": result.evaluator,
        "num_samples": result.num_samples,
        "summary": result.get_summary(),
        "metrics": [
            {
                "accuracy_score": m.accuracy_score,
                "style_score": m.style_score,
                "relevance_score": m.relevance_score,
                "coherence_score": m.coherence_score,
                "fluency_score": m.fluency_score,
                "consistency_score": m.consistency_score,
                "overall_score": m.overall_score,
            }
            for m in result.metrics
        ],
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    return str(output_file)


@step
def push_results_to_hub(
    result: EvaluationResult,
    dataset_name: str,
    token: str,
    private: bool = False,
) -> str:
    """
    Push evaluation results to HuggingFace Hub.
    
    Args:
        result: Evaluation results
        dataset_name: HuggingFace dataset name
        token: HuggingFace token
        private: Make dataset private
        
    Returns:
        Dataset URL
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets required. Install with: pip install datasets")
    
    from datasets import Dataset
    
    logger.info(f"Pushing results to HuggingFace Hub: {dataset_name}")
    
    # Prepare data
    data = {
        "accuracy_score": [m.accuracy_score for m in result.metrics],
        "style_score": [m.style_score for m in result.metrics],
        "relevance_score": [m.relevance_score for m in result.metrics],
        "coherence_score": [m.coherence_score for m in result.metrics],
        "fluency_score": [m.fluency_score for m in result.metrics],
        "consistency_score": [m.consistency_score for m in result.metrics],
        "overall_score": [m.overall_score for m in result.metrics],
    }
    
    # Create dataset
    dataset = Dataset.from_dict(data)
    
    # Push to hub
    dataset.push_to_hub(
        dataset_name,
        token=token,
        private=private,
    )
    
    url = f"https://huggingface.co/datasets/{dataset_name}"
    logger.info(f"Results pushed to: {url}")
    
    return url
