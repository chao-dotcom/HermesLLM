"""
Comprehensive examples for model evaluation.

This module demonstrates how to:
1. Evaluate models using G-Eval methodology
2. Run benchmark evaluations (MMLU, GSM8K)
3. Use ZenML pipelines for evaluation
4. Generate and evaluate answers with vLLM
"""

import os
from pathlib import Path
from typing import List

from loguru import logger

# Example 1: Basic G-Eval Evaluation
def example_g_eval_evaluation():
    """Evaluate model outputs using G-Eval."""
    
    from hermes.evaluation import GEval
    
    # Initialize evaluator
    evaluator = GEval(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        criteria=["relevance", "coherence", "fluency", "consistency"],
        max_workers=10,
    )
    
    # Sample data
    instructions = [
        "Explain how transformers work in machine learning.",
        "What are the benefits of LoRA fine-tuning?",
        "Describe the difference between supervised and unsupervised learning.",
    ]
    
    answers = [
        "Transformers use self-attention mechanisms to process sequential data...",
        "LoRA (Low-Rank Adaptation) reduces the number of trainable parameters...",
        "Supervised learning uses labeled data, while unsupervised learning...",
    ]
    
    # Evaluate
    logger.info("Running G-Eval evaluation...")
    metrics_list = evaluator.evaluate_batch(
        instructions=instructions,
        answers=answers,
    )
    
    # Display results
    for inst, metrics in zip(instructions, metrics_list):
        print(f"\nInstruction: {inst[:50]}...")
        print(f"  Relevance: {metrics.relevance_score:.2f}")
        print(f"  Coherence: {metrics.coherence_score:.2f}")
        print(f"  Fluency: {metrics.fluency_score:.2f}")
        print(f"  Consistency: {metrics.consistency_score:.2f}")
        print(f"  Overall: {metrics.overall_score:.2f}")


# Example 2: Accuracy & Style Evaluation (Blog/Social Content)
def example_accuracy_style_evaluation():
    """Evaluate for accuracy and style (blog/social media)."""
    
    from hermes.evaluation import AccuracyStyleEvaluator
    
    # Initialize evaluator
    evaluator = AccuracyStyleEvaluator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
        max_workers=10,
    )
    
    # Sample data (blog-style content)
    instructions = [
        "Write about the latest developments in GPT-4.",
        "Explain quantum computing for a general audience.",
    ]
    
    answers = [
        "GPT-4 represents a significant leap in AI capabilities. It demonstrates improved reasoning and can handle more complex tasks than its predecessors.",
        "Quantum computing leverages quantum mechanics principles like superposition and entanglement to perform calculations exponentially faster than classical computers.",
    ]
    
    # Evaluate
    logger.info("Running Accuracy-Style evaluation...")
    metrics_list = evaluator.evaluate_batch(
        instructions=instructions,
        answers=answers,
    )
    
    # Display results
    for inst, metrics in zip(instructions, metrics_list):
        print(f"\nInstruction: {inst[:50]}...")
        print(f"  Accuracy: {metrics.accuracy_score:.2f} - {metrics.accuracy_analysis}")
        print(f"  Style: {metrics.style_score:.2f} - {metrics.style_analysis}")
        print(f"  Overall: {metrics.overall_score:.2f}")


# Example 3: MMLU Benchmark Evaluation
def example_mmlu_evaluation():
    """Run MMLU benchmark evaluation."""
    
    from hermes.evaluation import MMLUBenchmark, run_benchmark
    from vllm import LLM, SamplingParams
    
    # Initialize benchmark
    benchmark = MMLUBenchmark(
        subjects=["machine_learning", "computer_security"],  # Subset for demo
        split="test",
        num_samples=10,  # Small sample for demo
    )
    
    # Initialize model
    model_path = "path/to/your/model"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for benchmarks
        max_tokens=128,
    )
    
    # Define generation function
    def generate_fn(prompts: List[str]) -> List[str]:
        outputs = llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    # Run benchmark
    logger.info("Running MMLU benchmark...")
    accuracy, results = run_benchmark(
        benchmark=benchmark,
        generate_fn=generate_fn,
        batch_size=8,
        save_results=Path("data/mmlu_results.json"),
    )
    
    print(f"\nMMLU Accuracy: {accuracy:.2%}")


# Example 4: GSM8K Benchmark Evaluation
def example_gsm8k_evaluation():
    """Run GSM8K (math reasoning) benchmark."""
    
    from hermes.evaluation import GSM8KBenchmark, run_benchmark
    from vllm import LLM, SamplingParams
    
    # Initialize benchmark
    benchmark = GSM8KBenchmark(
        split="test",
        num_samples=100,
    )
    
    # Initialize model
    model_path = "path/to/your/model"
    llm = LLM(model=model_path)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )
    
    # Generate function
    def generate_fn(prompts: List[str]) -> List[str]:
        outputs = llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    # Run benchmark
    logger.info("Running GSM8K benchmark...")
    accuracy, results = run_benchmark(
        benchmark=benchmark,
        generate_fn=generate_fn,
        save_results=Path("data/gsm8k_results.json"),
    )
    
    print(f"\nGSM8K Accuracy: {accuracy:.2%}")


# Example 5: vLLM-based Answer Generation + Evaluation
def example_end_to_end_evaluation():
    """Complete evaluation pipeline with vLLM generation."""
    
    from vllm import LLM, SamplingParams
    from hermes.evaluation import AccuracyStyleEvaluator, EvaluationResult
    
    # Load model
    model_path = "path/to/your/fine-tuned-model"
    logger.info(f"Loading model: {model_path}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    
    # Prepare test data
    instructions = [
        "Explain the difference between LoRA and full fine-tuning.",
        "What is the benefit of using 4-bit quantization?",
        "How does DPO improve model alignment?",
    ]
    
    # Generate answers
    logger.info("Generating answers...")
    prompts = [f"<|user|>\n{inst}<|end|>\n<|assistant|>\n" for inst in instructions]
    
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=512,
    )
    
    outputs = llm.generate(prompts, sampling_params)
    answers = [output.outputs[0].text for output in outputs]
    
    # Evaluate
    logger.info("Evaluating answers...")
    evaluator = AccuracyStyleEvaluator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    
    metrics_list = evaluator.evaluate_batch(
        instructions=instructions,
        answers=answers,
    )
    
    # Aggregate results
    result = EvaluationResult(
        metrics=metrics_list,
        evaluator="accuracy_style",
    )
    
    # Display summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    summary = result.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value:.3f}")
    
    # Save results
    output_path = Path("data/evaluation_results.json")
    import json
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "details": [
                {
                    "instruction": inst,
                    "answer": ans,
                    "accuracy": m.accuracy_score,
                    "style": m.style_score,
                    "overall": m.overall_score,
                }
                for inst, ans, m in zip(instructions, answers, metrics_list)
            ]
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


# Example 6: ZenML Pipeline Execution
def example_zenml_pipeline():
    """Run evaluation using ZenML pipeline."""
    
    from workflows.pipelines.evaluation import model_evaluation_pipeline
    
    # Configure pipeline
    model_evaluation_pipeline(
        # Dataset
        dataset_path="data/test_dataset.json",
        num_samples=100,
        
        # Model
        model_path="path/to/your/model",
        tensor_parallel_size=1,
        max_tokens=512,
        temperature=0.8,
        
        # Evaluation
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        evaluation_model="gpt-4o-mini",
        max_workers=10,
        
        # Output
        output_path="data/evaluation_results.json",
        push_to_hub=False,
    )


# Example 7: Benchmark Pipeline with ZenML
def example_benchmark_pipeline():
    """Run MMLU benchmark using ZenML."""
    
    from workflows.pipelines.evaluation import benchmark_evaluation_pipeline
    
    # Run MMLU
    benchmark_evaluation_pipeline(
        benchmark_name="mmlu",
        subjects=["machine_learning", "computer_security", "high_school_mathematics"],
        num_samples=50,
        model_path="path/to/your/model",
        output_path="data/mmlu_results.json",
    )


# Example 8: Batch Evaluation with Progress Tracking
def example_batch_evaluation_with_progress():
    """Evaluate large dataset with progress tracking."""
    
    import json
    from tqdm import tqdm
    from hermes.evaluation import AccuracyStyleEvaluator
    
    # Load large dataset
    with open("data/large_test_set.json") as f:
        dataset = json.load(f)
    
    instructions = [item["instruction"] for item in dataset]
    answers = [item["generated_answer"] for item in dataset]
    
    # Initialize evaluator
    evaluator = AccuracyStyleEvaluator(
        api_key=os.getenv("OPENAI_API_KEY"),
        max_workers=20,  # Increased parallelism
    )
    
    # Evaluate in chunks with progress bar
    batch_size = 100
    all_metrics = []
    
    for i in tqdm(range(0, len(instructions), batch_size), desc="Evaluating"):
        batch_inst = instructions[i:i + batch_size]
        batch_ans = answers[i:i + batch_size]
        
        metrics = evaluator.evaluate_batch(batch_inst, batch_ans)
        all_metrics.extend(metrics)
        
        # Save checkpoint
        if (i + batch_size) % 500 == 0:
            checkpoint_path = f"data/checkpoints/eval_checkpoint_{i}.json"
            Path(checkpoint_path).parent.mkdir(exist_ok=True, parents=True)
            with open(checkpoint_path, "w") as f:
                json.dump([m.dict() for m in all_metrics], f)
    
    # Final summary
    from hermes.evaluation import EvaluationResult
    result = EvaluationResult(metrics=all_metrics, evaluator="accuracy_style")
    
    print("\nFinal Results:")
    print(f"  Samples: {result.num_samples}")
    print(f"  Avg Accuracy: {result.average_accuracy:.3f}")
    print(f"  Avg Style: {result.average_style:.3f}")
    print(f"  Avg Overall: {result.average_overall_score:.3f}")


def main():
    """Run all examples."""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION EXAMPLES")
    print("="*60)
    
    examples = [
        ("G-Eval Evaluation", example_g_eval_evaluation),
        ("Accuracy & Style Evaluation", example_accuracy_style_evaluation),
        ("MMLU Benchmark", example_mmlu_evaluation),
        ("GSM8K Benchmark", example_gsm8k_evaluation),
        ("End-to-End Evaluation", example_end_to_end_evaluation),
        ("ZenML Pipeline", example_zenml_pipeline),
        ("Benchmark Pipeline", example_benchmark_pipeline),
        ("Batch Evaluation", example_batch_evaluation_with_progress),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{i}. {name}")
        print("-" * 60)
        
        try:
            func()
            print("✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")


if __name__ == "__main__":
    # Run specific example
    example_accuracy_style_evaluation()
    
    # Or run all examples
    # main()
