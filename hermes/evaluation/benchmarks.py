"""Benchmark evaluation for LLMs (MMLU, GSM8K, etc.)."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from tqdm.auto import tqdm

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets not installed. Install with: pip install datasets")

from hermes.evaluation.evaluator import BenchmarkEvaluator
from hermes.evaluation.metrics import EvaluationMetrics


class MMLUBenchmark(BenchmarkEvaluator):
    """
    MMLU (Massive Multitask Language Understanding) benchmark.
    
    Tests knowledge across 57 subjects including STEM, humanities, and social sciences.
    """
    
    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
        "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
        "college_medicine", "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
        "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
        "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
    ]
    
    def __init__(
        self,
        subjects: Optional[List[str]] = None,
        split: str = "test",
        num_samples: Optional[int] = None,
        name: str = "mmlu",
    ) -> None:
        """
        Initialize MMLU benchmark.
        
        Args:
            subjects: List of subjects to evaluate (defaults to all)
            split: Dataset split ("test", "validation", "dev")
            num_samples: Maximum samples per subject (None = all)
            name: Benchmark name
        """
        super().__init__(name=name)
        
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets required. Install with: pip install datasets")
        
        self.subjects = subjects or self.SUBJECTS
        self.split = split
        self.num_samples = num_samples
        
        logger.info(f"MMLU benchmark initialized with {len(self.subjects)} subjects")
    
    def load_benchmark(self) -> List[Dict[str, Any]]:
        """Load MMLU benchmark data."""
        
        all_samples = []
        
        for subject in tqdm(self.subjects, desc="Loading subjects"):
            try:
                dataset = load_dataset(
                    "cais/mmlu",
                    subject,
                    split=self.split,
                    trust_remote_code=True,
                )
                
                # Format samples
                for idx, sample in enumerate(dataset):
                    if self.num_samples and idx >= self.num_samples:
                        break
                    
                    # MMLU format: question + 4 choices (A, B, C, D)
                    question = sample["question"]
                    choices = sample["choices"]
                    answer = sample["answer"]  # Index (0-3)
                    
                    all_samples.append({
                        "instruction": self._format_question(question, choices),
                        "choices": choices,
                        "correct_answer": answer,
                        "subject": subject,
                        "metadata": {
                            "subject": subject,
                            "benchmark": "mmlu",
                        },
                    })
                
            except Exception as e:
                logger.error(f"Failed to load subject {subject}: {e}")
        
        logger.info(f"Loaded {len(all_samples)} samples from MMLU")
        return all_samples
    
    def _format_question(self, question: str, choices: List[str]) -> str:
        """Format question with multiple choices."""
        
        formatted = f"{question}\n\n"
        for idx, choice in enumerate(choices):
            letter = chr(ord('A') + idx)
            formatted += f"{letter}. {choice}\n"
        
        formatted += "\nAnswer with only the letter (A, B, C, or D)."
        return formatted
    
    def extract_answer(self, response: str) -> str:
        """Extract letter answer from model response."""
        
        response = response.strip().upper()
        
        # Try to find A, B, C, or D
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response:
                # Prefer if it's at the start or in "Answer: X" format
                if response.startswith(letter) or f"ANSWER: {letter}" in response or f"ANSWER {letter}" in response:
                    return letter
        
        # Fallback: return first A-D found
        for char in response:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        return "A"  # Default fallback
    
    def score_answer(self, predicted: str, correct: int) -> bool:
        """Check if predicted answer matches correct answer."""
        
        predicted_letter = self.extract_answer(predicted)
        correct_letter = chr(ord('A') + correct)
        
        return predicted_letter == correct_letter


class GSM8KBenchmark(BenchmarkEvaluator):
    """
    GSM8K (Grade School Math 8K) benchmark.
    
    Tests mathematical reasoning with grade-school level problems.
    """
    
    def __init__(
        self,
        split: str = "test",
        num_samples: Optional[int] = None,
        name: str = "gsm8k",
    ) -> None:
        """
        Initialize GSM8K benchmark.
        
        Args:
            split: Dataset split
            num_samples: Maximum samples
            name: Benchmark name
        """
        super().__init__(name=name)
        
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets required. Install with: pip install datasets")
        
        self.split = split
        self.num_samples = num_samples
    
    def load_benchmark(self) -> List[Dict[str, Any]]:
        """Load GSM8K benchmark."""
        
        dataset = load_dataset("gsm8k", "main", split=self.split)
        
        samples = []
        for idx, sample in enumerate(dataset):
            if self.num_samples and idx >= self.num_samples:
                break
            
            # GSM8K format: question and answer with reasoning
            question = sample["question"]
            answer = sample["answer"]
            
            # Extract final numeric answer
            final_answer = self._extract_final_answer(answer)
            
            samples.append({
                "instruction": question + "\n\nProvide your final answer as: The answer is: [number]",
                "correct_answer": final_answer,
                "full_solution": answer,
                "metadata": {
                    "benchmark": "gsm8k",
                },
            })
        
        logger.info(f"Loaded {len(samples)} samples from GSM8K")
        return samples
    
    def _extract_final_answer(self, answer: str) -> str:
        """Extract final numeric answer from solution."""
        
        # GSM8K answers end with "#### [number]"
        if "####" in answer:
            return answer.split("####")[-1].strip()
        
        return answer.strip()
    
    def extract_answer(self, response: str) -> str:
        """Extract numeric answer from model response."""
        
        # Look for "The answer is: X" format
        if "the answer is:" in response.lower():
            parts = response.lower().split("the answer is:")
            if len(parts) > 1:
                answer = parts[-1].strip()
                # Extract number
                import re
                numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer)
                if numbers:
                    # Remove commas
                    return numbers[0].replace(',', '')
        
        # Fallback: extract any number
        import re
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', response)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def score_answer(self, predicted: str, correct: str) -> bool:
        """Check if numeric answers match."""
        
        predicted_num = self.extract_answer(predicted)
        
        try:
            return float(predicted_num) == float(correct)
        except (ValueError, TypeError):
            return predicted_num == correct


def run_benchmark(
    benchmark: BenchmarkEvaluator,
    generate_fn: callable,
    batch_size: int = 8,
    save_results: Optional[Path] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run a benchmark evaluation.
    
    Args:
        benchmark: Benchmark evaluator instance
        generate_fn: Function to generate answers (takes list of instructions)
        batch_size: Batch size for generation
        save_results: Path to save detailed results (optional)
        
    Returns:
        Tuple of (accuracy, detailed_results)
    """
    logger.info(f"Running {benchmark.name} benchmark")
    
    # Load benchmark data
    samples = benchmark.load_benchmark()
    
    # Generate answers
    instructions = [s["instruction"] for s in samples]
    
    logger.info(f"Generating answers for {len(instructions)} samples")
    all_answers = []
    
    for i in tqdm(range(0, len(instructions), batch_size), desc="Generating"):
        batch = instructions[i:i + batch_size]
        answers = generate_fn(batch)
        all_answers.extend(answers)
    
    # Evaluate
    logger.info("Evaluating answers")
    metrics_list = benchmark.evaluate_batch(
        instructions=instructions,
        answers=all_answers,
        references=[s.get("correct_answer") for s in samples],
    )
    
    # Compute accuracy
    correct = sum(1 for m in metrics_list if m.accuracy_score == 1.0)
    total = len(metrics_list)
    accuracy = correct / total if total > 0 else 0.0
    
    logger.info(f"{benchmark.name} accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Prepare detailed results
    results = []
    for sample, answer, metrics in zip(samples, all_answers, metrics_list):
        results.append({
            "instruction": sample["instruction"],
            "generated_answer": answer,
            "correct_answer": sample.get("correct_answer"),
            "is_correct": metrics.accuracy_score == 1.0,
            "metadata": sample.get("metadata", {}),
        })
    
    # Save if requested
    if save_results:
        save_results.parent.mkdir(parents=True, exist_ok=True)
        with open(save_results, "w") as f:
            json.dump({
                "benchmark": benchmark.name,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results,
            }, f, indent=2)
        logger.info(f"Results saved to {save_results}")
    
    return accuracy, results
