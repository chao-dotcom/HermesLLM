"""Base evaluator classes."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from loguru import logger

try:
    from datasets import Dataset, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets not installed. Install with: pip install datasets")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not installed. Install with: pip install vllm")

from hermes.evaluation.metrics import EvaluationMetrics, EvaluationResult


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, name: str = "base") -> None:
        """
        Initialize evaluator.
        
        Args:
            name: Evaluator name
        """
        self.name = name
    
    @abstractmethod
    def evaluate_sample(
        self,
        instruction: str,
        answer: str,
        reference: Optional[str] = None,
        **kwargs,
    ) -> EvaluationMetrics:
        """
        Evaluate a single sample.
        
        Args:
            instruction: Input instruction/prompt
            answer: Model-generated answer
            reference: Reference/ground truth answer (optional)
            **kwargs: Additional parameters
            
        Returns:
            Evaluation metrics
        """
        pass
    
    def evaluate_batch(
        self,
        instructions: List[str],
        answers: List[str],
        references: Optional[List[str]] = None,
        **kwargs,
    ) -> List[EvaluationMetrics]:
        """
        Evaluate a batch of samples.
        
        Args:
            instructions: List of instructions
            answers: List of model answers
            references: List of reference answers (optional)
            **kwargs: Additional parameters
            
        Returns:
            List of evaluation metrics
        """
        metrics_list = []
        
        for i, (instruction, answer) in enumerate(zip(instructions, answers)):
            reference = references[i] if references else None
            metrics = self.evaluate_sample(
                instruction=instruction,
                answer=answer,
                reference=reference,
                **kwargs,
            )
            metrics_list.append(metrics)
        
        return metrics_list
    
    def evaluate_dataset(
        self,
        dataset,
        instruction_key: str = "instruction",
        answer_key: str = "answer",
        reference_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Evaluate a complete dataset.
        
        Args:
            dataset: Dataset or list of samples
            instruction_key: Key for instruction field
            answer_key: Key for answer field
            reference_key: Key for reference field (optional)
            **kwargs: Additional parameters
            
        Returns:
            Evaluation result
        """
        logger.info(f"Evaluating dataset with {len(dataset)} samples")
        
        start_time = time.time()
        
        # Extract fields
        try:
            from datasets import Dataset
            is_hf_dataset = isinstance(dataset, Dataset)
        except ImportError:
            is_hf_dataset = False
        
        if is_hf_dataset:
            instructions = dataset[instruction_key]
            answers = dataset[answer_key]
            references = dataset[reference_key] if reference_key and reference_key in dataset.column_names else None
        else:
            instructions = [s[instruction_key] for s in dataset]
            answers = [s[answer_key] for s in dataset]
            references = [s.get(reference_key) for s in dataset] if reference_key else None
        
        # Evaluate
        sample_metrics = self.evaluate_batch(
            instructions=instructions,
            answers=answers,
            references=references,
            **kwargs,
        )
        
        # Create result
        result = EvaluationResult(
            metrics=sample_metrics,
            evaluator=self.name,
        )
        
        logger.info(f"Evaluation complete in {time.time() - start_time:.2f}s")
        logger.info(f"Average overall score: {result.average_overall_score:.3f}")
        
        return result


class LLMEvaluator(BaseEvaluator):
    """Evaluator using LLM (vLLM or transformers)."""
    
    def __init__(
        self,
        model_id: str,
        max_model_len: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        name: str = "llm_evaluator",
    ) -> None:
        """
        Initialize LLM evaluator.
        
        Args:
            model_id: Model identifier (HuggingFace or local path)
            max_model_len: Maximum model context length
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            name: Evaluator name
        """
        super().__init__(name=name)
        
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM required. Install with: pip install vllm")
        
        self.model_id = model_id
        self.max_tokens = max_tokens
        
        logger.info(f"Loading model: {model_id}")
        self.llm = LLM(model=model_id, max_model_len=max_model_len)
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        logger.info(f"LLM evaluator initialized: {model_id}")
    
    def generate_answers(
        self,
        prompts: List[str],
        format_fn: Optional[callable] = None,
    ) -> List[str]:
        """
        Generate answers for prompts.
        
        Args:
            prompts: List of prompts
            format_fn: Optional function to format prompts
            
        Returns:
            List of generated answers
        """
        if format_fn:
            prompts = [format_fn(p) for p in prompts]
        
        outputs = self.llm.generate(prompts, self.sampling_params)
        answers = [output.outputs[0].text for output in outputs]
        
        return answers
    
    def evaluate_sample(
        self,
        instruction: str,
        answer: str,
        reference: Optional[str] = None,
        **kwargs,
    ) -> EvaluationMetrics:
        \"\"\"
        Evaluate using LLM (not typical - usually used for generation).
        This is a placeholder.
        \"\"\"
        # For LLM evaluator, we typically just generate answers
        # Actual evaluation would use another evaluator
        return EvaluationMetrics(
            evaluator=self.name,
            overall_score=None,  # No automatic scoring
        )


class BenchmarkEvaluator(BaseEvaluator):
    \"\"\"Evaluator for standard benchmarks (MMLU, HellaSwag, etc.).\"\"\"
    
    def __init__(
        self,
        benchmark_name: str,
        model_id: Optional[str] = None,
        name: str = \"benchmark_evaluator\",
    ) -> None:
        \"\"\"
        Initialize benchmark evaluator.
        
        Args:
            benchmark_name: Name of benchmark (mmlu, hellaswag, etc.)
            model_id: Model to evaluate (optional)
            name: Evaluator name
        \"\"\"
        super().__init__(name=name)
        self.benchmark_name = benchmark_name
        self.model_id = model_id
    
    def evaluate_sample(
        self,
        instruction: str,
        answer: str,
        reference: Optional[str] = None,
        **kwargs,
    ) -> EvaluationMetrics:
        \"\"\"Evaluate single sample (typically for multiple choice).\"\"\"
        # Extract choice from answer
        predicted_choice = self._extract_choice(answer)
        correct_choice = reference
        
        # Exact match
        exact_match = 1.0 if predicted_choice == correct_choice else 0.0
        
        return EvaluationMetrics(
            exact_match=exact_match,
            accuracy_score=exact_match,
            evaluator=self.name,
            overall_score=exact_match,
        )
    
    def _extract_choice(self, answer: str) -> str:
        \"\"\"Extract choice (A, B, C, D) from answer.\"\"\"
        answer = answer.strip().upper()
        
        # Check for direct choice
        for choice in [\"A\", \"B\", \"C\", \"D\", \"E\"]:
            if answer.startswith(choice):
                return choice
        
        # Check for choice in parentheses
        for choice in [\"A\", \"B\", \"C\", \"D\", \"E\"]:
            if f\"({choice})\" in answer or f\"[{choice}]\" in answer:
                return choice
        
        return answer[0] if answer else \"\"
