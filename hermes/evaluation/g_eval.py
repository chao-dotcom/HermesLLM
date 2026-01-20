"""G-Eval: LLM-based evaluation using GPT models."""

import json
import concurrent.futures
from typing import List, Dict, Any, Optional
from loguru import logger
from tqdm.auto import tqdm

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Install with: pip install openai")

from hermes.evaluation.evaluator import BaseEvaluator
from hermes.evaluation.metrics import EvaluationMetrics


class GEval(BaseEvaluator):
    """
    G-Eval: LLM-based evaluation using GPT models.
    
    Based on the G-Eval paper: https://arxiv.org/abs/2303.16634
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        criteria: Optional[List[str]] = None,
        max_workers: int = 10,
        name: str = "g_eval",
    ) -> None:
        """
        Initialize G-Eval evaluator.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            criteria: List of evaluation criteria
            max_workers: Number of parallel workers
            name: Evaluator name
        """
        super().__init__(name=name)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        
        # Default criteria
        self.criteria = criteria or [
            "relevance",
            "coherence",
            "fluency",
            "consistency",
        ]
        
        logger.info(f"G-Eval initialized with model: {model}")
        logger.info(f"Evaluation criteria: {', '.join(self.criteria)}")
    
    def _build_prompt(
        self,
        criterion: str,
        instruction: str,
        answer: str,
        reference: Optional[str] = None,
    ) -> str:
        """Build evaluation prompt for a criterion."""
        
        prompts = {
            "relevance": f"""You will be given an instruction and an answer. Your task is to rate how relevant the answer is to the instruction on a scale of 1 to 5.

Instruction: {instruction}

Answer: {answer}

Evaluation criteria:
1 (Poor): The answer is not relevant to the instruction
2 (Fair): The answer addresses the instruction minimally
3 (Good): The answer is mostly relevant with minor irrelevant parts
4 (Very Good): The answer is highly relevant with very minor issues
5 (Excellent): The answer is perfectly relevant to the instruction

Provide your evaluation in JSON format:
{{
    "score": <1-5>,
    "analysis": "<brief explanation>"
}}""",
            
            "coherence": f"""You will be given an instruction and an answer. Your task is to rate how coherent and well-organized the answer is on a scale of 1 to 5.

Instruction: {instruction}

Answer: {answer}

Evaluation criteria:
1 (Poor): The answer is incoherent and poorly organized
2 (Fair): The answer has some coherence but lacks organization
3 (Good): The answer is mostly coherent with reasonable organization
4 (Very Good): The answer is highly coherent and well-organized
5 (Excellent): The answer is perfectly coherent with excellent structure

Provide your evaluation in JSON format:
{{
    "score": <1-5>,
    "analysis": "<brief explanation>"
}}""",
            
            "fluency": f"""You will be given an answer. Your task is to rate how fluent and natural the language is on a scale of 1 to 5.

Answer: {answer}

Evaluation criteria:
1 (Poor): The text has major grammatical errors and is difficult to read
2 (Fair): The text has several grammatical issues affecting readability
3 (Good): The text is mostly fluent with minor issues
4 (Very Good): The text is highly fluent with very minor issues
5 (Excellent): The text is perfectly fluent and natural

Provide your evaluation in JSON format:
{{
    "score": <1-5>,
    "analysis": "<brief explanation>"
}}""",
            
            "consistency": f"""You will be given an instruction and an answer. Your task is to rate how consistent the answer is internally on a scale of 1 to 5.

Instruction: {instruction}

Answer: {answer}

Evaluation criteria:
1 (Poor): The answer has major contradictions
2 (Fair): The answer has several inconsistencies
3 (Good): The answer is mostly consistent with minor issues
4 (Very Good): The answer is highly consistent
5 (Excellent): The answer is perfectly consistent throughout

Provide your evaluation in JSON format:
{{
    "score": <1-5>,
    "analysis": "<brief explanation>"
}}""",
            
            "accuracy": f"""You will be given an instruction and an answer. Your task is to rate how factually accurate the answer is on a scale of 1 to 5.

Instruction: {instruction}

Answer: {answer}
{f'Reference answer: {reference}' if reference else ''}

Evaluation criteria:
1 (Poor): Contains major factual errors
2 (Fair): Contains several factual inaccuracies
3 (Good): Mostly accurate with minor errors
4 (Very Good): Highly accurate with very minor issues
5 (Excellent): Completely accurate

Provide your evaluation in JSON format:
{{
    "score": <1-5>,
    "analysis": "<brief explanation>"
}}""",
        }
        
        return prompts.get(criterion, prompts["relevance"])
    
    def _evaluate_criterion(
        self,
        criterion: str,
        instruction: str,
        answer: str,
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single criterion using GPT."""
        
        prompt = self._build_prompt(criterion, instruction, answer, reference)
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide your evaluation in JSON format with a score (1-5) and a brief analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.0,  # Deterministic evaluation
            )
            
            result = json.loads(completion.choices[0].message.content)
            return {
                "score": result.get("score", 0),
                "analysis": result.get("analysis", ""),
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate {criterion}: {e}")
            return {"score": 0, "analysis": str(e)}
    
    def evaluate_sample(
        self,
        instruction: str,
        answer: str,
        reference: Optional[str] = None,
        **kwargs,
    ) -> EvaluationMetrics:
        """
        Evaluate a single sample across all criteria.
        
        Args:
            instruction: Input instruction
            answer: Model answer
            reference: Reference answer (optional)
            **kwargs: Additional parameters
            
        Returns:
            Evaluation metrics
        """
        metrics = EvaluationMetrics(evaluator=self.name)
        
        # Evaluate each criterion
        for criterion in self.criteria:
            result = self._evaluate_criterion(criterion, instruction, answer, reference)
            
            # Normalize score to 0-1
            normalized_score = result["score"] / 5.0
            
            # Set appropriate field
            if criterion == "relevance":
                metrics.relevance_score = normalized_score
                metrics.relevance_analysis = result["analysis"]
            elif criterion == "coherence":
                metrics.coherence_score = normalized_score
                metrics.coherence_analysis = result["analysis"]
            elif criterion == "fluency":
                metrics.fluency_score = normalized_score
                metrics.fluency_analysis = result["analysis"]
            elif criterion == "consistency":
                metrics.consistency_score = normalized_score
                metrics.consistency_analysis = result["analysis"]
            elif criterion == "accuracy":
                metrics.accuracy_score = normalized_score
                metrics.accuracy_analysis = result["analysis"]
        
        # Compute overall score
        metrics.overall_score = metrics.compute_overall()
        
        return metrics
    
    def evaluate_batch(
        self,
        instructions: List[str],
        answers: List[str],
        references: Optional[List[str]] = None,
        **kwargs,
    ) -> List[EvaluationMetrics]:
        """
        Evaluate batch in parallel.
        
        Args:
            instructions: List of instructions
            answers: List of answers
            references: List of references (optional)
            **kwargs: Additional parameters
            
        Returns:
            List of evaluation metrics
        """
        logger.info(f"Evaluating {len(instructions)} samples with {self.max_workers} workers")
        
        def evaluate_one(idx):
            return self.evaluate_sample(
                instruction=instructions[idx],
                answer=answers[idx],
                reference=references[idx] if references else None,
                **kwargs,
            )
        
        metrics_list = [None] * len(instructions)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(evaluate_one, i): i for i in range(len(instructions))}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                idx = futures[future]
                try:
                    metrics_list[idx] = future.result()
                except Exception as e:
                    logger.error(f"Failed to evaluate sample {idx}: {e}")
                    metrics_list[idx] = EvaluationMetrics(evaluator=self.name)
        
        return metrics_list


class AccuracyStyleEvaluator(BaseEvaluator):
    """
    Evaluator for accuracy and style (similar to old codebase).
    
    Optimized for blog/social media content evaluation.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_workers: int = 10,
        name: str = "accuracy_style",
    ) -> None:
        """
        Initialize evaluator.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model
            max_workers: Number of parallel workers
            name: Evaluator name
        """
        super().__init__(name=name)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_workers = max_workers
    
    def evaluate_sample(
        self,
        instruction: str,
        answer: str,
        reference: Optional[str] = None,
        **kwargs,
    ) -> EvaluationMetrics:
        """Evaluate accuracy and style."""
        
        prompt = f"""You are an expert judge. Please evaluate the quality of a given answer to an instruction based on two criteria:

1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

Accuracy scale:
1 (Poor): Contains factual errors or misleading information
2 (Fair): Mostly accurate with some errors or omissions
3 (Good): Accurate with minor omissions
4 (Very Good): Highly accurate and comprehensive
5 (Excellent): Completely accurate and comprehensive

Style scale:
1 (Poor): Too formal, uses overly complex words
2 (Fair): Somewhat formal, uses some complex language
3 (Good): Good balance but still uses some formal expressions
4 (Very Good): Accessible language with good technical precision
5 (Excellent): Perfectly accessible for blog/social media with precise technical terms

Example of bad style: "The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence."
Example of excellent style: "Llama2 7B outperforms the original Llama model across multiple benchmarks."

Instruction: {instruction}

Answer: {answer}

Provide your evaluation in JSON format:
{{
    "accuracy": {{
        "score": <1-5>,
        "analysis": "..."
    }},
    "style": {{
        "score": <1-5>,
        "analysis": "..."
    }}
}}
"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who evaluates answers based on accuracy and style. Provide your response in JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=0.0,
            )
            
            result = json.loads(completion.choices[0].message.content)
            
            # Normalize scores to 0-1
            accuracy_score = result["accuracy"]["score"] / 5.0
            style_score = result["style"]["score"] / 5.0
            
            return EvaluationMetrics(
                accuracy_score=accuracy_score,
                accuracy_analysis=result["accuracy"]["analysis"],
                style_score=style_score,
                style_analysis=result["style"]["analysis"],
                overall_score=(accuracy_score + style_score) / 2,
                evaluator=self.name,
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationMetrics(evaluator=self.name)
    
    def evaluate_batch(
        self,
        instructions: List[str],
        answers: List[str],
        references: Optional[List[str]] = None,
        **kwargs,
    ) -> List[EvaluationMetrics]:
        """Evaluate batch in parallel."""
        
        logger.info(f"Evaluating {len(instructions)} samples")
        
        def evaluate_one(idx):
            return self.evaluate_sample(instructions[idx], answers[idx])
        
        metrics_list = [None] * len(instructions)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(evaluate_one, i): i for i in range(len(instructions))}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                idx = futures[future]
                try:
                    metrics_list[idx] = future.result()
                except Exception as e:
                    logger.error(f"Failed to evaluate sample {idx}: {e}")
                    metrics_list[idx] = EvaluationMetrics(evaluator=self.name)
        
        return metrics_list
