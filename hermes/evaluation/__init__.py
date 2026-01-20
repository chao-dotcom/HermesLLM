"""Model evaluation system."""

# Check optional dependencies
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# Core exports (always available)
from hermes.evaluation.metrics import EvaluationMetrics, EvaluationResult
from hermes.evaluation.evaluator import BaseEvaluator, LLMEvaluator, BenchmarkEvaluator

# Conditional exports
if OPENAI_AVAILABLE:
    from hermes.evaluation.g_eval import GEval, AccuracyStyleEvaluator

if DATASETS_AVAILABLE:
    from hermes.evaluation.benchmarks import MMLUBenchmark, GSM8KBenchmark, run_benchmark

__all__ = [
    # Metrics
    "EvaluationMetrics",
    "EvaluationResult",
    
    # Base evaluators
    "BaseEvaluator",
    "LLMEvaluator",
    "BenchmarkEvaluator",
    
    # Flags
    "OPENAI_AVAILABLE",
    "VLLM_AVAILABLE",
    "DATASETS_AVAILABLE",
]

# Add conditional exports
if OPENAI_AVAILABLE:
    __all__.extend(["GEval", "AccuracyStyleEvaluator"])

if DATASETS_AVAILABLE:
    __all__.extend(["MMLUBenchmark", "GSM8KBenchmark", "run_benchmark"])