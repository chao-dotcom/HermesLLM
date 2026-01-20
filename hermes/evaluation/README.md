# Model Evaluation System

Complete model evaluation infrastructure with G-Eval, benchmarks, and ZenML integration.

## Features

✅ **G-Eval**: LLM-based evaluation using GPT-4o-mini across multiple criteria  
✅ **Accuracy & Style**: Specialized evaluation for blog/social content  
✅ **Benchmarks**: MMLU (57 subjects), GSM8K (math reasoning)  
✅ **Parallel Processing**: Multi-threaded evaluation with configurable workers  
✅ **ZenML Integration**: Pipeline orchestration for reproducible evaluations  
✅ **HuggingFace Hub**: Push results directly to Hub  

## Quick Start

### 1. Install Dependencies

```bash
pip install openai vllm datasets
```

### 2. Basic Evaluation

```python
from hermes.evaluation import AccuracyStyleEvaluator
import os

evaluator = AccuracyStyleEvaluator(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
)

metrics = evaluator.evaluate_batch(
    instructions=["Explain transformers in ML."],
    answers=["Transformers use self-attention..."],
)

print(f"Accuracy: {metrics[0].accuracy_score:.2f}")
print(f"Style: {metrics[0].style_score:.2f}")
```

### 3. Run Benchmark

```python
from hermes.evaluation import MMLUBenchmark, run_benchmark
from vllm import LLM, SamplingParams

benchmark = MMLUBenchmark(subjects=["machine_learning"])
llm = LLM(model="path/to/model")

def generate(prompts):
    outputs = llm.generate(prompts, SamplingParams(temperature=0))
    return [o.outputs[0].text for o in outputs]

accuracy, _ = run_benchmark(benchmark, generate)
print(f"MMLU Accuracy: {accuracy:.2%}")
```

### 4. ZenML Pipeline

```python
from workflows.pipelines.evaluation import model_evaluation_pipeline

model_evaluation_pipeline(
    dataset_path="data/test.json",
    model_path="path/to/model",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    output_path="data/results.json",
)
```

## Module Structure

```
hermes/evaluation/
├── __init__.py           # Module exports
├── metrics.py            # EvaluationMetrics, EvaluationResult
├── evaluator.py          # Base evaluators (BaseEvaluator, LLMEvaluator, BenchmarkEvaluator)
├── g_eval.py             # G-Eval and AccuracyStyleEvaluator
└── benchmarks.py         # MMLU, GSM8K benchmarks

tasks/evaluation/
├── __init__.py
└── steps.py              # ZenML evaluation steps

workflows/pipelines/
└── evaluation.py         # Evaluation pipelines

examples/
└── evaluation_example.py # Comprehensive examples

docs/
└── EVALUATION.md         # Complete documentation
```

## Evaluation Methods

### 1. G-Eval (Multi-Criteria)

Evaluates across:
- Relevance
- Coherence
- Fluency
- Consistency
- Accuracy (with reference)

```python
from hermes.evaluation import GEval

evaluator = GEval(
    api_key=os.getenv("OPENAI_API_KEY"),
    criteria=["relevance", "coherence", "fluency"],
)
```

### 2. Accuracy & Style (Blog Content)

Optimized for blog posts and social media:

```python
from hermes.evaluation import AccuracyStyleEvaluator

evaluator = AccuracyStyleEvaluator(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_workers=10,
)
```

### 3. Benchmarks

Standard benchmarks:

```python
from hermes.evaluation import MMLUBenchmark, GSM8KBenchmark

# MMLU (57 subjects)
mmlu = MMLUBenchmark(subjects=None)  # All subjects

# GSM8K (math)
gsm8k = GSM8KBenchmark()
```

## Performance

- **Parallel evaluation**: 10-20 concurrent requests
- **Batch processing**: Process 100+ samples/batch
- **Checkpointing**: Save progress every N samples
- **Cost-effective**: Uses GPT-4o-mini (~$0.001/sample)

## Examples

See [examples/evaluation_example.py](../examples/evaluation_example.py) for:

1. G-Eval evaluation
2. Accuracy & style evaluation
3. MMLU benchmark
4. GSM8K benchmark
5. End-to-end evaluation with vLLM
6. ZenML pipelines
7. Batch processing with progress tracking

## Documentation

Complete guide: [docs/EVALUATION.md](../docs/EVALUATION.md)

Topics covered:
- Installation
- Quick start
- All evaluation methods
- ZenML pipelines
- Advanced usage
- Best practices
- Troubleshooting

## Configuration

### Development

```python
evaluator = GEval(
    api_key=api_key,
    model="gpt-4o-mini",
    criteria=["relevance", "coherence"],  # Fewer criteria
    max_workers=5,  # Lower parallelism
)
```

### Production

```python
evaluator = AccuracyStyleEvaluator(
    api_key=api_key,
    model="gpt-4o-mini",
    max_workers=20,  # High parallelism
)

# Batch processing with checkpoints
for i in range(0, len(data), 100):
    metrics = evaluator.evaluate_batch(data[i:i+100])
    if i % 500 == 0:
        save_checkpoint(metrics)
```

## Requirements

```bash
# Core
openai>=1.0.0

# Optional
vllm>=0.2.0        # For answer generation
datasets>=2.14.0    # For benchmark datasets
zenml>=0.50.0       # For pipeline orchestration
```

## API Reference

### EvaluationMetrics

```python
class EvaluationMetrics:
    accuracy_score: float = 0.0
    style_score: float = 0.0
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    fluency_score: float = 0.0
    consistency_score: float = 0.0
    overall_score: float = 0.0
    # ... analysis fields
```

### BaseEvaluator

```python
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate_sample(self, instruction, answer, **kwargs) -> EvaluationMetrics:
        """Evaluate single sample."""
        
    @abstractmethod
    def evaluate_batch(self, instructions, answers, **kwargs) -> List[EvaluationMetrics]:
        """Evaluate batch."""
```

### GEval

```python
GEval(
    api_key: str,
    model: str = "gpt-4o-mini",
    criteria: Optional[List[str]] = None,
    max_workers: int = 10,
)
```

### AccuracyStyleEvaluator

```python
AccuracyStyleEvaluator(
    api_key: str,
    model: str = "gpt-4o-mini",
    max_workers: int = 10,
)
```

### MMLUBenchmark

```python
MMLUBenchmark(
    subjects: Optional[List[str]] = None,  # All 57 subjects if None
    split: str = "test",
    num_samples: Optional[int] = None,
)
```

## License

Same as parent project (see root LICENSE file).
