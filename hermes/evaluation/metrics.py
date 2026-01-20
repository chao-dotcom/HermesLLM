"""Evaluation metrics and result classes."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import statistics


class EvaluationMetrics(BaseModel):
    """Metrics for a single evaluation."""
    
    # Accuracy metrics
    accuracy_score: Optional[float] = Field(default=None, description="Accuracy score (0-1 or 1-5)")
    accuracy_analysis: Optional[str] = Field(default=None, description="Accuracy analysis")
    
    # Style/Quality metrics
    style_score: Optional[float] = Field(default=None, description="Style score (0-1 or 1-5)")
    style_analysis: Optional[str] = Field(default=None, description="Style analysis")
    
    # Relevance
    relevance_score: Optional[float] = Field(default=None, description="Relevance score")
    relevance_analysis: Optional[str] = Field(default=None, description="Relevance analysis")
    
    # Coherence
    coherence_score: Optional[float] = Field(default=None, description="Coherence score")
    coherence_analysis: Optional[str] = Field(default=None, description="Coherence analysis")
    
    # Fluency
    fluency_score: Optional[float] = Field(default=None, description="Fluency score")
    fluency_analysis: Optional[str] = Field(default=None, description="Fluency analysis")
    
    # Consistency
    consistency_score: Optional[float] = Field(default=None, description="Consistency score")
    consistency_analysis: Optional[str] = Field(default=None, description="Consistency analysis")
    
    # Benchmark scores
    exact_match: Optional[float] = Field(default=None, description="Exact match score")
    f1_score: Optional[float] = Field(default=None, description="F1 score")
    bleu_score: Optional[float] = Field(default=None, description="BLEU score")
    rouge_scores: Optional[Dict[str, float]] = Field(default=None, description="ROUGE scores")
    
    # Overall
    overall_score: Optional[float] = Field(default=None, description="Overall/average score")
    
    # Metadata
    evaluator: Optional[str] = Field(default=None, description="Evaluator name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    
    def compute_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted overall score."""
        scores = {}
        
        if self.accuracy_score is not None:
            scores["accuracy"] = self.accuracy_score
        if self.style_score is not None:
            scores["style"] = self.style_score
        if self.relevance_score is not None:
            scores["relevance"] = self.relevance_score
        if self.coherence_score is not None:
            scores["coherence"] = self.coherence_score
        if self.fluency_score is not None:
            scores["fluency"] = self.fluency_score
        if self.consistency_score is not None:
            scores["consistency"] = self.consistency_score
        
        if not scores:
            return 0.0
        
        if weights:
            # Weighted average
            total_weight = sum(weights.get(k, 1.0) for k in scores.keys())
            weighted_sum = sum(scores[k] * weights.get(k, 1.0) for k in scores.keys())
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            # Simple average
            return statistics.mean(scores.values())


class EvaluationResult(BaseModel):
    """Complete evaluation result for a dataset or model."""
    
    model_id: str = Field(..., description="Model identifier")
    dataset_name: Optional[str] = Field(default=None, description="Dataset name")
    num_samples: int = Field(default=0, description="Number of samples evaluated")
    
    # Individual sample metrics
    sample_metrics: List[EvaluationMetrics] = Field(default_factory=list, description="Per-sample metrics")
    
    # Aggregated metrics
    average_accuracy: Optional[float] = Field(default=None, description="Average accuracy")
    average_style: Optional[float] = Field(default=None, description="Average style")
    average_relevance: Optional[float] = Field(default=None, description="Average relevance")
    average_coherence: Optional[float] = Field(default=None, description="Average coherence")
    average_fluency: Optional[float] = Field(default=None, description="Average fluency")
    average_overall: Optional[float] = Field(default=None, description="Average overall")
    
    # Benchmark results
    benchmark_scores: Optional[Dict[str, Any]] = Field(default=None, description="Benchmark scores")
    
    # Metadata
    evaluation_config: Optional[Dict[str, Any]] = Field(default=None, description="Evaluation configuration")
    timestamp: datetime = Field(default_factory=datetime.now, description="Evaluation timestamp")
    duration_seconds: Optional[float] = Field(default=None, description="Evaluation duration")
    
    def aggregate_metrics(self) -> None:
        """Aggregate sample metrics into averages."""
        if not self.sample_metrics:
            return
        
        # Collect scores
        accuracy_scores = [m.accuracy_score for m in self.sample_metrics if m.accuracy_score is not None]
        style_scores = [m.style_score for m in self.sample_metrics if m.style_score is not None]
        relevance_scores = [m.relevance_score for m in self.sample_metrics if m.relevance_score is not None]
        coherence_scores = [m.coherence_score for m in self.sample_metrics if m.coherence_score is not None]
        fluency_scores = [m.fluency_score for m in self.sample_metrics if m.fluency_score is not None]
        overall_scores = [m.overall_score for m in self.sample_metrics if m.overall_score is not None]
        
        # Compute averages
        self.average_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else None
        self.average_style = statistics.mean(style_scores) if style_scores else None
        self.average_relevance = statistics.mean(relevance_scores) if relevance_scores else None
        self.average_coherence = statistics.mean(coherence_scores) if coherence_scores else None
        self.average_fluency = statistics.mean(fluency_scores) if fluency_scores else None
        self.average_overall = statistics.mean(overall_scores) if overall_scores else None
        
        self.num_samples = len(self.sample_metrics)
    
    def to_summary(self) -> Dict[str, Any]:
        """Get summary dictionary."""
        return {
            "model_id": self.model_id,
            "dataset": self.dataset_name,
            "num_samples": self.num_samples,
            "average_accuracy": self.average_accuracy,
            "average_style": self.average_style,
            "average_relevance": self.average_relevance,
            "average_coherence": self.average_coherence,
            "average_fluency": self.average_fluency,
            "average_overall": self.average_overall,
            "benchmark_scores": self.benchmark_scores,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
        }