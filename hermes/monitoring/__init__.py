"""Monitoring and observability module for HermesLLM."""

from hermes.monitoring.opik_utils import (
    configure_opik,
    create_trace,
    flush_tracker,
    get_current_trace,
    is_opik_enabled,
    log_metrics,
    log_model_info,
    log_tokens,
    track_llm,
    track_pipeline,
)
from hermes.monitoring.decorators import (
    monitor_rag_query,
    monitor_retrieval,
    monitor_inference,
    monitor_training_step,
    monitor_data_processing,
    MonitoringContext,
    compute_tokens,
)
from hermes.monitoring.comet_tracker import (
    CometTracker,
    create_comet_experiment,
)

__all__ = [
    # Opik utilities
    "configure_opik",
    "is_opik_enabled",
    "track_llm",
    "track_pipeline",
    "log_metrics",
    "log_tokens",
    "log_model_info",
    "create_trace",
    "get_current_trace",
    "flush_tracker",
    # Decorators
    "monitor_rag_query",
    "monitor_retrieval",
    "monitor_inference",
    "monitor_training_step",
    "monitor_data_processing",
    "MonitoringContext",
    "compute_tokens",
    # Comet ML
    "CometTracker",
    "create_comet_experiment",
]
