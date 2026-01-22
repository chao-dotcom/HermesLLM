"""
Monitoring decorators and utilities for tracking ML operations.

This module provides high-level decorators and utilities for monitoring:
- RAG queries and retrieval
- Model inference
- Training runs
- Data processing pipelines
"""

import time
from functools import wraps
from typing import Any, Callable, Optional

import tiktoken
from loguru import logger

from hermes.monitoring.opik_utils import (
    is_opik_enabled,
    log_metrics,
    log_model_info,
    log_tokens,
    track_llm,
    track_pipeline,
)


def compute_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Compute number of tokens in text.

    Args:
        text: Text to tokenize
        model: Model name for tokenizer

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4


def monitor_rag_query(
    name: str = "rag_query",
    tags: Optional[list[str]] = None,
) -> Callable:
    """
    Decorator for monitoring RAG queries.

    Tracks:
    - Query latency
    - Token counts (query, context, answer)
    - Number of retrieved documents
    - Model configuration

    Args:
        name: Name for the tracked operation
        tags: Tags to add to the trace

    Returns:
        Decorated function

    Example:
        @monitor_rag_query(name="customer_support_rag")
        def query_rag(query: str) -> dict:
            docs = retrieve(query)
            answer = generate(query, docs)
            return {"answer": answer, "documents": docs}
    """
    def decorator(func: Callable) -> Callable:
        @track_pipeline(name=name, tags=tags or ["rag"])
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            # Extract data from result
            query = args[0] if args else kwargs.get("query", "")
            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            documents = result.get("documents", []) if isinstance(result, dict) else []
            context = result.get("context", "") if isinstance(result, dict) else ""
            
            # Compute metrics
            latency_ms = (time.time() - start_time) * 1000
            query_tokens = compute_tokens(str(query))
            context_tokens = compute_tokens(context) if context else 0
            answer_tokens = compute_tokens(answer)
            
            # Log to Opik
            if is_opik_enabled():
                log_tokens(
                    query_tokens=query_tokens,
                    context_tokens=context_tokens,
                    answer_tokens=answer_tokens,
                )
                log_metrics({
                    "latency_ms": latency_ms,
                    "num_documents": len(documents),
                    "has_context": bool(context),
                })
            
            logger.info(
                f"RAG query completed in {latency_ms:.0f}ms. "
                f"Tokens: {query_tokens + context_tokens + answer_tokens}"
            )
            
            return result
        
        return wrapper
    return decorator


def monitor_retrieval(
    name: str = "retrieve_documents",
    tags: Optional[list[str]] = None,
) -> Callable:
    """
    Decorator for monitoring document retrieval.

    Tracks:
    - Retrieval latency
    - Number of documents retrieved
    - Query tokens
    - Reranking usage

    Args:
        name: Name for the tracked operation
        tags: Tags to add to the trace

    Returns:
        Decorated function

    Example:
        @monitor_retrieval(name="vector_search")
        def retrieve(query: str, k: int = 5) -> list:
            return vector_db.search(query, k=k)
    """
    def decorator(func: Callable) -> Callable:
        @track_llm(name=name, tags=tags or ["retrieval"])
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Get query
            query = args[0] if args else kwargs.get("query", "")
            k = kwargs.get("k", 5)
            
            result = func(*args, **kwargs)
            
            # Compute metrics
            latency_ms = (time.time() - start_time) * 1000
            num_docs = len(result) if isinstance(result, list) else 0
            query_tokens = compute_tokens(str(query))
            
            # Log to Opik
            if is_opik_enabled():
                log_metrics({
                    "latency_ms": latency_ms,
                    "num_documents_retrieved": num_docs,
                    "query_tokens": query_tokens,
                    "requested_k": k,
                })
            
            logger.debug(
                f"Retrieved {num_docs} documents in {latency_ms:.0f}ms"
            )
            
            return result
        
        return wrapper
    return decorator


def monitor_inference(
    name: str = "llm_inference",
    tags: Optional[list[str]] = None,
) -> Callable:
    """
    Decorator for monitoring LLM inference.

    Tracks:
    - Inference latency
    - Token counts (input, output)
    - Model configuration
    - Inference success/failure

    Args:
        name: Name for the tracked operation
        tags: Tags to add to the trace

    Returns:
        Decorated function

    Example:
        @monitor_inference(name="generate_answer")
        def generate(prompt: str, temperature: float = 0.7) -> str:
            return model.generate(prompt, temperature=temperature)
    """
    def decorator(func: Callable) -> Callable:
        @track_llm(name=name, tags=tags or ["inference"])
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Get input
            prompt = args[0] if args else kwargs.get("prompt", "")
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 512)
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                # Compute metrics
                latency_ms = (time.time() - start_time) * 1000
                input_tokens = compute_tokens(str(prompt))
                output_tokens = compute_tokens(str(result)) if success else 0
                
                # Log to Opik
                if is_opik_enabled():
                    log_tokens(
                        query_tokens=input_tokens,
                        answer_tokens=output_tokens,
                    )
                    log_model_info(
                        model_id=kwargs.get("model_id", "unknown"),
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    log_metrics({
                        "latency_ms": latency_ms,
                        "success": success,
                        "error": error,
                    })
                
                logger.info(
                    f"Inference completed in {latency_ms:.0f}ms. "
                    f"Tokens: {input_tokens} → {output_tokens}"
                )
            
            return result
        
        return wrapper
    return decorator


def monitor_training_step(
    name: str = "training_step",
    tags: Optional[list[str]] = None,
) -> Callable:
    """
    Decorator for monitoring training steps.

    Tracks:
    - Step latency
    - Loss values
    - Learning rate
    - Batch size

    Args:
        name: Name for the tracked operation
        tags: Tags to add to the trace

    Returns:
        Decorated function

    Example:
        @monitor_training_step(name="train_epoch")
        def train_step(batch, optimizer, lr):
            loss = model(batch)
            return {"loss": loss.item()}
    """
    def decorator(func: Callable) -> Callable:
        @track_llm(name=name, tags=tags or ["training"])
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            # Compute metrics
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract metrics from result
            loss = result.get("loss") if isinstance(result, dict) else None
            lr = result.get("lr") if isinstance(result, dict) else kwargs.get("lr")
            
            # Log to Opik
            if is_opik_enabled():
                metrics = {
                    "latency_ms": latency_ms,
                }
                if loss is not None:
                    metrics["loss"] = loss
                if lr is not None:
                    metrics["learning_rate"] = lr
                
                log_metrics(metrics)
            
            return result
        
        return wrapper
    return decorator


def monitor_data_processing(
    name: str = "data_processing",
    tags: Optional[list[str]] = None,
) -> Callable:
    """
    Decorator for monitoring data processing operations.

    Tracks:
    - Processing latency
    - Number of items processed
    - Success rate

    Args:
        name: Name for the tracked operation
        tags: Tags to add to the trace

    Returns:
        Decorated function

    Example:
        @monitor_data_processing(name="chunk_documents")
        def chunk_docs(documents: list) -> list:
            return [chunk for doc in documents for chunk in chunker(doc)]
    """
    def decorator(func: Callable) -> Callable:
        @track_pipeline(name=name, tags=tags or ["processing"])
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Get input count
            input_data = args[0] if args else kwargs.get("documents", [])
            input_count = len(input_data) if isinstance(input_data, (list, tuple)) else 1
            
            result = func(*args, **kwargs)
            
            # Compute metrics
            latency_ms = (time.time() - start_time) * 1000
            output_count = len(result) if isinstance(result, (list, tuple)) else 1
            
            # Log to Opik
            if is_opik_enabled():
                log_metrics({
                    "latency_ms": latency_ms,
                    "input_count": input_count,
                    "output_count": output_count,
                    "items_per_second": output_count / (latency_ms / 1000) if latency_ms > 0 else 0,
                })
            
            logger.info(
                f"Processed {input_count} → {output_count} items in {latency_ms:.0f}ms"
            )
            
            return result
        
        return wrapper
    return decorator


class MonitoringContext:
    """
    Context manager for monitoring a block of code.

    Example:
        with MonitoringContext("my_operation", tags=["custom"]) as monitor:
            result = expensive_operation()
            monitor.log("items_processed", 100)
    """

    def __init__(
        self,
        name: str,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize monitoring context.

        Args:
            name: Operation name
            tags: Tags to add
            metadata: Initial metadata
        """
        self.name = name
        self.tags = tags or []
        self.metadata = metadata or {}
        self.start_time = None

    def __enter__(self):
        """Start monitoring."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End monitoring and log metrics."""
        if self.start_time:
            latency_ms = (time.time() - self.start_time) * 1000
            self.metadata["latency_ms"] = latency_ms
            
            if exc_type is not None:
                self.metadata["error"] = str(exc_val)
                self.metadata["success"] = False
            else:
                self.metadata["success"] = True
            
            if is_opik_enabled():
                log_metrics(self.metadata)
            
            logger.debug(f"{self.name} completed in {latency_ms:.0f}ms")

    def log(self, key: str, value: Any) -> None:
        """
        Log a metric.

        Args:
            key: Metric name
            value: Metric value
        """
        self.metadata[key] = value
