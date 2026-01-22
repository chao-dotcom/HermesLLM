"""
Opik utilities for monitoring and observability.

This module provides utilities for configuring and using Opik (powered by Comet ML)
for prompt monitoring, LLM tracking, and experiment tracking.
"""

import os
from functools import wraps
from typing import Any, Callable, Optional

from loguru import logger

try:
    import opik
    from opik import opik_context
    from opik.configurator.configure import OpikConfigurator
    
    OPIK_AVAILABLE = True
except ImportError:
    logger.warning("Opik not installed. Install with: pip install opik")
    OPIK_AVAILABLE = False
    opik = None
    opik_context = None


def configure_opik(
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    project_name: Optional[str] = None,
    use_local: bool = False,
    force: bool = True,
) -> bool:
    """
    Configure Opik for monitoring.

    Args:
        api_key: Opik API key (uses OPIK_API_KEY env var if not provided)
        workspace: Opik workspace (auto-detected if not provided)
        project_name: Project name (uses OPIK_PROJECT_NAME env var if not provided)
        use_local: Whether to use local Opik server
        force: Force reconfiguration

    Returns:
        True if configured successfully, False otherwise
    """
    if not OPIK_AVAILABLE:
        logger.warning("Opik not available. Skipping configuration.")
        return False

    # Get settings from environment or parameters
    from hermes.config import get_settings
    settings = get_settings()
    
    api_key = api_key or settings.opik_api_key or os.getenv("OPIK_API_KEY")
    workspace = workspace or settings.opik_workspace or os.getenv("OPIK_WORKSPACE")
    project_name = project_name or settings.comet_project or os.getenv("OPIK_PROJECT_NAME", "hermesllm")

    if not api_key:
        logger.warning(
            "OPIK_API_KEY not set. Opik monitoring disabled. "
            "Set OPIK_API_KEY to enable prompt monitoring."
        )
        return False

    try:
        # Get default workspace if not provided
        if not workspace and api_key:
            try:
                configurator = OpikConfigurator(api_key=api_key)
                workspace = configurator._get_default_workspace()
                logger.info(f"Auto-detected Opik workspace: {workspace}")
            except Exception as e:
                logger.warning(f"Could not auto-detect workspace: {e}")
                logger.info("Setting workspace to None and enabling interactive mode.")
                workspace = None

        # Set environment variable for project name
        os.environ["OPIK_PROJECT_NAME"] = project_name

        # Configure Opik
        opik.configure(
            api_key=api_key,
            workspace=workspace,
            use_local=use_local,
            force=force,
        )

        logger.info(
            f"✅ Opik configured successfully. "
            f"Project: {project_name}, Workspace: {workspace or 'default'}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to configure Opik: {e}")
        return False


def is_opik_enabled() -> bool:
    """
    Check if Opik is available and configured.

    Returns:
        True if Opik is available, False otherwise
    """
    return OPIK_AVAILABLE and opik is not None


def track_llm(
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Callable:
    """
    Decorator for tracking LLM calls with Opik.

    Args:
        name: Custom name for the tracked function
        tags: Tags to add to the trace
        metadata: Additional metadata to log

    Returns:
        Decorated function

    Example:
        @track_llm(name="generate_response", tags=["rag", "production"])
        def generate(prompt: str) -> str:
            return llm.generate(prompt)
    """
    def decorator(func: Callable) -> Callable:
        if not is_opik_enabled():
            # Return original function if Opik not available
            return func

        @opik.track(name=name or func.__name__)
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Update trace with tags and metadata
            if tags or metadata:
                try:
                    opik_context.update_current_trace(
                        tags=tags or [],
                        metadata=metadata or {},
                    )
                except Exception as e:
                    logger.debug(f"Failed to update Opik trace: {e}")
            
            return result
        
        return wrapper
    return decorator


def track_pipeline(
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Callable:
    """
    Decorator for tracking entire pipelines with Opik.

    Args:
        name: Custom name for the pipeline
        tags: Tags to add to the trace

    Returns:
        Decorated function

    Example:
        @track_pipeline(name="rag_pipeline", tags=["rag"])
        def rag_query(query: str) -> str:
            docs = retrieve(query)
            return generate(query, docs)
    """
    def decorator(func: Callable) -> Callable:
        if not is_opik_enabled():
            return func

        @opik.track(name=name or func.__name__)
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if tags:
                try:
                    opik_context.update_current_trace(tags=tags)
                except Exception as e:
                    logger.debug(f"Failed to update Opik trace: {e}")
            
            return result
        
        return wrapper
    return decorator


def log_metrics(metrics: dict[str, Any]) -> None:
    """
    Log metrics to current Opik trace.

    Args:
        metrics: Dictionary of metrics to log

    Example:
        log_metrics({
            "latency_ms": 150,
            "token_count": 50,
            "success": True,
        })
    """
    if not is_opik_enabled():
        logger.debug("Opik not enabled. Metrics not logged.")
        return

    try:
        opik_context.update_current_trace(metadata=metrics)
    except Exception as e:
        logger.debug(f"Failed to log metrics to Opik: {e}")


def log_tokens(
    query_tokens: int = 0,
    context_tokens: int = 0,
    answer_tokens: int = 0,
    total_tokens: int = 0,
) -> None:
    """
    Log token usage to current Opik trace.

    Args:
        query_tokens: Number of tokens in query
        context_tokens: Number of tokens in context
        answer_tokens: Number of tokens in answer
        total_tokens: Total number of tokens

    Example:
        log_tokens(
            query_tokens=10,
            context_tokens=500,
            answer_tokens=100,
            total_tokens=610,
        )
    """
    metrics = {
        "query_tokens": query_tokens,
        "context_tokens": context_tokens,
        "answer_tokens": answer_tokens,
        "total_tokens": total_tokens or (query_tokens + context_tokens + answer_tokens),
    }
    log_metrics(metrics)


def log_model_info(
    model_id: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    **kwargs,
) -> None:
    """
    Log model configuration to current Opik trace.

    Args:
        model_id: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        **kwargs: Additional model parameters

    Example:
        log_model_info(
            model_id="meta-llama/Llama-2-7b-hf",
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
        )
    """
    metadata = {
        "model_id": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }
    log_metrics(metadata)


def create_trace(
    name: str,
    input_data: Any = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Create a new Opik trace.

    Args:
        name: Trace name
        input_data: Input data to log
        tags: Tags for the trace
        metadata: Additional metadata

    Returns:
        Trace object if Opik enabled, None otherwise

    Example:
        with create_trace("rag_query", input_data={"query": "What is RAG?"}) as trace:
            result = process_query()
            trace.log_output(result)
    """
    if not is_opik_enabled():
        return None

    try:
        trace = opik.trace(
            name=name,
            input=input_data,
            tags=tags or [],
            metadata=metadata or {},
        )
        return trace
    except Exception as e:
        logger.debug(f"Failed to create Opik trace: {e}")
        return None


def get_current_trace():
    """
    Get the current Opik trace.

    Returns:
        Current trace object if available, None otherwise
    """
    if not is_opik_enabled():
        return None

    try:
        return opik_context.get_current_trace()
    except Exception:
        return None


def flush_tracker() -> None:
    """
    Flush Opik tracker to ensure all data is sent.

    Useful when ending a script or application.
    """
    if not is_opik_enabled():
        return

    try:
        opik.flush_tracker()
        logger.debug("Opik tracker flushed successfully")
    except Exception as e:
        logger.debug(f"Failed to flush Opik tracker: {e}")
