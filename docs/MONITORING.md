# Monitoring & Observability Documentation

## Overview

HermesLLM provides comprehensive monitoring and observability through:
- **Opik** (powered by Comet ML) for prompt monitoring and LLM tracking
- **Comet ML** for experiment tracking during training
- **Decorators** for automatic instrumentation
- **Rich logging** with structured metrics

## Architecture

```
hermes/monitoring/
├── __init__.py           # Module exports
├── opik_utils.py         # Opik configuration and tracking
├── decorators.py         # Monitoring decorators
└── comet_tracker.py      # Comet ML experiment tracking
```

## Opik Integration

### Configuration

#### Automatic Configuration

```python
from hermes.monitoring import configure_opik

# Configure with settings
success = configure_opik()

# Configure with custom parameters
success = configure_opik(
    api_key="your-api-key",
    project_name="my-project",
    workspace="my-workspace"
)
```

#### Environment Variables

```bash
# .env file
OPIK_API_KEY=your-opik-api-key
OPIK_WORKSPACE=your-workspace
OPIK_PROJECT_NAME=hermesllm
```

### Tracking Decorators

#### Track LLM Calls

```python
from hermes.monitoring import track_llm

@track_llm(name="generate_answer", tags=["rag", "production"])
def generate(prompt: str, temperature: float = 0.7) -> str:
    return llm.generate(prompt, temperature=temperature)
```

#### Track Pipelines

```python
from hermes.monitoring import track_pipeline

@track_pipeline(name="rag_pipeline", tags=["rag"])
def process_query(query: str) -> str:
    docs = retrieve(query)
    answer = generate(query, docs)
    return answer
```

### Logging Metrics

#### Log Custom Metrics

```python
from hermes.monitoring import log_metrics

log_metrics({
    "latency_ms": 150,
    "num_tokens": 500,
    "success": True,
})
```

#### Log Token Usage

```python
from hermes.monitoring import log_tokens

log_tokens(
    query_tokens=10,
    context_tokens=500,
    answer_tokens=100,
)
```

#### Log Model Information

```python
from hermes.monitoring import log_model_info

log_model_info(
    model_id="meta-llama/Llama-2-7b-hf",
    temperature=0.7,
    max_tokens=512,
    top_p=0.9,
)
```

### Advanced Usage

#### Create Custom Traces

```python
from hermes.monitoring import create_trace

trace = create_trace(
    name="custom_operation",
    input_data={"query": "What is RAG?"},
    tags=["custom"],
    metadata={"version": "1.0"},
)
```

#### Monitoring Context

```python
from hermes.monitoring import MonitoringContext

with MonitoringContext("expensive_operation", tags=["batch"]) as monitor:
    results = process_batch(data)
    monitor.log("items_processed", len(results))
```

## Monitoring Decorators

### RAG Query Monitoring

```python
from hermes.monitoring import monitor_rag_query

@monitor_rag_query(name="customer_support_rag")
def query_rag(query: str) -> dict:
    docs = retrieve(query)
    answer = generate(query, docs)
    return {
        "answer": answer,
        "documents": docs,
        "context": build_context(docs),
        "query": query,
    }
```

**Tracked Metrics**:
- Query latency
- Token counts (query, context, answer)
- Number of retrieved documents
- Model configuration

### Retrieval Monitoring

```python
from hermes.monitoring import monitor_retrieval

@monitor_retrieval(name="vector_search")
def retrieve(query: str, k: int = 5) -> list:
    return vector_db.search(query, k=k)
```

**Tracked Metrics**:
- Retrieval latency
- Number of documents retrieved
- Query tokens
- Requested k value

### Inference Monitoring

```python
from hermes.monitoring import monitor_inference

@monitor_inference(name="llm_generation")
def generate(prompt: str, temperature: float = 0.7) -> str:
    return model.generate(prompt, temperature=temperature)
```

**Tracked Metrics**:
- Inference latency
- Input/output token counts
- Model configuration
- Success/failure status

### Training Step Monitoring

```python
from hermes.monitoring import monitor_training_step

@monitor_training_step(name="train_epoch")
def train_step(batch, optimizer, lr):
    loss = model(batch)
    optimizer.step()
    return {"loss": loss.item(), "lr": lr}
```

**Tracked Metrics**:
- Step latency
- Loss values
- Learning rate
- Batch size

### Data Processing Monitoring

```python
from hermes.monitoring import monitor_data_processing

@monitor_data_processing(name="chunk_documents")
def chunk_docs(documents: list) -> list:
    return [chunk for doc in documents for chunk in chunker(doc)]
```

**Tracked Metrics**:
- Processing latency
- Input/output counts
- Items per second

## Comet ML Experiment Tracking

### Creating an Experiment

```python
from hermes.monitoring import create_comet_experiment

with create_comet_experiment(
    experiment_name="training_run_001",
    tags=["LoRA", "SFT", "Llama-2"],
) as tracker:
    # Log hyperparameters
    tracker.log_parameters({
        "learning_rate": 3e-4,
        "num_epochs": 3,
        "batch_size": 4,
        "lora_r": 16,
        "lora_alpha": 32,
    })
    
    # Training loop
    for epoch in range(3):
        loss = train_epoch()
        
        # Log metrics
        tracker.log_metric("loss", loss, epoch=epoch)
        tracker.log_metric("learning_rate", get_lr(), epoch=epoch)
    
    # Log final model
    tracker.log_model("final_model", "./checkpoints/final")
```

### CometTracker API

```python
from hermes.monitoring import CometTracker

tracker = CometTracker(
    project_name="hermesllm",
    experiment_name="my_experiment",
)

# Log parameters
tracker.log_parameters({"lr": 3e-4, "epochs": 3})

# Log metrics
tracker.log_metrics({"loss": 0.5, "accuracy": 0.95}, step=100)

# Log single metric
tracker.log_metric("f1_score", 0.93, step=100)

# Log model
tracker.log_model("checkpoint_100", "./models/checkpoint-100")

# Log artifacts
tracker.log_artifact("./data/dataset.json", artifact_type="dataset")

# Add tags
tracker.add_tags(["production", "v2"])

# End experiment
tracker.end()
```

## Integration Examples

### RAG Pipeline with Full Monitoring

```python
from hermes.monitoring import configure_opik, monitor_rag_query
from hermes.rag import RAGPipeline

# Configure Opik
configure_opik()

# RAG pipeline is automatically monitored
pipeline = RAGPipeline()

# Query with monitoring
result = pipeline.query(
    query="What is RAG?",
    use_query_expansion=True,
    use_self_query=True,
)

# Access answer
answer = result["answer"]
```

**Automatically Tracked**:
- Query processing time
- Token usage
- Number of documents retrieved
- Model configuration
- Query expansion details

### Training with Comet ML

```python
from hermes.monitoring import create_comet_experiment
from transformers import TrainingArguments

# Create experiment
with create_comet_experiment("fine_tune_llama", tags=["LoRA"]) as tracker:
    # Log configuration
    tracker.log_parameters({
        "model": "meta-llama/Llama-2-7b-hf",
        "learning_rate": 3e-4,
        "lora_r": 16,
    })
    
    # Training arguments with Comet ML
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        report_to="comet_ml",  # Enable Comet ML integration
        logging_steps=10,
        num_train_epochs=3,
    )
    
    # Train (automatically logs to Comet ML)
    trainer.train()
    
    # Log final model
    tracker.log_model("final_model", "./checkpoints/final")
```

### Custom Monitoring

```python
from hermes.monitoring import MonitoringContext, log_metrics

def process_data(documents):
    with MonitoringContext("data_processing", tags=["batch"]) as monitor:
        # Process documents
        results = []
        for doc in documents:
            result = process_document(doc)
            results.append(result)
        
        # Log metrics
        monitor.log("documents_processed", len(results))
        monitor.log("errors", count_errors(results))
        
        # Additional metrics
        log_metrics({
            "avg_doc_length": avg_length(results),
            "total_tokens": sum_tokens(results),
        })
        
        return results
```

## Best Practices

### 1. Always Configure Monitoring

```python
from hermes.monitoring import configure_opik

# At application startup
if __name__ == "__main__":
    configure_opik()
    main()
```

### 2. Use Appropriate Decorators

```python
# For RAG queries
@monitor_rag_query()
def rag_query(query: str) -> dict:
    ...

# For document retrieval
@monitor_retrieval()
def retrieve(query: str) -> list:
    ...

# For LLM inference
@monitor_inference()
def generate(prompt: str) -> str:
    ...
```

### 3. Log Meaningful Metrics

```python
# Good: Specific, actionable metrics
log_metrics({
    "retrieval_latency_ms": 150,
    "num_docs_retrieved": 5,
    "reranking_enabled": True,
})

# Avoid: Vague or redundant metrics
log_metrics({
    "status": "ok",
    "result": "success",
})
```

### 4. Use Tags for Organization

```python
@track_pipeline(tags=["production", "rag", "v2"])
def production_rag(query: str):
    ...

@track_pipeline(tags=["development", "testing"])
def test_rag(query: str):
    ...
```

### 5. Handle Monitoring Failures Gracefully

```python
from hermes.monitoring import is_opik_enabled, log_metrics

# Monitoring is optional
if is_opik_enabled():
    log_metrics({"custom_metric": value})
else:
    # Continue without monitoring
    logger.info("Monitoring disabled, skipping metrics")
```

## Monitoring in Production

### Environment Setup

```bash
# Production .env
OPIK_API_KEY=prod-api-key
OPIK_WORKSPACE=production
OPIK_PROJECT_NAME=hermesllm-prod
COMET_API_KEY=comet-api-key
COMET_PROJECT=hermesllm
COMET_WORKSPACE=production
```

### Deployment Checklist

- [ ] Configure Opik with production credentials
- [ ] Set up Comet ML workspace
- [ ] Add monitoring decorators to critical paths
- [ ] Test monitoring in staging environment
- [ ] Set up alerts for anomalies
- [ ] Review metrics dashboard regularly

### Monitoring Critical Operations

```python
# RAG queries (most critical)
@monitor_rag_query(name="production_rag", tags=["critical", "production"])
def production_query(query: str):
    ...

# Model inference
@monitor_inference(name="production_inference", tags=["critical"])
def production_inference(prompt: str):
    ...

# Document retrieval
@monitor_retrieval(name="production_retrieval", tags=["critical"])
def production_retrieve(query: str):
    ...
```

## Troubleshooting

### Monitoring Not Working

```python
from hermes.monitoring import is_opik_enabled, configure_opik

# Check if Opik is enabled
if not is_opik_enabled():
    print("Opik not configured")
    
    # Reconfigure
    success = configure_opik()
    if not success:
        print("Failed to configure Opik. Check API key.")
```

### Missing Metrics

```python
# Ensure decorators return appropriate data
@monitor_rag_query()
def rag_query(query: str) -> dict:
    # MUST return dict with these keys
    return {
        "answer": answer,
        "documents": docs,
        "context": context,
        "query": query,
    }
```

### High Overhead

```python
# Disable monitoring for non-critical operations
if os.getenv("ENABLE_DETAILED_MONITORING") == "true":
    @monitor_data_processing()
    def process_batch(data):
        ...
else:
    def process_batch(data):
        ...
```

## References

- [Opik Documentation](https://www.comet.com/docs/opik/)
- [Comet ML Documentation](https://www.comet.com/docs/)
- [OpenTelemetry Best Practices](https://opentelemetry.io/docs/)
- [Monitoring Best Practices](https://sre.google/sre-book/monitoring-distributed-systems/)
