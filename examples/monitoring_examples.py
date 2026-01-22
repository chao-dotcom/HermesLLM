"""
Monitoring Examples for HermesLLM

Demonstrates various monitoring patterns and use cases.
"""

import time
from hermes.monitoring import (
    configure_opik,
    monitor_rag_query,
    monitor_retrieval,
    monitor_inference,
    monitor_training_step,
    monitor_data_processing,
    MonitoringContext,
    log_metrics,
    log_tokens,
    log_model_info,
    create_comet_experiment,
)


def example_1_basic_opik_setup():
    """Example 1: Basic Opik configuration."""
    print("\n=== Example 1: Basic Opik Setup ===\n")
    
    # Configure Opik
    success = configure_opik()
    
    if success:
        print("✅ Opik configured successfully")
    else:
        print("⚠️  Opik configuration failed (may not be installed or configured)")


def example_2_monitor_rag_query():
    """Example 2: Monitor RAG query."""
    print("\n=== Example 2: Monitor RAG Query ===\n")
    
    @monitor_rag_query(name="example_rag", tags=["example"])
    def query_rag(query: str) -> dict:
        # Simulate retrieval
        time.sleep(0.1)
        documents = [
            {"content": "RAG stands for Retrieval Augmented Generation"},
            {"content": "RAG combines retrieval with generation"},
        ]
        
        # Simulate generation
        time.sleep(0.2)
        answer = "RAG is a technique that combines document retrieval with LLM generation."
        context = "\n".join([doc["content"] for doc in documents])
        
        return {
            "answer": answer,
            "documents": documents,
            "context": context,
            "query": query,
        }
    
    # Execute monitored query
    result = query_rag("What is RAG?")
    print(f"Answer: {result['answer']}")
    print("✅ Query monitored (check Opik dashboard)")


def example_3_monitor_retrieval():
    """Example 3: Monitor document retrieval."""
    print("\n=== Example 3: Monitor Retrieval ===\n")
    
    @monitor_retrieval(name="example_retrieval", tags=["example"])
    def retrieve_documents(query: str, k: int = 5) -> list:
        # Simulate vector search
        time.sleep(0.05)
        
        return [
            {"id": f"doc_{i}", "score": 0.9 - i * 0.1}
            for i in range(k)
        ]
    
    # Execute monitored retrieval
    docs = retrieve_documents("machine learning", k=3)
    print(f"Retrieved {len(docs)} documents")
    print("✅ Retrieval monitored")


def example_4_monitor_inference():
    """Example 4: Monitor LLM inference."""
    print("\n=== Example 4: Monitor Inference ===\n")
    
    @monitor_inference(name="example_inference", tags=["example"])
    def generate_text(prompt: str, temperature: float = 0.7, model_id: str = "gpt-4") -> str:
        # Simulate LLM inference
        time.sleep(0.3)
        return f"Generated response for: {prompt[:50]}..."
    
    # Execute monitored inference
    response = generate_text(
        prompt="Explain quantum computing",
        temperature=0.7,
        model_id="gpt-4o-mini",
    )
    print(f"Response: {response}")
    print("✅ Inference monitored")


def example_5_custom_metrics():
    """Example 5: Log custom metrics."""
    print("\n=== Example 5: Custom Metrics ===\n")
    
    # Log various metrics
    log_metrics({
        "custom_latency_ms": 150,
        "cache_hit_rate": 0.85,
        "num_retries": 2,
        "success": True,
    })
    
    # Log token usage
    log_tokens(
        query_tokens=20,
        context_tokens=500,
        answer_tokens=100,
    )
    
    # Log model information
    log_model_info(
        model_id="meta-llama/Llama-2-7b-hf",
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
    )
    
    print("✅ Custom metrics logged")


def example_6_monitoring_context():
    """Example 6: Monitoring context manager."""
    print("\n=== Example 6: Monitoring Context ===\n")
    
    with MonitoringContext("batch_processing", tags=["batch", "example"]) as monitor:
        # Simulate processing
        items = list(range(100))
        
        time.sleep(0.5)
        
        # Log custom metrics
        monitor.log("items_processed", len(items))
        monitor.log("avg_processing_time_ms", 5)
        monitor.log("errors", 0)
    
    print("✅ Context-based monitoring completed")


def example_7_training_with_comet():
    """Example 7: Training experiment tracking with Comet ML."""
    print("\n=== Example 7: Training with Comet ML ===\n")
    
    with create_comet_experiment(
        experiment_name="example_training_run",
        tags=["example", "tutorial"],
    ) as tracker:
        # Log hyperparameters
        tracker.log_parameters({
            "learning_rate": 3e-4,
            "num_epochs": 3,
            "batch_size": 4,
            "model": "meta-llama/Llama-2-7b-hf",
            "lora_r": 16,
            "lora_alpha": 32,
        })
        
        # Simulate training
        for epoch in range(3):
            # Simulate training step
            time.sleep(0.1)
            loss = 2.5 - epoch * 0.5  # Decreasing loss
            
            # Log metrics
            tracker.log_metric("train_loss", loss, epoch=epoch)
            tracker.log_metric("learning_rate", 3e-4, epoch=epoch)
            
            print(f"Epoch {epoch}: loss={loss:.3f}")
        
        # Log additional info
        tracker.add_tags(["completed"])
        
        print("✅ Training tracked with Comet ML")


def example_8_monitor_data_processing():
    """Example 8: Monitor data processing pipeline."""
    print("\n=== Example 8: Monitor Data Processing ===\n")
    
    @monitor_data_processing(name="chunk_documents", tags=["processing"])
    def chunk_documents(documents: list) -> list:
        # Simulate chunking
        time.sleep(0.2)
        
        chunks = []
        for doc in documents:
            # Split into chunks
            chunks.extend([
                f"{doc}_chunk_{i}"
                for i in range(5)
            ])
        
        return chunks
    
    # Execute monitored processing
    docs = ["doc1", "doc2", "doc3"]
    chunks = chunk_documents(docs)
    
    print(f"Processed {len(docs)} documents into {len(chunks)} chunks")
    print("✅ Data processing monitored")


def example_9_monitor_training_step():
    """Example 9: Monitor training steps."""
    print("\n=== Example 9: Monitor Training Steps ===\n")
    
    @monitor_training_step(name="train_step", tags=["training"])
    def train_step(batch, lr: float = 3e-4) -> dict:
        # Simulate training step
        time.sleep(0.05)
        
        loss = 1.5  # Simulated loss
        
        return {
            "loss": loss,
            "lr": lr,
        }
    
    # Simulate training loop
    for step in range(5):
        result = train_step({"data": "batch"}, lr=3e-4)
        print(f"Step {step}: loss={result['loss']:.3f}")
    
    print("✅ Training steps monitored")


def example_10_end_to_end_rag():
    """Example 10: End-to-end RAG with monitoring."""
    print("\n=== Example 10: End-to-End RAG ===\n")
    
    @monitor_retrieval(name="e2e_retrieve")
    def retrieve(query: str, k: int = 5):
        time.sleep(0.1)
        return [{"content": f"Document {i}"} for i in range(k)]
    
    @monitor_inference(name="e2e_generate")
    def generate(prompt: str, context: str, model_id: str = "gpt-4"):
        time.sleep(0.2)
        return f"Answer based on context: {context[:50]}..."
    
    @monitor_rag_query(name="e2e_rag")
    def rag_pipeline(query: str) -> dict:
        # Retrieve
        docs = retrieve(query, k=5)
        
        # Build context
        context = "\n".join([doc["content"] for doc in docs])
        
        # Generate
        answer = generate(query, context)
        
        return {
            "answer": answer,
            "documents": docs,
            "context": context,
            "query": query,
        }
    
    # Execute pipeline
    result = rag_pipeline("What is machine learning?")
    print(f"Answer: {result['answer']}")
    print("✅ End-to-end RAG monitored")


def example_11_production_monitoring():
    """Example 11: Production monitoring pattern."""
    print("\n=== Example 11: Production Monitoring ===\n")
    
    from hermes.monitoring import is_opik_enabled
    
    @monitor_rag_query(name="production_rag", tags=["production", "critical"])
    def production_rag(query: str) -> dict:
        # Production RAG logic
        time.sleep(0.3)
        
        # Additional logging for production
        if is_opik_enabled():
            log_metrics({
                "environment": "production",
                "version": "1.0.0",
                "cache_enabled": True,
            })
        
        return {
            "answer": "Production answer",
            "documents": [],
            "context": "",
            "query": query,
        }
    
    # Execute with production monitoring
    result = production_rag("Production query")
    print(f"Production answer: {result['answer']}")
    print("✅ Production monitoring active")


def main():
    """Run all examples."""
    print("=" * 80)
    print("MONITORING EXAMPLES")
    print("=" * 80)
    
    examples = [
        example_1_basic_opik_setup,
        example_2_monitor_rag_query,
        example_3_monitor_retrieval,
        example_4_monitor_inference,
        example_5_custom_metrics,
        example_6_monitoring_context,
        example_7_training_with_comet,
        example_8_monitor_data_processing,
        example_9_monitor_training_step,
        example_10_end_to_end_rag,
        example_11_production_monitoring,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("Check your Opik and Comet ML dashboards for tracked data.")
    print("=" * 80)


if __name__ == "__main__":
    main()
