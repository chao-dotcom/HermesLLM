"""
HermesLLM Tools - Usage Examples

This file demonstrates how to use the various CLI tools in HermesLLM.
"""

# ============================================================================
# 1. Pipeline Runner (hermes-run)
# ============================================================================

# Run complete end-to-end pipeline
"""
python -m hermes.tools.run run end-to-end \
    --author "John Doe" \
    --links https://medium.com/@johndoe/article-1 \
    --links https://github.com/johndoe \
    --platforms medium github \
    --samples 100 \
    --finetuning-type sft \
    --base-model meta-llama/Llama-2-7b-hf \
    --dummy
"""

# Run individual pipelines
"""
# Data Collection
python -m hermes.tools.run run collect \
    --author "John Doe" \
    --links https://medium.com/@johndoe/article \
    --platforms medium

# Document Processing  
python -m hermes.tools.run run process \
    --authors "John Doe" \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --tokens-per-chunk 512

# Dataset Generation
python -m hermes.tools.run run generate-dataset \
    --type instruction \
    --samples 100 \
    --model gpt-4o-mini \
    --authors "John Doe"

# Model Training
python -m hermes.tools.run run train \
    --model meta-llama/Llama-2-7b-hf \
    --dataset username/dataset-name \
    --finetuning-type sft \
    --epochs 3 \
    --use-lora \
    --dummy

# Model Evaluation
python -m hermes.tools.run run evaluate \
    --model username/model-name \
    --metrics g-eval accuracy \
    --benchmarks mmlu gsm8k
"""

# Run from YAML config
"""
python -m hermes.tools.run run end-to-end \
    --config configs/my_pipeline.yaml \
    --no-cache
"""


# ============================================================================
# 2. Data Warehouse Management (hermes-warehouse)
# ============================================================================

# Export data warehouse to JSON
"""
python -m hermes.tools.data_warehouse export \
    --output-dir data/backup \
    --collections documents cleaned_documents chunks
"""

# Import data warehouse from JSON
"""
python -m hermes.tools.data_warehouse import \
    --input-dir data/backup \
    --overwrite
"""

# List all collections
"""
python -m hermes.tools.data_warehouse list
"""

# Clear a collection (with confirmation)
"""
python -m hermes.tools.data_warehouse clear \
    --collection documents \
    --confirm
"""


# ============================================================================
# 3. RAG Demonstration (hermes-rag)
# ============================================================================

# Query the RAG system
"""
python -m hermes.tools.rag_demo query \
    -q "What is machine learning?" \
    -k 5 \
    --expand \
    --rerank \
    --verbose
"""

# Retrieve documents only (no generation)
"""
python -m hermes.tools.rag_demo retrieve \
    -q "deep learning" \
    -k 10 \
    --expand
"""

# Interactive RAG mode
"""
python -m hermes.tools.rag_demo interactive \
    --collection documents \
    -k 5 \
    --rerank
"""

# Run pre-configured demo
"""
python -m hermes.tools.rag_demo demo \
    --author "Paul Iusztin"
"""


# ============================================================================
# 4. ML Inference Service (hermes-serve)
# ============================================================================

# Start the FastAPI inference server
"""
python -m hermes.tools.ml_service

# Or with uvicorn directly
uvicorn hermes.tools.ml_service:app --host 0.0.0.0 --port 8000 --reload
"""

# API Usage Examples:

# Health check
"""
curl http://localhost:8000/health
"""

# Text generation
"""
curl -X POST http://localhost:8000/v1/inference \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Explain machine learning in simple terms",
        "max_tokens": 200,
        "temperature": 0.7
    }'
"""

# RAG query
"""
curl -X POST http://localhost:8000/v1/rag \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What are the benefits of RAG systems?",
        "k": 5,
        "max_tokens": 300
    }'
"""

# Reload model
"""
curl -X POST "http://localhost:8000/v1/reload-model?model_id=meta-llama/Llama-2-13b-hf"
"""


# ============================================================================
# 5. Legacy CLI Commands (hermes cli)
# ============================================================================

# These are the original CLI commands in hermes.cli
"""
# Collect data
hermes collect https://medium.com/@user/article -u user123 -n "John Doe"

# Process data  
hermes process -u user123

# Ingest to vector DB
hermes ingest -u user123

# Query RAG system
hermes query "What is machine learning?" -u user123 -k 5

# Generate dataset
hermes generate-dataset -u user123 --num-samples 100

# Start API server
hermes serve
"""


# ============================================================================
# 6. Combined Workflows
# ============================================================================

# Complete workflow from scratch
"""
# Step 1: Collect data
python -m hermes.tools.run run collect \
    --author "John Doe" \
    --links https://medium.com/@johndoe/article

# Step 2: Process documents
python -m hermes.tools.run run process \
    --authors "John Doe"

# Step 3: Export backup
python -m hermes.tools.data_warehouse export \
    --output-dir data/backup_$(date +%Y%m%d)

# Step 4: Test RAG
python -m hermes.tools.rag_demo query \
    -q "Summarize John Doe's work" \
    --rerank

# Step 5: Generate dataset
python -m hermes.tools.run run generate-dataset \
    --type instruction \
    --samples 100 \
    --authors "John Doe"

# Step 6: Train model (dummy mode)
python -m hermes.tools.run run train \
    --model meta-llama/Llama-2-7b-hf \
    --dataset username/dataset \
    --dummy

# Step 7: Start inference service
python -m hermes.tools.ml_service
"""


# ============================================================================
# 7. Development & Testing
# ============================================================================

# Run pipelines with caching disabled
"""
python -m hermes.tools.run run end-to-end \
    --author "Test User" \
    --dummy \
    --no-cache
"""

# Test RAG with verbose output
"""
python -m hermes.tools.rag_demo query \
    -q "test query" \
    --verbose
"""

# Quick data backup
"""
python -m hermes.tools.data_warehouse export --output-dir /tmp/backup
"""


# ============================================================================
# 8. Production Deployment
# ============================================================================

# Deploy inference service with Docker
"""
# Build image
docker build -t hermesllm:latest .

# Run service
docker run -p 8000:8000 \
    -e MODEL_ID=meta-llama/Llama-2-7b-hf \
    hermesllm:latest

# Health check
curl http://localhost:8000/health
"""

# Run pipelines on schedule (cron example)
"""
# Add to crontab
0 2 * * * cd /path/to/hermesllm && python -m hermes.tools.run run collect --config configs/daily_collection.yaml
0 3 * * * cd /path/to/hermesllm && python -m hermes.tools.run run process --authors "Author1" "Author2"
0 4 * * * cd /path/to/hermesllm && python -m hermes.tools.data_warehouse export --output-dir /backups/$(date +\%Y\%m\%d)
"""
