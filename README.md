# ?? HermesLLM

HermesLLM is a distributed LLM platform designed to ingest knowledge from heterogeneous sources?�including GitHub repositories, professional writing platforms, and long-form transcripts?�and deliver educational content that preserves authorial voice while remaining fact-aligned. The system emphasizes traceability, controlled style conditioning, and production-grade deployment through independently scalable training and real-time inference pipelines.

## ?? Features

- **?? Data Collection**: Multi-source data ingestion from LinkedIn, Medium, GitHub
- **?? Processing Pipeline**: Cleaning, chunking, and embedding generation
- **?�� RAG System**: Advanced retrieval with query expansion and reranking
- **?? AI-Powered Dataset Generation**: Automated instruction and preference dataset creation
- **?? Advanced Training**: LoRA/DPO fine-tuning with 4-bit quantization and Unsloth
- **?? Model Training**: Production-grade SFT and DPO training pipelines
- **☁️ AWS SageMaker**: Complete deployment, inference, and endpoint management
- **?�� API Service**: FastAPI-based inference endpoints
- **?? Monitoring**: Comprehensive tracking with Opik
- **??Orchestration**: ZenML-powered workflow management

## ?? Project Structure

```
hermes-llm/
?��??� hermes/              # Main package
??  ?��??� core/          # Domain entities
??  ?��??� collectors/    # Data collection
??  ?��??� processing/    # Data processing
??  ?��??� storage/       # Data persistence
??  ?��??� datasets/      # AI dataset generation
??  ?��??� rag/           # RAG system
??  ?��??� training/      # Model training (basic + advanced)
??  ?��??� inference/     # Model inference
??  ?��??� api/           # API service
??  ?��??� utils/         # Utilities
?��??� workflows/         # Pipeline orchestration
?��??� tasks/            # Workflow steps (collection, datasets, training)
?��??� examples/         # Usage examples
?��??� docs/             # Documentation
?��??� configs/          # Configuration files
```

## ??�?Installation

### Prerequisites

- Python 3.11
- Poetry >= 1.8.3
- Docker >= 27.1.1
- AWS CLI >= 2.15.42

### Setup

```bash
# Install dependencies
poetry install

# Install with cloud support
poetry install --with cloud

# Setup environment
cp .env.example .env
# Edit .env with your credentials
```

## ?�� Quick Start

### 1. Data Collection

```bash
# Collect data from configured sources
HERMES collect --source all

# Collect from specific source
HERMES collect --source linkedin
```

### 2. Process Data

```bash
# Run feature engineering pipeline
HERMES process --pipeline features

# Generate embeddings and load to vector DB
HERMES process --pipeline embeddings
```

### 3. Generate Training Datasets

```bash
# Generate instruction dataset from your documents
python examples/generate_datasets_example.py

# Or use the pipeline directly
from workflows.pipelines import generate_datasets_pipeline

generate_datasets_pipeline(
    5ataset_name="my-instruction-dataset",
    num_samples=100,
)
```

### 4. Train Model

```bash
# Supervised Fine-Tuning with LoRA
python examples/advanced_training_example.py

# Or use the training pipeline
from workflows.pipelines.training import sft_training_pipeline

sft_training_pipeline(
    dataset_path="username/my-instruction-dataset",
    model_name="meta-llama/Llama-3.2-1B",
    output_dir="models/my-sft-model",
)
```

### 4. Deploy & Serve

```bash
# Deploy to AWS SageMaker
python examples/sagemaker_deployment_example.py

# Or use the deployment pipeline
from workflows.pipelines.deployment import sagemaker_deployment_pipeline

sagemaker_deployment_pipeline(
    model_path="models/my-sft-model",
    model_name="my-llm-model",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    s3_bucket="my-sagemaker-bucket",
)

# Run local API server
python -m hermes.api.app
```

### 6. Query RAG System

```bash
# Interactive RAG query
HERMES query "How to implement RAG systems?"

# RAG with API
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'
```

## ?�� Configuration

Configuration files are in `configs/`:

- `data_collection.yaml` - Data source configuration
- `training.yaml` - Training parameters
- `deployment.yaml` - Deployment settings

Environment variables in `.env`:

```env
# API Keys
OPENAI_API_KEY=your_key
HUGGINGFACE_ACCESS_TOKEN=your_token
COMET_API_KEY=your_key

# Database
DATABASE_HOST=mongodb://localhost:27017
DATABASE_NAME=HERMES

# Vector DB
QDRANT_DATABASE_HOST=localhost
QDRANT_DATABASE_PORT=6333

# AWS (optional)
AWS_REGION=us-east-1
AWS_ACCESS_KEY=your_key
AWS_SECRET_KEY=your_secret
```

## ?? Usage Examples

### RAG System

```python
from hermes.rag import RAGPipeline

# Initialize RAG system
rag = RAGPipeline()

# Query
result = rag.generate_answer(
    query="Explain vector databases",
    top_k=5,
)

print(result["answer"])
print(result["context"])
```

### Generate Training Datasets

```pythonTraining Pipeline

```python
from zenml import pipeline
from tasks.training import load_training_dataset, train_sft, push_model_to_hub

@pipeline
def my_training_pipeline(dataset_path: str, hub_repo_id: str):
    # Load dataset
    dataset = load_training_dataset(dataset_path=dataset_path)
    
    # Train model
    metrics = train_sft(
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        model_name="meta-llama/Llama-3.2-1B",
        output_dir="models/my-model",
    )
    
    # Push to HuggingFace Hub
    push_model_to_hub(
        model_path="models/my-model",
        repo_id=hub_repo_id,
    )

# Run pipeline
my_training_pipeline(
    dataset_path="username/my-dataset",
    hub_repo_id="username/my-finetuned-model",
erator.generate_dataset(
    documents=documents,
    num_samples=100,
)

# Save to HuggingFace
dataset.push_to_hub("username/my-dataset")
```

### Advanced Training

```python
from hermes.training.advanced import TrainingConfig, SFTTrainer

# Configure training
config = TrainingConfig(
    model_name="meta-llama/Llama-3.2-1B",
    output_dir="models/my-model",
    dataset_path="username/my-dataset",
    load_in_4bit=True,
    num_train_epochs=3,
)

# Train with LoRA
trainer = SFTTrainer(config)
metrics = trainer.train()
```

### SageMaker Deployment

```python
from hermes.deployment import SageMakerConfig, SageMakerDeployer, SageMakerInferenceClient

# Deploy model to SageMaker
config = SageMakerConfig(
    region="us-east-1",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
    model_path="models/my-model",
    model_name="my-llm",
    instance_type="ml.g5.2xlarge",
    s3_bucket="my-sagemaker-bucket",
)

deployer = SageMakerDeployer(config)
endpoint = deployer.deploy(wait=True)

# Invoke endpoint
client = SageMakerInferenceClient(endpoint_name=endpoint)
response = client.generate("Explain machine learning:")
```

### Custom Pipeline

```python
from workflows import DataPipeline
from tasks.collection import CollectTask
from tasks.processing import CleanTask, EmbedTask

# Build custom workflow
pipeline = DataPipeline([
    CollectTask(source="medium"),
    CleanTask(),
    EmbedTask(model="all-MiniLM-L6-v2")
])

# Execute
pipeline.run()
```

## ?�� Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=HERMES

# Run specific test
poetry run pytest tests/unit/test_rag.py
```

## ?�� Docker

```bash
# Build image
docker build -t hermes-llm .

# Run with docker-compose
docker-compose up -d

# Run specific pipeline
docker-compose run HERMES HERMES collect --source all
```

## ?? Monitoring

View pipeline runs and metrics in:
- **ZenML Dashboard**: Pipeline orchestration
- **Opik**: LLM monitoring and tracing
- **Comet ML**: Experiment tracking

## ?? Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## ?? License

MIT License - see LICENSE file for details