# 🌍 HermesLLM

HermesLLM is a distributed LLM platform designed to ingest knowledge from heterogeneous sources—including GitHub repositories, professional writing platforms, and long-form transcripts—and deliver educational content that preserves authorial voice while remaining fact-aligned. The system emphasizes traceability, controlled style conditioning, and production-grade deployment through independently scalable training and real-time inference pipelines.

## 🚀 Features

- **📊 Data Collection**: Multi-source data ingestion from LinkedIn, Medium, GitHub
- **🔄 Processing Pipeline**: Cleaning, chunking, and embedding generation
- **🎯 RAG System**: Advanced retrieval with query expansion and reranking
- **🧠 Model Training**: Fine-tuning pipelines with evaluation
- **☁️ Cloud Deployment**: AWS SageMaker integration
- **📡 API Service**: FastAPI-based inference endpoints
- **🔍 Monitoring**: Comprehensive tracking with Opik
- **⚡ Orchestration**: ZenML-powered workflow management

## 📂 Project Structure

```
atlas-llm/
├── atlas/              # Main package
│   ├── core/          # Domain entities
│   ├── collectors/    # Data collection
│   ├── processing/    # Data processing
│   ├── storage/       # Data persistence
│   ├── rag/           # RAG system
│   ├── training/      # Model training
│   ├── inference/     # Model inference
│   ├── cloud/         # Cloud integrations
│   └── utils/         # Utilities
├── workflows/         # Pipeline orchestration
├── tasks/            # Workflow steps
├── api/              # API service
├── cli/              # Command-line interface
└── configs/          # Configuration files
```

## 🛠️ Installation

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

## 🎯 Quick Start

### 1. Data Collection

```bash
# Collect data from configured sources
atlas collect --source all

# Collect from specific source
atlas collect --source linkedin
```

### 2. Process Data

```bash
# Run feature engineering pipeline
atlas process --pipeline features

# Generate embeddings and load to vector DB
atlas process --pipeline embeddings
```

### 3. Train Model

```bash
# Generate training datasets
atlas train --generate-datasets

# Run training pipeline
atlas train --run
```

### 4. Deploy & Serve

```bash
# Deploy to AWS SageMaker
atlas deploy --cloud aws

# Run local API server
atlas serve --port 8000
```

### 5. Query RAG System

```bash
# Interactive RAG query
atlas query "How to implement RAG systems?"

# RAG with API
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'
```

## 🔧 Configuration

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
DATABASE_NAME=atlas

# Vector DB
QDRANT_DATABASE_HOST=localhost
QDRANT_DATABASE_PORT=6333

# AWS (optional)
AWS_REGION=us-east-1
AWS_ACCESS_KEY=your_key
AWS_SECRET_KEY=your_secret
```

## 📚 Usage Examples

### Programmatic Usage

```python
from atlas.rag import RAGSystem
from atlas.config import settings

# Initialize RAG system
rag = RAGSystem(config=settings)

# Query
result = rag.query(
    "Explain vector databases",
    k=5,
    use_reranking=True
)

print(result.answer)
print(result.sources)
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

## 🧪 Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=atlas

# Run specific test
poetry run pytest tests/unit/test_rag.py
```

## 🐳 Docker

```bash
# Build image
docker build -t atlas-llm .

# Run with docker-compose
docker-compose up -d

# Run specific pipeline
docker-compose run atlas atlas collect --source all
```

## 📊 Monitoring

View pipeline runs and metrics in:
- **ZenML Dashboard**: Pipeline orchestration
- **Opik**: LLM monitoring and tracing
- **Comet ML**: Experiment tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details