FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy application code
COPY atlas/ ./atlas/
COPY workflows/ ./workflows/
COPY tasks/ ./tasks/
COPY configs/ ./configs/

# Install the package
RUN poetry install --no-interaction --no-ansi

# Create data directory
RUN mkdir -p /app/data

# Expose API port
EXPOSE 8000

# Default command
CMD ["atlas", "--help"]
