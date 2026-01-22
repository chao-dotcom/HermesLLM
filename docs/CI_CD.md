# CI/CD & DevOps Documentation

## Overview

HermesLLM includes a comprehensive CI/CD pipeline built with GitHub Actions, pre-commit hooks, and automated task management with Poe the Poet.

## Architecture

```
.github/
├── workflows/
│   ├── ci.yaml          # Continuous Integration
│   └── cd.yaml          # Continuous Deployment
.pre-commit-config.yaml  # Pre-commit hooks
pyproject.toml           # Poe tasks configuration
```

## Continuous Integration (CI)

### CI Pipeline Jobs

The CI pipeline runs on every pull request and push to `develop` branch:

#### 1. Security Scan
- **Tool**: Gitleaks
- **Purpose**: Detect secrets, API keys, and credentials
- **Runs**: On all commits with full git history
- **Action**: `gitleaks/gitleaks-action@v2`

#### 2. Code Quality
- **Linting**: Ruff (PEP 8 compliance, imports, complexity)
- **Formatting**: Ruff formatter (consistent code style)
- **Caching**: Poetry dependencies cached for faster runs
- **Commands**:
  ```bash
  poetry poe lint-check
  poetry poe format-check
  ```

#### 3. Test Suite
- **Matrix Testing**: Python 3.10 and 3.11
- **Coverage**: Generated with pytest-cov
- **Upload**: Coverage reports to Codecov
- **Test Environment**: Uses `.env.testing`
- **Command**:
  ```bash
  poetry poe test
  ```

#### 4. Docker Lint
- **Tool**: Hadolint
- **Purpose**: Lint Dockerfile for best practices
- **Threshold**: Warning level
- **Action**: `hadolint/hadolint-action@v3.1.0`

#### 5. Validate Configurations
- **YAML Validation**: All config files in `configs/`
- **Settings Validation**: HermesLLM settings check
- **Commands**:
  ```bash
  poetry poe validate-configs
  hermes-settings validate
  ```

#### 6. Build Test
- **Purpose**: Ensure package builds successfully
- **Output**: Wheel and source distribution
- **Artifacts**: Uploaded for 7 days
- **Command**:
  ```bash
  poetry build
  ```

### CI Workflow Triggers

```yaml
on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [develop]
```

### CI Performance Optimizations

1. **Dependency Caching**: Poetry virtualenv cached by `poetry.lock` hash
2. **Concurrency Control**: Cancels previous runs on new commits
3. **Parallel Jobs**: Independent jobs run in parallel
4. **Matrix Strategy**: Multi-version Python testing

## Continuous Deployment (CD)

### CD Pipeline Jobs

The CD pipeline runs on pushes to `main`, releases, or manual triggers:

#### 1. Build & Push Docker Image
- **Platform**: linux/amd64
- **Registry**: AWS ECR
- **Tags**:
  - Branch name (e.g., `main`)
  - Git SHA with prefix (e.g., `main-abc123`)
  - Semantic version (e.g., `1.0.0`, `1.0`)
  - `latest` for main branch
- **Caching**: GitHub Actions cache for layers
- **Build Args**:
  - `BUILD_DATE`: Repository update time
  - `VCS_REF`: Git commit SHA
  - `VERSION`: Git ref name

**Example Tags**:
```
123456789.dkr.ecr.us-east-1.amazonaws.com/hermesllm:main
123456789.dkr.ecr.us-east-1.amazonaws.com/hermesllm:main-abc1234
123456789.dkr.ecr.us-east-1.amazonaws.com/hermesllm:latest
```

#### 2. Publish Package (on release)
- **Registry**: PyPI
- **Trigger**: GitHub release published
- **Authentication**: `PYPI_TOKEN` secret
- **Command**:
  ```bash
  poetry publish
  ```

#### 3. Deploy Documentation (on main)
- **Target**: GitHub Pages
- **Builder**: MkDocs (if configured)
- **Trigger**: Push to main branch
- **Action**: `peaceiris/actions-gh-pages@v3`

#### 4. Notify Deployment
- **Status Check**: Success/failure of Docker build
- **Outputs**: Image tag and digest
- **Slack Integration**: Optional webhook notification
- **Payload**: Deployment status, branch, commit, image info

### CD Workflow Triggers

```yaml
on:
  push:
    branches: [main]
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      deploy_to_ecr:
        type: boolean
        default: true
```

### Manual Deployment

Trigger CD manually from GitHub Actions UI:
1. Go to Actions → CD workflow
2. Click "Run workflow"
3. Select branch
4. Toggle "Deploy to AWS ECR" (optional)

## Pre-commit Hooks

### Installation

```bash
# Install pre-commit
poetry add --group dev pre-commit

# Install hooks
poetry poe pre-commit-install
# or
pre-commit install

# Run manually
poetry poe pre-commit-run
# or
pre-commit run --all-files
```

### Configured Hooks

1. **Ruff Linter** (`ruff`)
   - Auto-fixes common issues
   - Checks PEP 8 compliance
   - Organizes imports

2. **Ruff Formatter** (`ruff-format`)
   - Formats code consistently
   - Line length: 120 characters

3. **Gitleaks** (security)
   - Scans for secrets in commits
   - Prevents credential leaks

4. **Pre-commit Standard Hooks**
   - Trailing whitespace removal
   - End-of-file fixer
   - YAML/JSON/TOML syntax check
   - Large file detection (>1MB)
   - Merge conflict detection
   - Private key detection

5. **Poetry Check**
   - Validates `pyproject.toml`
   - Checks lock file consistency

6. **Codespell**
   - Detects common typos
   - Custom ignore list

7. **Validate Settings** (local hook)
   - Runs on config file changes
   - Command: `hermes-settings validate`
   - Files: `hermes/config.py`, `.env*`, `configs/*.yaml`

### Hook Execution Flow

```
git commit
    ↓
Ruff Linter (auto-fix)
    ↓
Ruff Formatter
    ↓
Gitleaks Scan
    ↓
File Checks (whitespace, EOF, etc.)
    ↓
Syntax Validation (YAML, JSON, TOML)
    ↓
Poetry Check
    ↓
Spelling Check
    ↓
Settings Validation (if config changed)
    ↓
Commit Success ✅
```

## Poe the Poet Tasks

### Installation

```bash
poetry self add 'poethepoet[poetry_plugin]'
```

### QA Tasks

```bash
# Linting
poetry poe lint-check          # Check code quality
poetry poe lint-fix            # Auto-fix issues
poetry poe format-check        # Check formatting
poetry poe format-fix          # Auto-format code
poetry poe gitleaks-check      # Security scan
poetry poe lint-check-docker   # Lint Dockerfile
```

### Testing Tasks

```bash
# Run tests
poetry poe test                # Standard test run
poetry poe test-cov            # With HTML coverage report
poetry poe test-verbose        # Verbose output
```

### Pipeline Tasks

```bash
# End-to-end
poetry poe run-end-to-end      # Complete pipeline

# Individual pipelines
poetry poe run-collect         # Data collection
poetry poe run-process         # Document processing
poetry poe run-generate-dataset # Dataset generation
poetry poe run-train           # Model training
poetry poe run-evaluate        # Model evaluation
```

### Data Warehouse Tasks

```bash
poetry poe export-data-warehouse  # Export to JSON
poetry poe import-data-warehouse  # Import from JSON
poetry poe list-collections       # List all collections
poetry poe clear-collection       # Clear specific collection
```

### Settings Tasks

```bash
poetry poe export-settings     # Export to ZenML
poetry poe import-settings     # Import from ZenML
poetry poe validate-settings   # Validate configuration
poetry poe show-settings       # Display settings
```

### Inference & RAG Tasks

```bash
poetry poe run-inference-service  # Start FastAPI server
poetry poe call-rag              # Query RAG system
poetry poe rag-demo              # Run RAG demo
```

### Infrastructure Tasks

```bash
# Local infrastructure
poetry poe local-docker-up          # Start Docker services
poetry poe local-docker-down        # Stop Docker services
poetry poe local-zenml-up           # Start ZenML server
poetry poe local-zenml-down         # Stop ZenML server
poetry poe local-infrastructure-up  # Start all services
poetry poe local-infrastructure-down # Stop all services

# ZenML stacks
poetry poe set-local-stack     # Switch to local stack
poetry poe set-aws-stack       # Switch to AWS stack

# AWS SageMaker
poetry poe create-sagemaker-role      # Create IAM role
poetry poe deploy-inference-endpoint  # Deploy to SageMaker
poetry poe delete-inference-endpoint  # Delete endpoint
```

### Docker Tasks

```bash
poetry poe build-docker           # Build Docker image
poetry poe run-docker-pipeline    # Run pipeline in container
poetry poe bash-docker            # Interactive container shell
```

### Configuration Tasks

```bash
poetry poe validate-configs    # Validate all YAML configs
```

### Documentation Tasks

```bash
poetry poe build-docs          # Build MkDocs documentation
poetry poe serve-docs          # Serve docs locally
```

### Cleanup Tasks

```bash
poetry poe clean              # Remove build artifacts, cache, coverage
```

## GitHub Secrets Configuration

### Required Secrets

#### AWS Deployment
```
AWS_ACCESS_KEY_ID          # AWS access key
AWS_SECRET_ACCESS_KEY      # AWS secret key
AWS_REGION                 # AWS region (e.g., us-east-1)
AWS_ACCOUNT_ID             # AWS account ID
```

#### Optional Secrets
```
GITLEAKS_LICENSE           # Gitleaks Pro license (optional)
CODECOV_TOKEN              # Codecov upload token
PYPI_TOKEN                 # PyPI publishing token
SLACK_WEBHOOK_URL          # Slack notifications
```

### Setting Secrets

1. Go to Repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add name and value
4. Click "Add secret"

## Best Practices

### Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Install pre-commit hooks** (one-time)
   ```bash
   poetry poe pre-commit-install
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # Pre-commit hooks run automatically
   ```

4. **Run tests locally**
   ```bash
   poetry poe test
   poetry poe lint-check
   poetry poe format-check
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   # CI pipeline runs automatically
   ```

6. **Merge to develop**
   - CI passes ✅
   - Code review approved
   - Merge PR

7. **Release to main**
   ```bash
   git checkout main
   git merge develop
   git push origin main
   # CD pipeline deploys automatically
   ```

### CI/CD Optimization Tips

1. **Use cached dependencies**: Speeds up CI by 2-3x
2. **Run lint locally**: Catch issues before CI
3. **Write focused tests**: Faster test execution
4. **Keep Docker images small**: Faster builds and deploys
5. **Use workflow concurrency**: Prevent redundant runs

### Troubleshooting

#### CI Failing on Lint
```bash
# Fix locally
poetry poe lint-fix
poetry poe format-fix
git add .
git commit -m "fix: lint issues"
```

#### CI Failing on Tests
```bash
# Run tests locally
poetry poe test-verbose

# Check coverage
poetry poe test-cov
# Open htmlcov/index.html
```

#### Pre-commit Hook Failing
```bash
# Skip hooks (emergency only)
git commit --no-verify -m "fix: urgent hotfix"

# Or fix issues
poetry poe pre-commit-run
```

#### Docker Build Failing
```bash
# Test locally
poetry poe build-docker

# Check Dockerfile
poetry poe lint-check-docker

# Interactive debug
poetry poe bash-docker
```

## Monitoring & Notifications

### Build Status Badges

Add to README.md:

```markdown
![CI](https://github.com/USERNAME/HermesLLM/workflows/CI/badge.svg)
![CD](https://github.com/USERNAME/HermesLLM/workflows/CD/badge.svg)
[![codecov](https://codecov.io/gh/USERNAME/HermesLLM/branch/main/graph/badge.svg)](https://codecov.io/gh/USERNAME/HermesLLM)
```

### Slack Notifications

Configure Slack webhook in repository secrets:
1. Create Slack incoming webhook
2. Add `SLACK_WEBHOOK_URL` to GitHub secrets
3. CD workflow automatically sends notifications

**Notification Content**:
- Deployment status (success/failure)
- Branch name
- Commit SHA
- Docker image tag and digest

## Examples

See [examples/ci_cd_examples.sh](../examples/ci_cd_examples.sh) for complete usage examples.

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Hooks](https://pre-commit.com/)
- [Poe the Poet](https://poethepoet.natn.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Gitleaks](https://github.com/gitleaks/gitleaks)
