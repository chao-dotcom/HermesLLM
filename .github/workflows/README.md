# GitHub Actions Workflows

This directory contains automated CI/CD workflows for HermesLLM.

## Workflows

### CI (Continuous Integration)
**File**: `ci.yaml`  
**Triggers**: Pull requests to `main`/`develop`, pushes to `develop`

**Jobs**:
1. **Security Scan** - Gitleaks secret detection
2. **Code Quality** - Ruff linting and formatting
3. **Test Suite** - Pytest with coverage (Python 3.10 & 3.11)
4. **Docker Lint** - Hadolint Dockerfile validation
5. **Validate Configs** - YAML and settings validation
6. **Build Test** - Package build verification

### CD (Continuous Deployment)
**File**: `cd.yaml`  
**Triggers**: Push to `main`, releases, manual dispatch

**Jobs**:
1. **Build & Push Docker** - Multi-tag ECR deployment
2. **Publish Package** - PyPI publishing (releases only)
3. **Deploy Docs** - GitHub Pages deployment
4. **Notify Deployment** - Slack notifications

## Usage

### Viewing Workflow Runs
1. Go to the "Actions" tab in GitHub
2. Select a workflow (CI or CD)
3. View run history and logs

### Manual CD Trigger
1. Go to Actions → CD workflow
2. Click "Run workflow"
3. Select branch and options
4. Click "Run workflow" button

### Required Secrets
Configure in Repository Settings → Secrets:

```
AWS_ACCESS_KEY_ID          # Required for ECR
AWS_SECRET_ACCESS_KEY      # Required for ECR
AWS_REGION                 # Required for ECR
AWS_ACCOUNT_ID             # Required for ECR
CODECOV_TOKEN              # Optional for coverage
PYPI_TOKEN                 # Required for PyPI publish
SLACK_WEBHOOK_URL          # Optional for notifications
```

## Status Badges

Add to README.md:
```markdown
![CI](https://github.com/USERNAME/HermesLLM/workflows/CI/badge.svg)
![CD](https://github.com/USERNAME/HermesLLM/workflows/CD/badge.svg)
```

## Documentation

See [docs/CI_CD.md](../../docs/CI_CD.md) for complete documentation.
