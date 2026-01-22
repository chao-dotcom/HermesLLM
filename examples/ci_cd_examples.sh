#!/bin/bash
# CI/CD Examples for HermesLLM

echo "=== HermesLLM CI/CD Examples ==="

# ============================================================================
# Example 1: Local Development Setup
# ============================================================================
example_local_setup() {
    echo -e "\n=== Example 1: Local Development Setup ===\n"
    
    # Install dependencies
    poetry install
    
    # Install Poe the Poet
    poetry self add 'poethepoet[poetry_plugin]'
    
    # Install pre-commit hooks
    poetry poe pre-commit-install
    
    echo "✅ Local development environment ready!"
}

# ============================================================================
# Example 2: Run QA Checks Locally (before committing)
# ============================================================================
example_qa_checks() {
    echo -e "\n=== Example 2: QA Checks ===\n"
    
    # Lint check
    echo "Running linter..."
    poetry poe lint-check
    
    # Format check
    echo "Running formatter check..."
    poetry poe format-check
    
    # Security scan
    echo "Running security scan..."
    poetry poe gitleaks-check
    
    # Validate configurations
    echo "Validating configurations..."
    poetry poe validate-configs
    
    # Validate settings
    echo "Validating settings..."
    poetry poe validate-settings
    
    echo "✅ All QA checks passed!"
}

# ============================================================================
# Example 3: Fix Lint and Format Issues
# ============================================================================
example_fix_issues() {
    echo -e "\n=== Example 3: Auto-fix Issues ===\n"
    
    # Auto-fix lint issues
    poetry poe lint-fix
    
    # Auto-format code
    poetry poe format-fix
    
    echo "✅ Code fixed and formatted!"
}

# ============================================================================
# Example 4: Run Tests with Coverage
# ============================================================================
example_run_tests() {
    echo -e "\n=== Example 4: Run Tests ===\n"
    
    # Run all tests
    poetry poe test
    
    # Run with HTML coverage report
    poetry poe test-cov
    echo "Open htmlcov/index.html to view coverage report"
    
    # Run verbose
    poetry poe test-verbose
}

# ============================================================================
# Example 5: Pre-commit Hook Workflow
# ============================================================================
example_pre_commit() {
    echo -e "\n=== Example 5: Pre-commit Workflow ===\n"
    
    # Run all pre-commit hooks manually
    poetry poe pre-commit-run
    
    # Or just commit (hooks run automatically)
    echo "git add ."
    echo "git commit -m 'feat: add new feature'"
    echo "# Hooks will run automatically"
}

# ============================================================================
# Example 6: Build and Test Docker Image
# ============================================================================
example_docker_workflow() {
    echo -e "\n=== Example 6: Docker Workflow ===\n"
    
    # Lint Dockerfile
    poetry poe lint-check-docker
    
    # Build image
    poetry poe build-docker
    
    # Test interactively
    poetry poe bash-docker
    
    # Run pipeline in container
    poetry poe run-docker-pipeline
}

# ============================================================================
# Example 7: Local Infrastructure Setup
# ============================================================================
example_infrastructure() {
    echo -e "\n=== Example 7: Local Infrastructure ===\n"
    
    # Start all services
    poetry poe local-infrastructure-up
    
    # Check services
    docker ps
    zenml status
    
    # Set ZenML stack
    poetry poe set-local-stack
    
    # Stop services when done
    poetry poe local-infrastructure-down
}

# ============================================================================
# Example 8: Complete CI Pipeline Simulation
# ============================================================================
example_ci_simulation() {
    echo -e "\n=== Example 8: CI Pipeline Simulation ===\n"
    
    echo "Step 1: Security Scan"
    poetry poe gitleaks-check
    
    echo "Step 2: Code Quality"
    poetry poe lint-check
    poetry poe format-check
    
    echo "Step 3: Tests"
    poetry poe test
    
    echo "Step 4: Docker Lint"
    poetry poe lint-check-docker
    
    echo "Step 5: Config Validation"
    poetry poe validate-configs
    poetry poe validate-settings
    
    echo "Step 6: Build Test"
    poetry build
    
    echo "✅ All CI checks passed!"
}

# ============================================================================
# Example 9: Feature Branch Workflow
# ============================================================================
example_feature_workflow() {
    echo -e "\n=== Example 9: Feature Branch Workflow ===\n"
    
    # Create feature branch
    git checkout -b feature/my-feature
    
    # Make changes
    echo "# Make your code changes..."
    
    # Run QA checks
    poetry poe lint-check
    poetry poe format-check
    poetry poe test
    
    # Commit (pre-commit hooks run)
    git add .
    git commit -m "feat: add new feature"
    
    # Push (triggers CI)
    git push origin feature/my-feature
    
    echo "Create PR on GitHub - CI will run automatically"
}

# ============================================================================
# Example 10: Release Workflow
# ============================================================================
example_release_workflow() {
    echo -e "\n=== Example 10: Release Workflow ===\n"
    
    # Ensure on main branch
    git checkout main
    git pull origin main
    
    # Merge develop
    git merge develop
    
    # Run full checks
    poetry poe lint-check
    poetry poe format-check
    poetry poe test
    poetry build
    
    # Push to main (triggers CD)
    git push origin main
    
    # Create GitHub release (triggers PyPI publish)
    echo "Create release on GitHub:"
    echo "- Tag: v1.0.0"
    echo "- Title: Release 1.0.0"
    echo "- Description: Release notes..."
    
    echo "CD pipeline will:"
    echo "1. Build and push Docker image to ECR"
    echo "2. Publish package to PyPI"
    echo "3. Deploy documentation"
    echo "4. Send Slack notification"
}

# ============================================================================
# Example 11: Manual CD Trigger
# ============================================================================
example_manual_cd() {
    echo -e "\n=== Example 11: Manual CD Trigger ===\n"
    
    echo "Trigger CD manually from GitHub:"
    echo "1. Go to Actions tab"
    echo "2. Select 'CD' workflow"
    echo "3. Click 'Run workflow'"
    echo "4. Select branch: main"
    echo "5. Toggle 'Deploy to AWS ECR': true/false"
    echo "6. Click 'Run workflow'"
}

# ============================================================================
# Example 12: Troubleshooting Failed CI
# ============================================================================
example_troubleshooting() {
    echo -e "\n=== Example 12: Troubleshooting ===\n"
    
    echo "If lint fails:"
    poetry poe lint-fix
    poetry poe format-fix
    git add .
    git commit -m "fix: lint issues"
    
    echo "If tests fail:"
    poetry poe test-verbose
    poetry poe test-cov
    # Open htmlcov/index.html
    
    echo "If Docker build fails:"
    poetry poe build-docker
    poetry poe lint-check-docker
    
    echo "If pre-commit fails:"
    poetry poe pre-commit-run
    # Or skip in emergency
    git commit --no-verify -m "fix: urgent hotfix"
}

# ============================================================================
# Example 13: Clean Build Artifacts
# ============================================================================
example_cleanup() {
    echo -e "\n=== Example 13: Cleanup ===\n"
    
    # Remove all build artifacts
    poetry poe clean
    
    # Rebuild from scratch
    poetry install
    poetry build
}

# ============================================================================
# Example 14: AWS Deployment
# ============================================================================
example_aws_deployment() {
    echo -e "\n=== Example 14: AWS Deployment ===\n"
    
    # Set AWS stack
    poetry poe set-aws-stack
    
    # Create SageMaker role
    poetry poe create-sagemaker-role
    
    # Deploy inference endpoint
    poetry poe deploy-inference-endpoint
    
    # Test endpoint
    hermes-serve
    curl -X POST http://localhost:8000/v1/inference \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello, world!"}'
    
    # Delete when done
    poetry poe delete-inference-endpoint
}

# ============================================================================
# Main Menu
# ============================================================================
main() {
    echo "Select an example to run:"
    echo "1. Local Development Setup"
    echo "2. Run QA Checks"
    echo "3. Fix Lint and Format Issues"
    echo "4. Run Tests with Coverage"
    echo "5. Pre-commit Workflow"
    echo "6. Docker Workflow"
    echo "7. Local Infrastructure"
    echo "8. CI Pipeline Simulation"
    echo "9. Feature Branch Workflow"
    echo "10. Release Workflow"
    echo "11. Manual CD Trigger"
    echo "12. Troubleshooting"
    echo "13. Cleanup"
    echo "14. AWS Deployment"
    
    read -p "Enter example number (1-14): " choice
    
    case $choice in
        1) example_local_setup ;;
        2) example_qa_checks ;;
        3) example_fix_issues ;;
        4) example_run_tests ;;
        5) example_pre_commit ;;
        6) example_docker_workflow ;;
        7) example_infrastructure ;;
        8) example_ci_simulation ;;
        9) example_feature_workflow ;;
        10) example_release_workflow ;;
        11) example_manual_cd ;;
        12) example_troubleshooting ;;
        13) example_cleanup ;;
        14) example_aws_deployment ;;
        *) echo "Invalid choice" ;;
    esac
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" -eq "${0}" ]; then
    main
fi
