---
description: Deploy CI/CD pipeline configuration for Claude Agents Pro project
args: [--provider github|gitlab] [--environment dev|staging|prod] [--dry-run]
tools: Write, Bash(git:*), Bash(gh:*), Bash(glab:*)
model: claude-sonnet-4-5
---

## Objective
Generate and deploy CI/CD pipeline configuration tailored for Claude Agents Pro repository automation.

## Before You Run
- Ensure you have appropriate repository permissions
- Review target environment configuration
- Verify required secrets and environment variables are available

## Execution
Generate pipeline configuration:

```bash
# GitHub Actions (default)
!/deploy-pipeline --provider github --environment $1

# GitLab CI with dry-run
!/deploy-pipeline --provider gitlab --environment staging --dry-run
```

## Pipeline Features
- **Agent Validation**: Runs verify-agents and quality scoring
- **Security Scanning**: SAST analysis of agent configurations
- **Automated Testing**: Validates agent functionality
- **Deployment Gates**: Environment-specific approval workflows
- **Rollback Capability**: Automatic reversion on failure

## Follow Up
- Monitor pipeline execution in your CI/CD platform
- Review deployment logs for any validation failures
- Configure branch protection rules and approval workflows
- Set up notification channels for pipeline status

## Pipeline Components
1. **Validation Stage**: Agent syntax and quality checks
2. **Security Stage**: Security policy validation
3. **Test Stage**: Functional agent testing
4. **Deploy Stage**: Agent installation and configuration
5. **Monitoring Stage**: Health checks and metrics collection
