---
name: ci-cd-patterns
description: Load when user needs CI/CD pipeline patterns, GitHub Actions, deployment strategies, blue-green deployments, canary releases, or infrastructure as code
trigger_keywords: [ci/cd, cicd, continuous integration, continuous deployment, github actions, pipeline, deployment, blue-green, canary, rolling deployment, infrastructure as code, iac, docker, kubernetes]
---

# CI/CD Pipeline Patterns

Production-grade CI/CD pipeline patterns using GitHub Actions, Docker, Kubernetes, and modern deployment strategies.

## Core Concepts

### CI/CD Fundamentals

**Continuous Integration (CI)**:
- Automated testing on every commit
- Fast feedback loops (<10 minutes)
- Build artifacts for deployment
- Quality gates (linting, security scans, tests)

**Continuous Deployment (CD)**:
- Automated deployment to production
- Progressive rollout strategies
- Automated rollback on failure
- Zero-downtime deployments

### Pipeline Stages

```yaml
1. Build
   - Compile code
   - Build Docker images
   - Generate artifacts

2. Test
   - Unit tests
   - Integration tests
   - E2E tests

3. Security
   - Dependency scanning
   - SAST (Static Application Security Testing)
   - Container scanning

4. Deploy
   - Deploy to staging
   - Run smoke tests
   - Deploy to production
   - Post-deployment verification
```

## GitHub Actions Patterns

### 1. Complete CI Pipeline

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '20.x'
  DOCKER_REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  lint:
    name: Lint Code
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Check formatting
        run: npm run format:check

  test:
    name: Run Tests
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpassword
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run unit tests
        run: npm run test:unit

      - name: Run integration tests
        run: npm run test:integration
        env:
          DATABASE_URL: postgresql://postgres:testpassword@localhost:5432/testdb

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/coverage-final.json

  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Check dependencies
        run: npm audit --audit-level=moderate

  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [lint, test, security]
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.DOCKER_REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=sha,prefix={{branch}}-
            type=semver,pattern={{version}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 2. Reusable Workflows

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
      image-tag:
        required: true
        type: string
    secrets:
      KUBECONFIG:
        required: true

jobs:
  deploy:
    name: Deploy to ${{ inputs.environment }}
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'v1.28.0'

      - name: Configure kubectl
        run: |
          echo "${{ secrets.KUBECONFIG }}" > kubeconfig.yaml
          export KUBECONFIG=kubeconfig.yaml

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/myapp \
            myapp=${{ inputs.image-tag }} \
            -n ${{ inputs.environment }}

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/myapp \
            -n ${{ inputs.environment }} \
            --timeout=5m

      - name: Run smoke tests
        run: npm run test:smoke
        env:
          API_URL: https://${{ inputs.environment }}.example.com
```

```yaml
# .github/workflows/production.yml
name: Production Deployment

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  deploy-staging:
    uses: ./.github/workflows/deploy.yml
    with:
      environment: staging
      image-tag: ghcr.io/myorg/myapp:${{ github.sha }}
    secrets:
      KUBECONFIG: ${{ secrets.STAGING_KUBECONFIG }}

  deploy-production:
    needs: deploy-staging
    uses: ./.github/workflows/deploy.yml
    with:
      environment: production
      image-tag: ghcr.io/myorg/myapp:${{ github.sha }}
    secrets:
      KUBECONFIG: ${{ secrets.PRODUCTION_KUBECONFIG }}
```

### 3. Matrix Testing

```yaml
name: Cross-Platform Tests

on: [push, pull_request]

jobs:
  test:
    name: Test on ${{ matrix.os }} with Node ${{ matrix.node }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node: ['18.x', '20.x', '21.x']

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test
```

## Deployment Strategies

### 1. Blue-Green Deployment

```yaml
# Two identical environments: blue (current) and green (new)
# Switch traffic after green is validated

apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
    version: blue  # Initially points to blue
  ports:
    - port: 80
      targetPort: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: green
  template:
    metadata:
      labels:
        app: myapp
        version: green
    spec:
      containers:
        - name: myapp
          image: myapp:v2.0.0
          ports:
            - containerPort: 8080
```

```bash
# Deployment script
#!/bin/bash

# Deploy green version
kubectl apply -f deployment-green.yaml

# Wait for green to be ready
kubectl rollout status deployment/myapp-green

# Run smoke tests against green
./smoke-tests.sh http://myapp-green.internal

# Switch traffic to green
kubectl patch service myapp -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor for 10 minutes
sleep 600

# If successful, scale down blue
kubectl scale deployment myapp-blue --replicas=0
```

### 2. Canary Deployment

```yaml
# Gradually shift traffic to new version
# Using Istio for traffic splitting

apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: myapp
spec:
  hosts:
    - myapp.example.com
  http:
    - match:
        - headers:
            canary:
              exact: "true"
      route:
        - destination:
            host: myapp
            subset: v2
    - route:
        - destination:
            host: myapp
            subset: v1
          weight: 90  # 90% to stable version
        - destination:
            host: myapp
            subset: v2
          weight: 10  # 10% to canary version
```

```yaml
# GitHub Actions canary deployment
name: Canary Deployment

on:
  push:
    branches: [main]

jobs:
  deploy-canary:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy canary (10%)
        run: kubectl apply -f k8s/canary-10.yaml

      - name: Monitor metrics
        run: ./scripts/monitor-canary.sh --duration=30m

      - name: Increase to 50%
        run: kubectl apply -f k8s/canary-50.yaml

      - name: Monitor metrics
        run: ./scripts/monitor-canary.sh --duration=30m

      - name: Promote to 100%
        run: kubectl apply -f k8s/canary-100.yaml
```

### 3. Rolling Deployment

```yaml
# Kubernetes rolling update (default strategy)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Max 2 extra pods during rollout
      maxUnavailable: 1  # Max 1 pod down during rollout
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: myapp
          image: myapp:v2.0.0
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 10
```

## Docker Best Practices

### Optimized Dockerfile

```dockerfile
# Multi-stage build for smaller images
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency files first (better caching)
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production image
FROM node:20-alpine

# Security: Run as non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

WORKDIR /app

# Copy built artifacts from builder
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package.json ./

# Switch to non-root user
USER nodejs

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:8080/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

EXPOSE 8080

CMD ["node", "dist/server.js"]
```

### Docker Compose for Local Development

```yaml
version: '3.9'

services:
  app:
    build:
      context: .
      target: builder
    ports:
      - "8080:8080"
    environment:
      NODE_ENV: development
      DATABASE_URL: postgresql://postgres:password@postgres:5432/myapp
      REDIS_URL: redis://redis:6379
    volumes:
      - .:/app
      - /app/node_modules
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: myapp
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  postgres-data:
  redis-data:
```

## Infrastructure as Code

### Terraform Example

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "app" {
  family                   = "${var.project_name}-app"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([{
    name  = "app"
    image = "${var.ecr_repository}:${var.image_tag}"
    portMappings = [{
      containerPort = 8080
      protocol      = "tcp"
    }]
    environment = [
      { name = "NODE_ENV", value = var.environment },
      { name = "PORT", value = "8080" }
    ]
    secrets = [
      {
        name      = "DATABASE_URL"
        valueFrom = aws_secretsmanager_secret.db_url.arn
      }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${var.project_name}"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "app"
      }
    }
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

# ECS Service
resource "aws_ecs_service" "app" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 8080
  }

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }

  depends_on = [aws_lb_listener.app]
}
```

## Best Practices

### 1. Pipeline Optimization

```yaml
# Cache dependencies
- name: Cache node modules
  uses: actions/cache@v3
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-

# Parallel jobs
jobs:
  unit-tests:
    runs-on: ubuntu-latest
  integration-tests:
    runs-on: ubuntu-latest
  security-scan:
    runs-on: ubuntu-latest
  # All run in parallel

# Fail fast
strategy:
  fail-fast: true
```

### 2. Secrets Management

```yaml
# Use GitHub Secrets
env:
  DATABASE_URL: ${{ secrets.DATABASE_URL }}
  API_KEY: ${{ secrets.API_KEY }}

# Use encrypted secrets at rest
- name: Decrypt secrets
  run: |
    echo "${{ secrets.GPG_PASSPHRASE }}" | \
    gpg --quiet --batch --yes --decrypt \
    --passphrase-fd 0 secrets.gpg > secrets.json
```

### 3. Environment Promotion

```yaml
# Deploy to staging first
deploy-staging:
  environment: staging
  steps:
    - run: deploy.sh staging

# Require approval for production
deploy-production:
  needs: deploy-staging
  environment:
    name: production
    url: https://example.com
  steps:
    - run: deploy.sh production
```

## Monitoring and Rollback

### Health Checks

```typescript
// Express.js health check endpoint
app.get('/health', (req, res) => {
  const health = {
    uptime: process.uptime(),
    timestamp: Date.now(),
    status: 'OK'
  };

  // Check dependencies
  try {
    await db.query('SELECT 1');
    await redis.ping();
    res.status(200).json(health);
  } catch (error) {
    health.status = 'ERROR';
    res.status(503).json(health);
  }
});
```

### Automated Rollback

```yaml
name: Auto Rollback

on:
  workflow_run:
    workflows: ["Deploy to Production"]
    types: [completed]

jobs:
  monitor:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    steps:
      - name: Monitor error rate
        id: monitor
        run: |
          ERROR_RATE=$(curl -s "https://metrics.example.com/error_rate")
          if (( $(echo "$ERROR_RATE > 5.0" | bc -l) )); then
            echo "High error rate detected: $ERROR_RATE%"
            echo "rollback=true" >> $GITHUB_OUTPUT
          fi

      - name: Rollback deployment
        if: steps.monitor.outputs.rollback == 'true'
        run: |
          kubectl rollout undo deployment/myapp -n production
          # Send alert
          curl -X POST "${{ secrets.SLACK_WEBHOOK }}" \
            -d '{"text":"🚨 Auto-rollback triggered due to high error rate"}'
```

## Common Anti-Patterns

### ❌ Anti-Pattern 1: No Pipeline Tests

```yaml
# ❌ BAD: Deploy without testing
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - run: deploy.sh production
```

### ❌ Anti-Pattern 2: Secrets in Code

```dockerfile
# ❌ BAD: Hardcoded secrets
ENV DATABASE_URL=postgresql://user:password@host:5432/db

# ✅ GOOD: Use environment variables
ENV DATABASE_URL=${DATABASE_URL}
```

### ❌ Anti-Pattern 3: No Rollback Strategy

```yaml
# ❌ BAD: No way to rollback
- run: kubectl apply -f deployment.yaml

# ✅ GOOD: Tag deployments for easy rollback
- run: kubectl apply -f deployment.yaml --record
```

## Quality Standards

- **Fast Feedback**: CI pipeline completes in <10 minutes
- **Test Coverage**: >80% code coverage with unit tests
- **Security**: Automated dependency scanning, container scanning
- **Zero Downtime**: Rolling deployments with health checks
- **Observability**: Structured logging, metrics, tracing
- **Rollback**: Automated rollback on failure detection

---

**Skill Type**: DevOps - Automation
**Complexity**: Moderate
**Typical Usage**: Activated when DevOps engineers design CI/CD pipelines or deployment strategies
**Tools**: GitHub Actions, Docker, Kubernetes, Terraform
