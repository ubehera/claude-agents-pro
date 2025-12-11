---
name: docker-security-patterns
description: Advanced Docker security patterns including multi-stage builds, BuildKit, security scanning, rootless containers, and production hardening. Use when building secure container images, implementing container security best practices, or optimizing Docker workflows.
trigger_keywords: [docker, dockerfile, multi-stage build, buildkit, docker security, container security, distroless, rootless docker, docker compose, container scanning, trivy, hadolint]
---

# Docker Security & Advanced Patterns

Production-grade Docker patterns for security, optimization, and best practices using multi-stage builds, BuildKit, and security scanning.

## Multi-Stage Builds

### Node.js Application

**Optimized for size and security:**

```dockerfile
# syntax=docker/dockerfile:1.4

# Stage 1: Build dependencies
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && \
    npm cache clean --force

# Stage 2: Build application
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build && \
    npm prune --production

# Stage 3: Production image
FROM node:20-alpine AS runner
WORKDIR /app

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Copy built application
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/package.json ./

# Security: Run as non-root
USER nodejs

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:${PORT:-3000}/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

EXPOSE 3000
ENV NODE_ENV=production

CMD ["node", "dist/server.js"]
```

### Go Application

```dockerfile
# syntax=docker/dockerfile:1.4

# Stage 1: Build
FROM golang:1.21-alpine AS builder
WORKDIR /build

# Cache dependencies
COPY go.mod go.sum ./
RUN go mod download

# Build binary
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags='-w -s -extldflags "-static"' \
    -o app ./cmd/server

# Stage 2: Minimal runtime
FROM gcr.io/distroless/static-debian12
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/app /app/

# Run as non-root (distroless uses uid 65532)
USER nonroot:nonroot

# Health check not supported in distroless
# Use liveness probe in Kubernetes instead

EXPOSE 8080
ENTRYPOINT ["/app/app"]
```

### Python Application

```dockerfile
# syntax=docker/dockerfile:1.4

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim
WORKDIR /app

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy application
COPY --chown=appuser:appuser . .

USER appuser

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
```

## BuildKit Features

### Cache Mounts

**Speed up builds with persistent cache:**

```dockerfile
# syntax=docker/dockerfile:1.4

FROM node:20-alpine

WORKDIR /app

# Cache npm packages
RUN --mount=type=cache,target=/root/.npm \
    npm install -g npm@latest

COPY package*.json ./

# Use cache mount for node_modules
RUN --mount=type=cache,target=/root/.npm \
    --mount=type=cache,target=/app/node_modules \
    npm ci

COPY . .

RUN --mount=type=cache,target=/app/node_modules \
    npm run build
```

### Secret Mounts

**Safely use secrets during build:**

```dockerfile
# syntax=docker/dockerfile:1.4

FROM python:3.12

# Use secret for private package installation
RUN --mount=type=secret,id=pip_credentials \
    pip config set global.index-url $(cat /run/secrets/pip_credentials)

# Secret never persists in image layers
RUN --mount=type=secret,id=github_token \
    git clone https://oauth2:$(cat /run/secrets/github_token)@github.com/private/repo.git
```

**Build with secrets:**

```bash
docker buildx build \
  --secret id=pip_credentials,src=.pip-credentials \
  --secret id=github_token,env=GITHUB_TOKEN \
  -t myapp:latest .
```

### SSH Mounts

**Access SSH keys during build:**

```dockerfile
# syntax=docker/dockerfile:1.4

FROM golang:1.21-alpine

# Configure git to use SSH
RUN apk add --no-cache git openssh-client && \
    mkdir -p ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts

# Clone private repos using SSH
RUN --mount=type=ssh \
    git clone git@github.com:myorg/private-repo.git
```

```bash
# Build with SSH forwarding
docker buildx build --ssh default=$SSH_AUTH_SOCK -t myapp:latest .
```

## Security Best Practices

### Minimal Base Images

**Choose smallest secure base:**

```dockerfile
# ❌ Large base (900+ MB)
FROM ubuntu:22.04

# ✅ Better (150 MB)
FROM alpine:3.19

# ✅ Best for Go/Rust (2 MB)
FROM gcr.io/distroless/static-debian12

# ✅ Best for Node.js (40 MB)
FROM node:20-alpine

# ✅ Best for Python (50 MB)
FROM python:3.12-slim
```

### Non-Root User

```dockerfile
# Alpine-based
RUN addgroup -g 1001 -S appuser && \
    adduser -S appuser -u 1001

# Debian/Ubuntu-based
RUN useradd -m -u 1001 appuser

# Create user with specific UID/GID
RUN groupadd -r -g 999 appgroup && \
    useradd -r -u 999 -g appgroup appuser

# Switch to non-root
USER appuser

# Verify (for debugging)
RUN whoami  # Should output: appuser
```

### Read-Only Root Filesystem

```dockerfile
FROM node:20-alpine

WORKDIR /app

# Create directories that need write access
RUN mkdir -p /app/tmp /app/logs && \
    chown -R node:node /app

USER node

# Application will run with read-only root
# Mount volumes for writable directories:
# docker run --read-only -v /app/tmp -v /app/logs ...
```

**Docker Compose with read-only:**

```yaml
services:
  app:
    image: myapp:latest
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs
```

### Drop Capabilities

```dockerfile
# Dockerfile security context
FROM alpine:3.19
USER nobody

# Runtime with dropped capabilities
# docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE myapp
```

**Docker Compose:**

```yaml
services:
  app:
    image: myapp:latest
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Only if binding to ports < 1024
    security_opt:
      - no-new-privileges:true
```

## Security Scanning

### Trivy Scanning

```bash
# Scan image for vulnerabilities
trivy image myapp:latest

# Scan with severity filter
trivy image --severity HIGH,CRITICAL myapp:latest

# Output as JSON
trivy image -f json -o scan-results.json myapp:latest

# Scan Dockerfile
trivy config Dockerfile

# Fail build on critical vulnerabilities
trivy image --exit-code 1 --severity CRITICAL myapp:latest
```

**GitHub Actions integration:**

```yaml
name: Container Security

on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Run Trivy scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### Hadolint (Dockerfile Linter)

```bash
# Install hadolint
brew install hadolint

# Lint Dockerfile
hadolint Dockerfile

# Output as JSON
hadolint -f json Dockerfile

# Ignore specific rules
hadolint --ignore DL3008 --ignore DL3009 Dockerfile
```

**.hadolint.yaml:**

```yaml
ignored:
  - DL3008  # Pin versions in apt-get
  - DL3018  # Pin versions in apk

trustedRegistries:
  - docker.io
  - gcr.io
  - ghcr.io
```

### Docker Bench Security

```bash
# Run Docker security audit
docker run --rm --net host --pid host --userns host --cap-add audit_control \
  -v /var/lib:/var/lib:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v /etc:/etc:ro \
  docker/docker-bench-security
```

## Advanced Patterns

### BuildKit Inline Cache

**Speed up CI/CD builds:**

```bash
# Build with cache export
docker buildx build \
  --cache-to type=inline \
  --push \
  -t myapp:latest .

# Build using cache
docker buildx build \
  --cache-from myapp:latest \
  -t myapp:new-version .
```

### Multi-Platform Builds

```bash
# Create builder
docker buildx create --name multiplatform --use

# Build for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64,linux/arm/v7 \
  -t myapp:latest \
  --push .
```

**Dockerfile with platform-specific logic:**

```dockerfile
FROM --platform=$BUILDPLATFORM golang:1.21-alpine AS builder
ARG TARGETPLATFORM
ARG BUILDPLATFORM

WORKDIR /build
COPY . .

# Build for target platform
RUN GOOS=$(echo $TARGETPLATFORM | cut -d/ -f1) \
    GOARCH=$(echo $TARGETPLATFORM | cut -d/ -f2) \
    go build -o app .

FROM alpine:3.19
COPY --from=builder /build/app /app
ENTRYPOINT ["/app"]
```

### Docker Compose Production

```yaml
version: '3.9'

services:
  app:
    image: myapp:${VERSION:-latest}
    build:
      context: .
      dockerfile: Dockerfile
      cache_from:
        - myapp:latest
      args:
        BUILDKIT_INLINE_CACHE: 1

    # Security
    read_only: true
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

    # Health check
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:3000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s

    # Logging
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

    # Environment
    environment:
      NODE_ENV: production

    # Secrets
    secrets:
      - db_password
      - api_key

    # Tmpfs for writable directories
    tmpfs:
      - /tmp
      - /app/cache

  # Secrets definition
secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    external: true
```

### Rootless Docker

**Run Docker daemon as non-root:**

```bash
# Install rootless Docker
curl -fsSL https://get.docker.com/rootless | sh

# Start daemon
systemctl --user start docker

# Set environment
export PATH=$HOME/bin:$PATH
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock

# Verify
docker run hello-world
```

## Image Optimization

### Layer Caching

```dockerfile
# ❌ Bad - breaks cache on any file change
FROM node:20-alpine
COPY . .
RUN npm install

# ✅ Good - cache dependencies separately
FROM node:20-alpine
COPY package*.json ./
RUN npm install
COPY . .
```

### Minimize Layers

```dockerfile
# ❌ Bad - creates multiple layers
FROM alpine:3.19
RUN apk add --no-cache curl
RUN apk add --no-cache git
RUN apk add --no-cache vim

# ✅ Good - single layer
FROM alpine:3.19
RUN apk add --no-cache \
    curl \
    git \
    vim && \
    rm -rf /var/cache/apk/*
```

### Use .dockerignore

```
# .dockerignore
.git
.gitignore
.env
.env.local
node_modules
npm-debug.log
Dockerfile
.dockerignore
README.md
tests/
docs/
*.md
.vscode/
.idea/
__pycache__/
*.pyc
*.pyo
*.pyd
```

## Best Practices Checklist

- [ ] Use specific image tags (not `latest`)
- [ ] Implement multi-stage builds
- [ ] Run as non-root user
- [ ] Use minimal base images (Alpine, Distroless)
- [ ] Scan for vulnerabilities (Trivy, Snyk)
- [ ] Lint Dockerfiles (Hadolint)
- [ ] Pin package versions
- [ ] Use BuildKit features (cache mounts, secrets)
- [ ] Implement health checks
- [ ] Drop unnecessary capabilities
- [ ] Use read-only root filesystem
- [ ] Set resource limits
- [ ] Configure proper logging
- [ ] Use `.dockerignore` to exclude files
- [ ] Sign and verify images (Docker Content Trust)

## Quality Standards

- **Security**: No critical vulnerabilities, runs as non-root, minimal attack surface
- **Size**: Optimized images (<100MB for apps, <10MB for static binaries)
- **Performance**: Efficient layer caching, fast builds (<5 min)
- **Reliability**: Health checks, proper signal handling, graceful shutdown
- **Maintainability**: Clear Dockerfile comments, reproducible builds

## Related Skills

- `ci-cd-patterns` - For automated builds and deployment
- `kubernetes-advanced-patterns` - For container orchestration
- `prometheus-configuration` - For container monitoring

---

**Skill Type**: DevOps - Container Security
**Complexity**: Advanced
**Typical Usage**: Building production container images, implementing security best practices
**Prerequisites**: Docker basics, Linux fundamentals, security awareness
