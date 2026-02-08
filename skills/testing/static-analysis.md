---
name: static-analysis
description: Load when configuring static analysis tools (SAST, linters, formatters) for automated code quality and security scanning in CI/CD pipelines
trigger_keywords: [static analysis, sast, semgrep, sonarqube, eslint, pylint, code scanning, security scanning, linting, code quality automation]
---

# Static Analysis Patterns

Comprehensive static analysis tool configuration for automated code quality, security vulnerability detection, and compliance enforcement.

## Overview

Static Application Security Testing (SAST) and code quality tools analyze source code without execution to detect vulnerabilities, bugs, and style violations.

**When to Use**:
- Setting up CI/CD security gates
- Implementing DevSecOps practices
- Automating code quality enforcement
- Compliance scanning (PCI-DSS, SOC 2)
- Pre-commit hooks for code standards

**Tool Categories**:
- **Security**: Semgrep, SonarQube, CodeQL, Bandit
- **Quality**: ESLint, Pylint, RuboCop, golangci-lint
- **Formatting**: Prettier, Black, gofmt
- **Type Checking**: TypeScript, mypy, Flow

## Core Concepts

- **Shift Left Security**: SAST tools catch vulnerabilities before code review - automated scanning finds 70%+ of common security issues faster than manual review
- **Incremental Adoption**: Start with security-focused rules only, establish baseline for existing issues, then gradually add quality rules - avoid overwhelming teams with thousands of warnings
- **False Positive Management**: Tune severity levels, add path exclusions, use inline suppressions with justification - high false positive rates cause developers to ignore all warnings
- **Pre-commit vs CI**: Pre-commit hooks catch issues instantly during development; CI gates enforce quality on all code - use both for defense in depth
- **Custom Rules for Patterns**: Write organization-specific rules for internal security patterns, deprecated APIs, and architectural constraints - generic rules miss domain-specific issues

## Tool Selection Matrix

| Tool | Best For | Language Support | Cost | CI Integration |
|------|----------|------------------|------|----------------|
| **Semgrep** | Custom rules, fast scans | 30+ languages | Free/Enterprise | Excellent |
| **SonarQube** | Code quality + security | 25+ languages | Free/Commercial | Good |
| **CodeQL** | Deep analysis, research | 10+ languages | Free (OSS) | GitHub native |
| **ESLint** | JavaScript/TypeScript quality | JS/TS | Free | Excellent |
| **Pylint** | Python quality | Python | Free | Good |

## Semgrep Configuration

### Quick Start

```bash
# Install Semgrep
pip install semgrep

# Run with default rulesets
semgrep --config=auto --error

# Scan with specific rulesets
semgrep --config "p/security-audit" --config "p/owasp-top-ten" .
```

### Configuration File

```yaml
# .semgrep.yml
rules:
  # Custom security rule
  - id: hardcoded-jwt-secret
    pattern: jwt.encode($DATA, "...", ...)
    message: JWT secret should not be hardcoded
    severity: ERROR
    languages: [python]
    metadata:
      category: security
      cwe: "CWE-798: Use of Hard-coded Credentials"
      owasp: "A02:2021 - Cryptographic Failures"

  # SQL injection prevention
  - id: sql-injection-risk
    patterns:
      - pattern: execute($QUERY)
      - pattern-inside: |
          $QUERY = f"... {$VAR} ..."
    message: Potential SQL injection - use parameterized queries
    severity: ERROR
    languages: [python]
    fix: Use parameterized queries with placeholders

  # API key exposure
  - id: exposed-api-key
    pattern-regex: '(?i)(api[_-]?key|apikey|api[_-]?secret)[\s]*=[\s]*["\'][a-zA-Z0-9]{20,}["\']'
    message: Potential API key exposure
    severity: WARNING
    languages: [python, javascript, typescript]

  # Missing authentication
  - id: missing-auth-decorator
    pattern: |
      @app.route(...)
      def $FUNC(...):
        ...
    pattern-not: |
      @app.route(...)
      @login_required
      def $FUNC(...):
        ...
    message: Route missing authentication decorator
    severity: WARNING
    languages: [python]
    paths:
      include:
        - "api/"
```

### Advanced Pattern Matching

```yaml
# Detect insecure random usage
- id: insecure-random
  pattern-either:
    - pattern: random.random()
    - pattern: random.randint(...)
  message: Use secrets module for cryptographic randomness
  severity: WARNING
  languages: [python]
  fix: |
    import secrets
    secrets.randbelow(...)

# Detect missing input validation
- id: missing-input-validation
  pattern: |
    @app.route("/api/<path:$PATH>", methods=["POST"])
    def $FUNC($PARAM):
      $DATA = request.json
      ...
  pattern-not-inside: |
    ...
    if not $DATA:
      ...
    ...
  message: Missing input validation for POST endpoint
  severity: WARNING
```

### CI/CD Integration

```yaml
# .github/workflows/semgrep.yml
name: Semgrep Security Scan

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  semgrep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/owasp-top-ten
            p/python
            .semgrep.yml
          generateSarif: true

      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: semgrep.sarif
```

## ESLint Configuration (JavaScript/TypeScript)

### Setup

```bash
npm install --save-dev eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin

# Initialize config
npx eslint --init
```

### Configuration

```javascript
// .eslintrc.js
module.exports = {
    parser: '@typescript-eslint/parser',
    parserOptions: {
        ecmaVersion: 2022,
        sourceType: 'module',
        project: './tsconfig.json',
    },
    extends: [
        'eslint:recommended',
        'plugin:@typescript-eslint/recommended',
        'plugin:@typescript-eslint/recommended-requiring-type-checking',
        'plugin:security/recommended',
    ],
    plugins: ['@typescript-eslint', 'security'],
    rules: {
        // Security rules
        'no-eval': 'error',
        'no-implied-eval': 'error',
        'no-new-func': 'error',
        'security/detect-object-injection': 'warn',
        'security/detect-non-literal-regexp': 'warn',

        // Code quality rules
        '@typescript-eslint/no-explicit-any': 'error',
        '@typescript-eslint/explicit-function-return-type': 'warn',
        '@typescript-eslint/no-unused-vars': ['error', {
            argsIgnorePattern: '^_',
        }],
        'no-console': ['warn', { allow: ['warn', 'error'] }],
        'prefer-const': 'error',
        'no-var': 'error',

        // Complexity rules
        'complexity': ['warn', 10],
        'max-depth': ['warn', 3],
        'max-lines-per-function': ['warn', 100],
    },
    overrides: [
        {
            files: ['*.test.ts', '*.spec.ts'],
            rules: {
                '@typescript-eslint/no-explicit-any': 'off',
            },
        },
    ],
};
```

### Custom Rules

```javascript
// eslint-rules/no-hardcoded-credentials.js
module.exports = {
    meta: {
        type: 'problem',
        docs: {
            description: 'Disallow hardcoded credentials',
            category: 'Security',
        },
        messages: {
            hardcodedCredential: 'Hardcoded credential detected: {{name}}',
        },
    },
    create(context) {
        const sensitiveNames = ['password', 'apiKey', 'secret', 'token'];

        return {
            VariableDeclarator(node) {
                const name = node.id.name;
                if (sensitiveNames.some(s => name.toLowerCase().includes(s))) {
                    if (node.init && node.init.type === 'Literal') {
                        context.report({
                            node,
                            messageId: 'hardcodedCredential',
                            data: { name },
                        });
                    }
                }
            },
        };
    },
};
```

## Pylint Configuration (Python)

### Setup

```bash
pip install pylint pylint-django bandit
```

### Configuration

```ini
# .pylintrc
[MASTER]
jobs=4
load-plugins=pylint_django

[MESSAGES CONTROL]
disable=
    missing-docstring,
    too-few-public-methods,
    invalid-name

enable=
    use-symbolic-message-instead,
    useless-suppression

[FORMAT]
max-line-length=100
indent-string='    '

[BASIC]
good-names=i,j,k,_,id,db

[DESIGN]
max-args=7
max-attributes=10
max-locals=15
max-returns=6
max-branches=12
max-statements=50
min-public-methods=1

[SIMILARITIES]
min-similarity-lines=5
ignore-comments=yes
ignore-docstrings=yes

[SECURITY]
# Additional security checks
enable=
    eval-used,
    exec-used,
    dangerous-default-value
```

### Security-Specific with Bandit

```yaml
# .bandit
skips: ['B101', 'B601']

tests:
  - B201  # Flask debug mode
  - B301  # Pickle usage
  - B302  # Insecure deserialization
  - B303  # Insecure MD5/SHA1
  - B304  # Insecure cipher
  - B305  # Insecure cipher mode
  - B306  # Insecure mktemp
  - B307  # Eval usage
  - B308  # Mark safe usage
  - B309  # HTTPSConnection
  - B310  # URL open
  - B311  # Random for crypto
  - B312  # Telnet usage
  - B313  # XML vulnerabilities
  - B314  # XML vulnerabilities
  - B315  # XML vulnerabilities
  - B316  # XML vulnerabilities
  - B317  # XML vulnerabilities
  - B318  # XML vulnerabilities
  - B319  # XML vulnerabilities
  - B320  # XML vulnerabilities
  - B321  # FTP usage
  - B322  # Input usage
  - B323  # Unverified context
  - B324  # Insecure hash
  - B325  # Tempfile
  - B501  # SSL/TLS issues
  - B502  # SSL/TLS issues
  - B503  # SSL/TLS issues
  - B504  # SSL/TLS issues
  - B505  # Weak crypto
  - B506  # YAML load
  - B507  # SSH no host key verification
  - B601  # Shell injection
  - B602  # Shell injection
  - B603  # Process without shell
  - B604  # Shell true
  - B605  # Shell string
  - B606  # Process no shell
  - B607  # Partial path
  - B608  # SQL injection
  - B609  # Wildcard injection

exclude_dirs:
  - /test
  - /tests
  - /venv
```

## SonarQube Configuration

### Setup

```bash
# Docker setup
docker run -d --name sonarqube \
  -p 9000:9000 \
  -v sonarqube_data:/opt/sonarqube/data \
  -v sonarqube_logs:/opt/sonarqube/logs \
  -v sonarqube_extensions:/opt/sonarqube/extensions \
  sonarqube:latest
```

### Project Configuration

```properties
# sonar-project.properties
sonar.projectKey=my-project
sonar.projectName=My Project
sonar.projectVersion=1.0

# Source code
sonar.sources=src
sonar.tests=tests
sonar.exclusions=**/node_modules/**,**/*.test.ts

# Coverage
sonar.javascript.lcov.reportPaths=coverage/lcov.info
sonar.python.coverage.reportPaths=coverage.xml

# Quality gates
sonar.qualitygate.wait=true
sonar.qualitygate.timeout=300

# Thresholds
sonar.coverage.exclusions=**/*.test.ts,**/test/**
```

### Quality Gate Configuration

```javascript
// Quality Gate: Production Ready
{
    "conditions": [
        {
            "metric": "new_coverage",
            "op": "LT",
            "error": "80"
        },
        {
            "metric": "new_duplicated_lines_density",
            "op": "GT",
            "error": "3"
        },
        {
            "metric": "new_maintainability_rating",
            "op": "GT",
            "error": "1"
        },
        {
            "metric": "new_reliability_rating",
            "op": "GT",
            "error": "1"
        },
        {
            "metric": "new_security_rating",
            "op": "GT",
            "error": "1"
        },
        {
            "metric": "new_security_hotspots_reviewed",
            "op": "LT",
            "error": "100"
        }
    ]
}
```

## Pre-commit Hooks

### Setup

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: detect-private-key

  - repo: https://github.com/returntocorp/semgrep
    rev: v1.45.0
    hooks:
      - id: semgrep
        args: ['--config=auto', '--error']

  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/pylint
    rev: v3.0.0
    hooks:
      - id: pylint
        args: ['--max-line-length=100']

  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.51.0
    hooks:
      - id: eslint
        files: \.[jt]sx?$
        types: [file]
        additional_dependencies:
          - eslint@8.51.0
          - '@typescript-eslint/eslint-plugin@6.7.5'
          - '@typescript-eslint/parser@6.7.5'
```

## CI/CD Pipeline Integration

### GitHub Actions

```yaml
# .github/workflows/code-quality.yml
name: Code Quality

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run ESLint
        run: npm run lint

      - name: Run TypeScript check
        run: npm run type-check

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: p/security-audit

      - name: Run npm audit
        run: npm audit --audit-level=moderate

  quality-gate:
    runs-on: ubuntu-latest
    needs: [lint, security]
    steps:
      - uses: actions/checkout@v3

      - name: SonarQube Scan
        uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}

      - name: Quality Gate Check
        uses: sonarsource/sonarqube-quality-gate-action@master
        timeout-minutes: 5
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

## Best Practices

### 1. Incremental Adoption

```bash
# Start with security-focused rules only
semgrep --config "p/security-audit" --baseline

# Gradually add quality rules
semgrep --config "p/security-audit" --config "p/code-quality"

# Full coverage
semgrep --config=auto
```

### 2. False Positive Management

```yaml
# .semgrep.yml
rules:
  - id: sql-injection
    pattern: execute($QUERY)
    paths:
      include:
        - "src/"
      exclude:
        - "src/migrations/"  # Exclude migrations
    # Allow legitimate uses
    pattern-not: execute("SELECT version()")
```

### 3. Performance Optimization

```yaml
# Only scan changed files in CI
- name: Get changed files
  id: changed-files
  run: |
    git diff --name-only origin/main...HEAD > changed-files.txt

- name: Scan changed files
  run: |
    semgrep --config=auto $(cat changed-files.txt)
```

### 4. Baseline Scanning

```bash
# Create baseline for existing issues
semgrep --config=auto --baseline > semgrep-baseline.json

# Only flag new issues
semgrep --config=auto --baseline semgrep-baseline.json
```

## Common Use Cases

### New Project Setup

```bash
#!/bin/bash
# scripts/setup-static-analysis.sh

# Detect primary language
if [ -f "package.json" ]; then
    echo "Setting up JavaScript/TypeScript analysis..."
    npm install --save-dev eslint @typescript-eslint/parser
    npx eslint --init
fi

if [ -f "requirements.txt" ]; then
    echo "Setting up Python analysis..."
    pip install pylint bandit black
    pylint --generate-rcfile > .pylintrc
fi

# Install Semgrep
pip install semgrep

# Create baseline
semgrep --config=auto --baseline > semgrep-baseline.json

echo "Static analysis setup complete!"
```

### Compliance Scanning

```bash
# PCI-DSS compliance scan
semgrep --config "p/pci-dss" --json -o pci-scan.json

# OWASP Top 10
semgrep --config "p/owasp-top-ten" --json -o owasp-scan.json

# Generate compliance report
python scripts/generate-compliance-report.py pci-scan.json owasp-scan.json
```

## Troubleshooting

### High False Positive Rate

**Problem**: Too many false positives slow down development

**Solutions**:
- Tune rule severity levels
- Add path filters to exclude test files
- Create organization-specific exceptions
- Use `nosemgrep` comments for legitimate cases

```python
# nosemgrep: python.lang.security.audit.dangerous-system-call
os.system("safe-command")  # Reviewed: command is safe
```

### Performance Issues

**Problem**: Scans take too long in CI

**Solutions**:
- Enable incremental scanning
- Parallelize across modules
- Cache scan results
- Optimize rule patterns

```yaml
# Cache Semgrep results
- name: Cache Semgrep
  uses: actions/cache@v3
  with:
    path: ~/.semgrep
    key: semgrep-${{ hashFiles('.semgrep.yml') }}
```

## Quality Metrics

- **Coverage**: >80% code scanned
- **False Positive Rate**: <10%
- **Scan Time**: <5 minutes in CI
- **Security Issues**: 0 critical, <5 high
- **Code Quality**: Maintainability rating A/B

---

**Skill Type**: Code Quality - Automation
**Complexity**: Moderate to High
**Typical Usage**: Activated when setting up CI/CD pipelines or implementing DevSecOps
**Performance**: Automated scanning catches 70%+ of common vulnerabilities before code review
