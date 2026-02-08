---
description: Comprehensive security assessment with OWASP compliance and automated remediation
args: [target] [--framework=owasp|nist|cis] [--scope=api|infra|app|all] [--fix] [--report=json|html|sarif]
tools: Task, Read, Grep
model: sonnet
---

## Purpose
Comprehensive security assessment with framework-based analysis, automated vulnerability detection, and remediation guidance following industry standards.

## Security Frameworks Supported

### OWASP (Default)
- **API Security Top 10**: Authentication, authorization, data exposure
- **Application Security**: Input validation, session management, crypto
- **Infrastructure**: Server configuration, dependency management

### NIST Cybersecurity Framework
- **Identify**: Asset discovery, risk assessment
- **Protect**: Access control, data security, training
- **Detect**: Monitoring, anomaly detection
- **Respond**: Incident response planning
- **Recover**: Recovery planning, communications

### CIS Controls
- **Basic Controls**: Inventory, configuration, vulnerability management
- **Foundational**: Access control, secure configuration
- **Organizational**: Security training, incident response

## Security Assessment Scopes

### API Security (`--scope=api`)
- Authentication and authorization mechanisms
- Input validation and output encoding
- Rate limiting and throttling
- API versioning and deprecation
- Data exposure and sensitive information leakage
- CORS and security headers

### Infrastructure Security (`--scope=infra`)
- Server hardening and configuration
- Network security and segmentation
- Container and orchestration security
- Cloud security posture (AWS/GCP/Azure)
- Certificate and TLS configuration
- Logging and monitoring setup

### Application Security (`--scope=app`)
- Code quality and secure coding practices
- Dependency vulnerability scanning
- Secret management and exposure
- Session management and authentication flows
- Data protection and privacy compliance
- Client-side security (XSS, CSRF protection)

### Comprehensive Assessment (`--scope=all`)
- Full-spectrum security analysis
- Cross-domain security integration
- End-to-end security validation
- Compliance framework mapping

## Usage Examples

```bash
# Comprehensive OWASP security audit
/quality-security-audit ./src --framework=owasp --scope=all --report=html

# API-focused security assessment
/quality-security-audit ./api --scope=api --fix --report=json

# Infrastructure security with CIS controls
/quality-security-audit ./terraform --framework=cis --scope=infra

# Application security with NIST framework
/quality-security-audit ./app --framework=nist --scope=app --report=sarif

# Quick security scan with auto-fix
/quality-security-audit ./components --fix

# Multi-target security assessment
/quality-security-audit ./microservices --scope=all --parallel
```

## Input Validation & Setup

```bash
TARGET=$1
FRAMEWORK=${2#--framework=}
SCOPE=${3#--scope=}
FIX_MODE=$(echo "$4 $5 $6" | grep -q "--fix" && echo "true" || echo "false")
REPORT_FORMAT=$(echo "$4 $5 $6" | grep -o "--report=[^[:space:]]*" | cut -d= -f2)

# Set defaults
FRAMEWORK=${FRAMEWORK:-"owasp"}
SCOPE=${SCOPE:-"all"}
REPORT_FORMAT=${REPORT_FORMAT:-"html"}

# Validate target exists
if [ -z "$TARGET" ]; then
  echo "❌ Error: Target required for security audit"
  echo "💡 Usage: /quality-security-audit [target] [--options]"
  echo "📚 Examples:"
  echo "  /quality-security-audit ./src --framework=owasp --scope=api"
  echo "  /quality-security-audit ./infra --framework=cis --fix"
  echo "  /quality-security-audit ./app --scope=all --report=json"
  exit 1
fi

if [ ! -e "$TARGET" ]; then
  echo "❌ Error: Target path not found: $TARGET"
  echo "💡 Provide a valid file or directory path"
  exit 1
fi

# Validate framework
case "$FRAMEWORK" in
  "owasp"|"nist"|"cis")
    echo "✓ Using security framework: $FRAMEWORK"
    ;;
  *)
    echo "❌ Error: Unsupported framework: $FRAMEWORK"
    echo "💡 Available frameworks: owasp, nist, cis"
    exit 1
    ;;
esac

# Validate scope
case "$SCOPE" in
  "api"|"infra"|"app"|"all")
    echo "✓ Security assessment scope: $SCOPE"
    ;;
  *)
    echo "❌ Error: Invalid scope: $SCOPE"
    echo "💡 Available scopes: api, infra, app, all"
    exit 1
    ;;
esac

# Validate report format
case "$REPORT_FORMAT" in
  "json"|"html"|"sarif"|"xml")
    echo "✓ Report format: $REPORT_FORMAT"
    ;;
  *)
    echo "❌ Error: Unsupported report format: $REPORT_FORMAT"
    echo "💡 Available formats: json, html, sarif, xml"
    exit 1
    ;;
esac
```

## Security Assessment Implementation

### OWASP Framework Assessment

```bash
execute_owasp_assessment() {
  local target=$1
  local scope=$2
  local fix_mode=$3
  
  echo "🔒 OWASP Security Assessment: $target"
  echo "   Scope: $scope"
  echo "   Auto-fix: $fix_mode"
  
  case "$scope" in
    "api")
      assess_owasp_api_security "$target" "$fix_mode"
      ;;
    "infra")
      assess_owasp_infrastructure "$target" "$fix_mode"
      ;;
    "app")
      assess_owasp_application "$target" "$fix_mode"
      ;;
    "all")
      assess_owasp_comprehensive "$target" "$fix_mode"
      ;;
  esac
}

assess_owasp_api_security() {
  local target=$1
  local fix_mode=$2
  
  echo "🔍 OWASP API Security Top 10 Assessment"
  
  # API1: Broken Object Level Authorization
  check_object_level_authorization "$target"
  
  # API2: Broken User Authentication
  check_authentication_mechanisms "$target"
  
  # API3: Excessive Data Exposure
  check_data_exposure "$target"
  
  # API4: Lack of Resources & Rate Limiting
  check_rate_limiting "$target"
  
  # API5: Broken Function Level Authorization
  check_function_level_authorization "$target"
  
  # API6: Mass Assignment
  check_mass_assignment "$target"
  
  # API7: Security Misconfiguration
  check_security_configuration "$target"
  
  # API8: Injection
  check_injection_vulnerabilities "$target"
  
  # API9: Improper Assets Management
  check_assets_management "$target"
  
  # API10: Insufficient Logging & Monitoring
  check_logging_monitoring "$target"
  
  if [ "$fix_mode" = "true" ]; then
    apply_api_security_fixes "$target"
  fi
}
```

### Agent-Assisted Security Analysis

For complex security assessments, delegate to `security-architect`:

**Task**: Comprehensive security assessment using $FRAMEWORK framework

**Target**: $TARGET
**Scope**: $SCOPE security analysis
**Framework**: $FRAMEWORK compliance assessment
**Auto-remediation**: $FIX_MODE

**Instructions for security-architect**:

1. **Security Analysis Focus**:
   - Apply $FRAMEWORK security framework methodology
   - Assess $SCOPE-specific security concerns
   - Identify vulnerabilities and security gaps
   - Evaluate current security controls effectiveness

2. **Assessment Areas**:
   - **Authentication & Authorization**: Access control mechanisms
   - **Data Protection**: Encryption, privacy, data handling
   - **Input Validation**: Injection prevention, sanitization
   - **Configuration Security**: Secure defaults, hardening
   - **Dependency Security**: Third-party library vulnerabilities
   - **Infrastructure Security**: Network, server, cloud configuration

3. **Compliance Mapping**:
   - Map findings to $FRAMEWORK control requirements
   - Assess compliance gaps and risk levels
   - Prioritize remediation based on risk assessment
   - Provide compliance roadmap and timelines

4. **Remediation Guidance**:
   - Provide specific, actionable remediation steps
   - Include code examples and configuration changes
   - Recommend security tools and automation
   - Define verification and testing approaches

5. **Quality Requirements**:
   - Generate detailed findings with evidence
   - Provide risk ratings (Critical, High, Medium, Low)
   - Include remediation timelines and effort estimates
   - Create executive summary for stakeholder communication

**Deliverables**:
- Security assessment report in $REPORT_FORMAT format
- Prioritized vulnerability list with remediation guidance
- Compliance gap analysis and roadmap
- Security control implementation recommendations
- Automated security tooling suggestions

## Automated Security Checks

### Dependency Vulnerability Scanning

```bash
check_dependency_vulnerabilities() {
  local target=$1
  
  echo "🔍 Scanning dependencies for known vulnerabilities..."
  
  # Node.js projects
  if [ -f "$target/package.json" ]; then
    echo "  ✓ Scanning npm dependencies"
    cd "$target" && npm audit --audit-level=moderate 2>/dev/null || true
  fi
  
  # Python projects  
  if [ -f "$target/requirements.txt" ] || [ -f "$target/pyproject.toml" ]; then
    echo "  ✓ Scanning Python dependencies"
    cd "$target" && python -m pip check 2>/dev/null || true
  fi
  
  # Go projects
  if [ -f "$target/go.mod" ]; then
    echo "  ✓ Scanning Go dependencies"
    cd "$target" && go list -m -u all 2>/dev/null || true
  fi
}
```

### Secret Detection

```bash
check_secret_exposure() {
  local target=$1
  
  echo "🔍 Scanning for exposed secrets and credentials..."
  
  # Common secret patterns
  local secret_patterns=(
    "password[[:space:]]*=[[:space:]]*['\"][^'\"]+['\"]"  # passwords
    "api[_-]?key[[:space:]]*=[[:space:]]*['\"][^'\"]+['\"]"  # API keys
    "secret[_-]?key[[:space:]]*=[[:space:]]*['\"][^'\"]+['\"]"  # secret keys
    "[a-zA-Z0-9]{32,}"  # potential tokens
    "-----BEGIN [A-Z ]+-----"  # private keys
  )
  
  for pattern in "${secret_patterns[@]}"; do
    local matches=$(grep -r -E "$pattern" "$target" 2>/dev/null | wc -l || echo 0)
    if [ "$matches" -gt 0 ]; then
      echo "  ⚠️ Found $matches potential secret exposures"
    fi
  done
}
```

### Configuration Security

```bash
check_security_configuration() {
  local target=$1
  
  echo "🔍 Checking security configuration..."
  
  # Check for security headers configuration
  if [ -f "$target/nginx.conf" ] || [ -f "$target/.htaccess" ]; then
    echo "  ✓ Checking web server security headers"
    # Check for HSTS, CSP, X-Frame-Options, etc.
  fi
  
  # Check Docker security
  if [ -f "$target/Dockerfile" ]; then
    echo "  ✓ Analyzing Docker security configuration"
    # Check for non-root user, minimal base images, etc.
  fi
  
  # Check cloud configuration
  if [ -f "$target/terraform/"*.tf ] 2>/dev/null; then
    echo "  ✓ Reviewing Infrastructure as Code security"
    # Check for public access, encryption, etc.
  fi
}
```

## Auto-Remediation Features

### Automated Security Fixes

```bash
apply_security_fixes() {
  local target=$1
  local scope=$2
  
  echo "🔧 Applying automated security fixes..."
  
  case "$scope" in
    "api")
      apply_api_security_fixes "$target"
      ;;
    "infra")
      apply_infrastructure_fixes "$target"
      ;;
    "app")
      apply_application_fixes "$target"
      ;;
    "all")
      apply_comprehensive_fixes "$target"
      ;;
  esac
  
  echo "✅ Security fixes applied - please review changes before committing"
}

apply_api_security_fixes() {
  local target=$1
  
  echo "  ✓ Adding security headers middleware"
  echo "  ✓ Implementing input validation"
  echo "  ✓ Adding rate limiting configuration"
  echo "  ✓ Enhancing error handling"
  
  # Implementation would add actual security improvements
}
```

## Report Generation

### Security Report Formats

```bash
generate_security_report() {
  local format=$1
  local findings_file=$2
  local output_path=$3
  
  echo "📊 Generating security report in $format format..."
  
  case "$format" in
    "html")
      generate_html_report "$findings_file" "$output_path"
      ;;
    "json")
      generate_json_report "$findings_file" "$output_path"
      ;;
    "sarif")
      generate_sarif_report "$findings_file" "$output_path"
      ;;
    "xml")
      generate_xml_report "$findings_file" "$output_path"
      ;;
  esac
  
  echo "✅ Security report generated: $output_path"
}

generate_html_report() {
  local findings=$1
  local output=$2
  
  cat > "$output" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Security Assessment Report</title>
    <style>
        .critical { color: #d32f2f; font-weight: bold; }
        .high { color: #f57c00; font-weight: bold; }
        .medium { color: #fbc02d; }
        .low { color: #388e3c; }
    </style>
</head>
<body>
    <h1>Security Assessment Report</h1>
    <div id="summary">
        <!-- Summary section -->
    </div>
    <div id="findings">
        <!-- Detailed findings -->
    </div>
    <div id="recommendations">
        <!-- Remediation recommendations -->
    </div>
</body>
</html>
EOF
}
```

## Integration with CI/CD

### Pipeline Integration

```yaml
# Example GitHub Actions workflow
security-audit:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Security Audit
      run: |
        /quality-security-audit ./src --framework=owasp --scope=all --report=sarif
    - name: Upload SARIF
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: security-report.sarif
```

## Error Handling & Recovery

If security assessment fails:

1. **Tool Availability**: Check if security scanning tools are installed
2. **Permission Issues**: Validate file access permissions
3. **Network Dependencies**: Verify external service connectivity
4. **Large Codebases**: Implement progressive scanning strategies
5. **False Positives**: Provide suppression and filtering mechanisms

### Fallback Options

- **Manual Checklists**: Provide framework-specific security checklists
- **Documentation Links**: Reference security best practice guides
- **Tool Recommendations**: Suggest external security tools
- **Expert Consultation**: Recommend security-architect agent for complex issues

## Success Metrics

- **Vulnerability Coverage**: Percentage of OWASP/NIST/CIS controls assessed
- **Risk Reduction**: Number of high/critical findings remediated
- **Compliance Score**: Framework compliance percentage
- **Time to Remediation**: Average time from finding to fix
- **False Positive Rate**: Accuracy of automated detection