---
name: threat-modeling
description: Apply STRIDE, PASTA, and MITRE ATT&CK frameworks to identify security threats, assess risks, and design mitigations for applications and systems.
tags:
  - security
  - threat-modeling
  - risk-assessment
  - stride
  - pasta
  - mitre-attack
category: security
version: 1.0.0
---

# Threat Modeling

Systematic threat identification and risk assessment using industry-standard frameworks (STRIDE, PASTA, MITRE ATT&CK) to build secure systems from design phase through deployment.

## When to Use This Skill

- **Design Phase**: Architecture review and threat identification
- **Security Reviews**: Assess existing systems for vulnerabilities
- **Incident Response**: Understand attack vectors and TTPs
- **Compliance**: Meet security standards (NIST, ISO 27001)
- **Risk Assessment**: Prioritize security investments
- **Threat Intelligence**: Map real-world threats to your systems

## Threat Modeling Frameworks

### STRIDE Framework

**Purpose**: Comprehensive threat categorization for applications

**Categories**:
- **S**poofing: Identity impersonation
- **T**ampering: Data modification
- **R**epudiation: Denial of actions
- **I**nformation Disclosure: Data exposure
- **D**enial of Service: Availability attacks
- **E**levation of Privilege: Unauthorized access

**Example Application**:
```yaml
Component: User Authentication API
Trust Boundary: Internet → API Gateway → Auth Service → Database

Threats:
  Spoofing:
    - Credential stuffing attacks
    - Session token theft
    - Phishing campaigns
    Mitigations:
      - Multi-factor authentication (MFA)
      - Rate limiting (10 attempts/hour)
      - CAPTCHA after 3 failures
      - JWT with short expiry (15 min)

  Tampering:
    - JWT token modification
    - Request parameter manipulation
    - Database injection
    Mitigations:
      - JWT signature verification (RS256)
      - Input validation with whitelists
      - Parameterized queries
      - Immutable audit logs

  Repudiation:
    - User denies password reset
    - Admin denies privilege escalation
    Mitigations:
      - Comprehensive audit logging
      - Non-repudiation via digital signatures
      - Tamper-proof log storage (WORM)

  Information Disclosure:
    - Sensitive data in logs
    - API enumeration
    - Error message leakage
    Mitigations:
      - PII masking in logs
      - Generic error messages
      - Rate limiting on enumeration
      - HTTPS with TLS 1.3

  Denial of Service:
    - Login endpoint flooding
    - Resource exhaustion
    - SlowLoris attacks
    Mitigations:
      - Rate limiting (per IP, per user)
      - Request size limits (1MB max)
      - Connection timeouts (30s)
      - CDN and WAF protection

  Elevation of Privilege:
    - IDOR vulnerabilities
    - Broken access control
    - Admin panel exposure
    Mitigations:
      - RBAC with least privilege
      - Authorization checks on every request
      - Admin endpoints on separate domain
      - Regular permission audits
```

### PASTA Framework (Process for Attack Simulation and Threat Analysis)

**Purpose**: Risk-centric, business-focused threat analysis

**7 Stages**:

**Stage 1: Define Business Objectives**
```yaml
Objectives:
  - Protect customer PII (GDPR compliance)
  - Maintain 99.9% uptime SLA
  - Process payments securely (PCI-DSS)
  - Prevent data breaches (< $1M liability)

Impact Scoring:
  - Critical: Revenue loss, regulatory fines
  - High: Reputation damage, customer churn
  - Medium: Service degradation
  - Low: Minor inconvenience
```

**Stage 2: Define Technical Scope**
```yaml
In-Scope:
  - Web application (React SPA)
  - REST API (Node.js/Express)
  - PostgreSQL database
  - Redis cache
  - AWS infrastructure (EC2, RDS, S3)
  - Third-party integrations (Stripe, SendGrid)

Out-of-Scope:
  - Mobile applications (separate assessment)
  - Internal admin tools
  - Legacy mainframe system
```

**Stage 3: Application Decomposition**
```yaml
Components:
  Frontend:
    - React SPA hosted on CloudFront
    - Client-side validation
    - JWT token storage (httpOnly cookies)

  API Layer:
    - Express.js REST API
    - Authentication middleware
    - Rate limiting (express-rate-limit)
    - Input validation (Joi schemas)

  Data Layer:
    - PostgreSQL (RDS with encryption)
    - Redis (session storage)
    - S3 (file uploads with presigned URLs)

Data Flow:
  User Registration:
    1. User submits form → Frontend validation
    2. HTTPS POST → API Gateway → WAF inspection
    3. API validates input → Checks email uniqueness
    4. Hashes password (bcrypt, 12 rounds)
    5. Stores user in database
    6. Sends verification email (SendGrid)
    7. Returns JWT token (httpOnly cookie)

Trust Boundaries:
  - Internet ↔ CloudFront CDN
  - CloudFront ↔ API Gateway
  - API Gateway ↔ Application servers
  - Application ↔ Database
  - Application ↔ Third-party APIs
```

**Stage 4: Threat Analysis**
```yaml
Authentication Service:
  Entry Points:
    - /api/auth/register
    - /api/auth/login
    - /api/auth/reset-password

  Assets:
    - User credentials (passwords, tokens)
    - Session data
    - PII (email, name, phone)

  Attack Vectors:
    - Credential stuffing (automated login attempts)
    - Brute force attacks
    - Session hijacking
    - Password reset token theft
    - SQL injection via login form

  Threat Actors:
    - Script kiddies (automated tools)
    - Competitors (data theft)
    - Nation-state actors (if high-value target)
    - Insider threats (disgruntled employees)
```

**Stage 5: Vulnerability & Weakness Analysis**
```yaml
Technical Vulnerabilities:
  - CWE-89: SQL Injection (if not using parameterized queries)
  - CWE-79: Cross-Site Scripting (XSS)
  - CWE-352: Cross-Site Request Forgery (CSRF)
  - CWE-307: Improper restriction of authentication attempts
  - CWE-798: Use of hardcoded credentials

Configuration Weaknesses:
  - Default admin credentials
  - Unencrypted database connections
  - Overly permissive CORS policy
  - Missing security headers
  - Outdated dependencies with CVEs
```

**Stage 6: Attack Modeling**
```yaml
Attack Scenario 1: Credential Stuffing
  Prerequisites:
    - Attacker has leaked credentials from another breach
    - No rate limiting on login endpoint

  Attack Steps:
    1. Attacker obtains credential list (1M+ emails/passwords)
    2. Uses automated tool (SentryMBA, SNIPR)
    3. Tests credentials at /api/auth/login
    4. Successfully authenticates as 2-5% of users
    5. Exfiltrates data or performs fraudulent actions

  Impact: High (account takeover, data breach)
  Likelihood: High (common attack)
  Risk Score: Critical

  Mitigations:
    - Rate limiting: 5 attempts/hour per IP
    - CAPTCHA after 2 failed attempts
    - MFA enforcement
    - Anomaly detection (impossible travel)
    - Credential stuffing detection (DeviceID tracking)

Attack Scenario 2: SQL Injection
  Prerequisites:
    - Dynamic SQL queries with string concatenation
    - No input validation

  Attack Steps:
    1. Attacker identifies vulnerable parameter
    2. Injects payload: ' OR '1'='1' --
    3. Bypasses authentication
    4. Escalates to UNION-based injection
    5. Exfiltrates entire database

  Impact: Critical (full database compromise)
  Likelihood: Medium (if poor coding practices)
  Risk Score: Critical

  Mitigations:
    - Parameterized queries (100% coverage)
    - ORM usage (Sequelize, TypeORM)
    - Input validation with whitelists
    - Web Application Firewall (WAF)
    - Database user with minimal privileges
```

**Stage 7: Risk & Impact Analysis**
```yaml
Risk Matrix:
  Critical (P1): Address immediately (0-7 days)
    - SQL injection vulnerabilities
    - Authentication bypass
    - Hardcoded secrets in code
    - Remote code execution (RCE)

  High (P2): Address within 30 days
    - Missing MFA enforcement
    - Weak password policy
    - Insecure session management
    - Sensitive data exposure

  Medium (P3): Address within 90 days
    - Missing security headers
    - Verbose error messages
    - Outdated dependencies
    - Insufficient logging

  Low (P4): Address opportunistically
    - Information disclosure (version numbers)
    - Missing rate limiting (non-critical endpoints)
    - Code quality issues

Business Impact:
  Data Breach (100K records):
    - Regulatory fines: $500K - $2M (GDPR)
    - Legal costs: $200K - $500K
    - Reputation damage: 20-30% customer churn
    - Remediation costs: $100K - $300K
    - Total impact: $1M - $3.5M

  Service Downtime (24 hours):
    - Revenue loss: $50K/hour × 24 = $1.2M
    - SLA penalties: $100K
    - Customer refunds: $50K
    - Total impact: $1.35M
```

### MITRE ATT&CK Framework

**Purpose**: Real-world adversary tactics, techniques, and procedures (TTPs)

**Tactics (What adversaries are trying to do)**:

```yaml
Initial Access:
  T1190 - Exploit Public-Facing Application:
    Description: SQL injection in login form
    Detection: WAF logs, anomaly detection
    Mitigation: Input validation, parameterized queries

  T1078 - Valid Accounts:
    Description: Credential stuffing with leaked passwords
    Detection: Impossible travel, unusual login times
    Mitigation: MFA, rate limiting, anomaly detection

Execution:
  T1059 - Command and Scripting Interpreter:
    Description: OS command injection via file upload
    Detection: Process monitoring, EDR alerts
    Mitigation: Input validation, sandboxing

Persistence:
  T1136 - Create Account:
    Description: Attacker creates backdoor admin account
    Detection: User creation audit logs
    Mitigation: Alert on new admin accounts, MFA

Privilege Escalation:
  T1068 - Exploitation for Privilege Escalation:
    Description: Exploit vulnerable npm package
    Detection: Runtime application monitoring
    Mitigation: Dependency scanning, least privilege

Defense Evasion:
  T1070 - Indicator Removal on Host:
    Description: Attacker deletes audit logs
    Detection: SIEM monitoring for log gaps
    Mitigation: Immutable log storage, WORM

Credential Access:
  T1110 - Brute Force:
    Description: Password spraying attack
    Detection: Multiple failed login attempts
    Mitigation: Account lockout, CAPTCHA

Discovery:
  T1083 - File and Directory Discovery:
    Description: Path traversal to list files
    Detection: Unusual file access patterns
    Mitigation: Input validation, chroot jail

Lateral Movement:
  T1021 - Remote Services:
    Description: SSH lateral movement using stolen keys
    Detection: Unusual SSH connections
    Mitigation: SSH key rotation, bastion hosts

Collection:
  T1005 - Data from Local System:
    Description: Database dump via SQL injection
    Detection: Large data transfers
    Mitigation: DLP, network segmentation

Exfiltration:
  T1041 - Exfiltration Over C2 Channel:
    Description: Data sent to attacker C2 server
    Detection: Outbound connections to unknown IPs
    Mitigation: Egress filtering, DLP

Impact:
  T1486 - Data Encrypted for Impact:
    Description: Ransomware encryption
    Detection: Unusual file system activity
    Mitigation: Backups, EDR, network segmentation
```

**ATT&CK Navigator Heatmap**:
```yaml
Focus Areas (Your Organization):
  Initial Access: High risk (public API)
  Credential Access: High risk (authentication service)
  Exfiltration: Medium risk (sensitive data)
  Impact: Medium risk (customer data)

Detection Coverage:
  Initial Access: 80% (WAF, IDS)
  Persistence: 60% (SIEM alerts)
  Privilege Escalation: 70% (EDR)
  Defense Evasion: 40% (gaps in monitoring)
  Credential Access: 90% (strong controls)
  Discovery: 50% (limited file monitoring)
  Lateral Movement: 60% (network segmentation)
  Collection: 70% (DLP)
  Exfiltration: 65% (egress monitoring)
  Impact: 80% (backups, incident response)
```

## Threat Modeling Process

### Step 1: Identify Assets
```yaml
Critical Assets:
  - User credentials (passwords, API keys)
  - Payment information (credit cards, bank accounts)
  - Personal data (PII under GDPR)
  - Business logic (proprietary algorithms)
  - Infrastructure (servers, databases)

Asset Valuation:
  - Credentials: High (account takeover)
  - Payment data: Critical (PCI-DSS, fraud)
  - PII: High (GDPR fines)
  - Business logic: Medium (competitive advantage)
  - Infrastructure: Medium (availability)
```

### Step 2: Create Architecture Diagram
```
┌─────────────────────────────────────────────────────────┐
│                    Internet (Untrusted)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              CloudFront CDN + WAF                        │
│  - DDoS protection                                       │
│  - SSL/TLS termination                                   │
│  - Bot detection                                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ (Trust Boundary 1)
┌─────────────────────────────────────────────────────────┐
│                 API Gateway                              │
│  - Rate limiting                                         │
│  - Authentication (JWT validation)                       │
│  - Request/response logging                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ (Trust Boundary 2)
┌─────────────────────────────────────────────────────────┐
│             Application Tier (Private Subnet)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Auth Service │  │ API Service  │  │ Worker Queue │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │            │
└─────────┼─────────────────┼─────────────────┼───────────┘
          │                 │                 │
          ▼ (Trust Boundary 3)                │
┌─────────────────────────────────────────────┼───────────┐
│              Data Tier (Private Subnet)     │           │
│  ┌──────────────┐  ┌──────────────┐  ┌─────▼────────┐  │
│  │ PostgreSQL   │  │ Redis Cache  │  │ S3 Storage   │  │
│  │ (encrypted)  │  │ (sessions)   │  │ (encrypted)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Step 3: Identify Entry Points
```yaml
External Entry Points:
  - Public API endpoints (/api/*)
  - File upload endpoints (/api/upload)
  - OAuth callback URLs (/auth/callback)
  - Webhook receivers (/webhooks/*)

Internal Entry Points:
  - Admin API (internal network only)
  - Database connections (application tier)
  - Inter-service communication (microservices)
  - Message queues (async processing)
```

### Step 4: Apply STRIDE to Each Component
```yaml
Component: File Upload API
  Spoofing:
    - Upload malicious file disguised as image
    - MIME type spoofing
  Tampering:
    - Modify file during transit
    - Replace file after upload
  Repudiation:
    - User denies uploading malicious file
  Information Disclosure:
    - Path traversal to read other files
    - Metadata leakage (EXIF data)
  Denial of Service:
    - Upload huge files (exhaust storage)
    - ZIP bomb attack
  Elevation of Privilege:
    - Upload webshell (PHP, JSP)
    - Execute uploaded script
```

### Step 5: Define Mitigations
```yaml
File Upload Mitigations:
  Input Validation:
    - Whitelist file extensions (.jpg, .png, .pdf)
    - Validate MIME type (magic numbers)
    - Max file size: 10MB
    - Filename sanitization (remove path traversal)

  Processing:
    - Scan with antivirus (ClamAV)
    - Strip metadata (ExifTool)
    - Re-encode images (prevent pixel flood)
    - Generate random filename (UUIDs)

  Storage:
    - Store outside webroot
    - S3 with presigned URLs (temporary access)
    - Separate domain for downloads (prevent XSS)

  Access Control:
    - Authenticate uploader
    - Authorize download requests
    - Rate limit uploads (5/hour)
```

## Practical Examples

### Example 1: OAuth 2.0 Authentication Flow

**Threat Model**:
```yaml
Data Flow:
  1. User clicks "Login with Google"
  2. Redirect to Google OAuth consent screen
  3. User approves, Google redirects to /auth/callback?code=xyz
  4. App exchanges code for access token
  5. App validates token and creates session

Threats (STRIDE):
  Spoofing:
    - Fake OAuth provider
    - Phishing for OAuth consent

  Tampering:
    - Authorization code interception
    - Token modification

  Repudiation:
    - User denies OAuth consent

  Information Disclosure:
    - Token leakage in URL (referrer header)
    - Token in browser history

  Denial of Service:
    - Callback endpoint flooding

  Elevation of Privilege:
    - Open redirect vulnerability
    - CSRF in OAuth flow

Mitigations:
  - HTTPS only (no HTTP)
  - Validate redirect_uri against whitelist
  - Use state parameter (CSRF protection)
  - Use PKCE for mobile apps
  - Short-lived authorization codes (10 min)
  - Rotate refresh tokens
  - Rate limit callback endpoint
  - Audit OAuth grants regularly
```

### Example 2: Payment Processing System

**PASTA Analysis**:
```yaml
Stage 1 - Business Objectives:
  - PCI-DSS compliance (mandatory)
  - Prevent fraud (< 0.1% transaction rate)
  - Maintain payment uptime (99.99%)

Stage 2 - Technical Scope:
  - Payment form (frontend)
  - Payment API (backend)
  - Stripe integration (third-party)
  - Fraud detection service

Stage 3 - Decomposition:
  Components:
    - Tokenization service (Stripe.js)
    - Payment processing API
    - Webhook receiver (payment events)
    - Fraud scoring engine

Stage 4 - Threat Analysis:
  Attack Vectors:
    - Card testing (stolen card validation)
    - Man-in-the-middle (intercept tokens)
    - Replay attacks (duplicate charges)
    - Webhook spoofing (fake payment events)

Stage 5 - Vulnerabilities:
  - Missing webhook signature verification
  - No rate limiting on payment attempts
  - Insufficient fraud detection
  - Weak idempotency implementation

Stage 6 - Attack Modeling:
  Card Testing Attack:
    1. Attacker has stolen card list
    2. Tests each card with $1 transaction
    3. No rate limiting → 1000 tests/minute
    4. Identifies valid cards
    5. Uses for larger fraud

  Mitigations:
    - Rate limit: 3 payment attempts/hour per IP
    - CAPTCHA after 2 failures
    - Velocity checks (transaction frequency)
    - BIN validation (card issuer checks)
    - 3D Secure (SCA) for EU cards

Stage 7 - Risk & Impact:
  Card Testing:
    - Impact: High (fraud liability, Stripe penalties)
    - Likelihood: High (common attack)
    - Risk: Critical
    - Priority: P1 (immediate fix)

  Webhook Spoofing:
    - Impact: Critical (fake payment confirmations)
    - Likelihood: Medium (requires knowledge)
    - Risk: High
    - Priority: P1 (immediate fix)
```

## Tools & Automation

### Threat Modeling Tools
```yaml
Microsoft Threat Modeling Tool:
  - Visual STRIDE analysis
  - Auto-generate threat reports
  - Integration with Azure DevOps

OWASP Threat Dragon:
  - Open-source alternative
  - Diagram-based modeling
  - Export to JSON/PNG

IriusRisk:
  - Automated threat modeling
  - Compliance mapping (GDPR, PCI-DSS)
  - Risk scoring and prioritization

ThreatModeler:
  - Collaborative platform
  - API-driven automation
  - CI/CD integration
```

### Automation Scripts
```bash
# Generate threat model from OpenAPI spec
threat-composer --input api-spec.yaml --framework stride --output threats.json

# Map vulnerabilities to MITRE ATT&CK
mitre-mapper --cve CVE-2024-1234 --tactics initial-access --output attack-map.json

# Validate threat coverage
threat-validator --threats threats.json --controls controls.yaml --report coverage.html
```

## Threat Modeling Checklist

```markdown
- [ ] Architecture diagram created with trust boundaries
- [ ] All entry points identified and documented
- [ ] Assets classified by sensitivity (critical, high, medium, low)
- [ ] Data flows mapped end-to-end
- [ ] STRIDE analysis completed for each component
- [ ] Threat scenarios documented with attack steps
- [ ] Risk scores calculated (impact × likelihood)
- [ ] Mitigations defined with owners and timelines
- [ ] Residual risk documented and accepted
- [ ] Threat model reviewed by security team
- [ ] Threat model integrated into development lifecycle
- [ ] Periodic reviews scheduled (quarterly)
```

## Best Practices

1. **Start Early**: Threat model during design phase, not after implementation
2. **Iterate**: Update threat model as system evolves
3. **Collaborate**: Include developers, architects, security team
4. **Focus on Risk**: Prioritize threats by business impact
5. **Be Practical**: Focus on realistic threats, not theoretical
6. **Document**: Keep threat models accessible and up-to-date
7. **Automate**: Integrate threat modeling into CI/CD pipeline
8. **Review**: Conduct threat model reviews before major releases
9. **Learn**: Analyze incidents and update threat models
10. **Compliance**: Map threats to regulatory requirements

## Integration with SDLC

```yaml
Requirements Phase:
  - Define security objectives
  - Identify compliance requirements
  - Document sensitive data handling

Design Phase:
  - Create architecture diagrams
  - Identify trust boundaries
  - Apply STRIDE framework
  - Document attack scenarios

Development Phase:
  - Implement mitigations
  - Security code review
  - SAST/DAST scanning
  - Unit tests for security controls

Testing Phase:
  - Penetration testing
  - Validate threat mitigations
  - Red team exercises

Deployment Phase:
  - Security monitoring
  - Incident response readiness
  - Audit logging enabled

Maintenance Phase:
  - Threat model updates
  - Vulnerability management
  - Threat intelligence integration
  - Post-incident reviews
```

## Resources

- **OWASP Threat Modeling Cheat Sheet**: https://cheatsheetseries.owasp.org/cheatsheets/Threat_Modeling_Cheat_Sheet.html
- **Microsoft STRIDE**: https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool
- **MITRE ATT&CK**: https://attack.mitre.org/
- **PASTA Framework**: https://versprite.com/blog/what-is-pasta-threat-modeling/

Threat modeling is a proactive security practice that identifies and mitigates risks before they become vulnerabilities. Apply these frameworks systematically to build defensible systems.
