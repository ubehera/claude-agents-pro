# Tier 07: Quality (Security & Compliance)

Quality tier agents provide security architecture, threat modeling, and compliance expertise. They protect systems through defense-in-depth strategies and ensure adherence to industry security standards.

## When to Use Quality Agents

Use these agents when you need to:
- **Model threats** using STRIDE, PASTA, or MITRE ATT&CK frameworks
- **Implement secure authentication** (OAuth 2.0, OIDC, WebAuthn, MFA)
- **Ensure compliance** with GDPR, PCI-DSS, SOC2, HIPAA, or ISO 27001
- **Review code for vulnerabilities** (OWASP Top 10, injection, XSS, SSRF)
- **Design security architecture** with Zero Trust and defense-in-depth patterns
- **Plan incident response** with automated playbooks and forensics

## Available Agents

### [security-architect](security-architect.md)
World-class security expert covering application security, threat modeling, OWASP Top 10, secure coding, authentication/authorization, encryption, compliance frameworks, vulnerability assessment, penetration testing, and incident response.

**Use when:** Security assessments, threat modeling, OAuth/OIDC implementation, compliance requirements (GDPR, PCI-DSS, SOC2), security code review, incident response planning, container/Kubernetes security hardening.

## Quick Selection Guide

| If you need to... | Use this agent |
|-------------------|----------------|
| Perform threat modeling | **security-architect** |
| Implement authentication/authorization | **security-architect** |
| Ensure regulatory compliance | **security-architect** |
| Review code for security vulnerabilities | **security-architect** |
| Design Zero Trust architecture | **security-architect** |
| Plan incident response | **security-architect** |

## Common Combinations

**Secure System Design:**
1. `system-design-specialist` (Tier 01) --> Architecture design
2. `security-architect` --> Threat model and security controls
3. `api-platform-engineer` (Tier 01) --> Secure API contracts
4. `code-reviewer` (Tier 01) --> Security-focused code review

**Compliance Implementation:**
1. `security-architect` --> Define compliance requirements and controls
2. `devops-automation-expert` (Tier 03) --> DevSecOps pipeline integration
3. `test-engineer` (Tier 01) --> Security test automation
4. `workflow-validator` (Tier 00) --> Validate compliance gates

**Production Security Hardening:**
1. `security-architect` --> Security assessment and threat model
2. `aws-cloud-architect` (Tier 03) --> Cloud security controls
3. `kubernetes-architect` (Tier 03) --> Container security policies
4. `observability-engineer` (Tier 03) --> Security monitoring and alerting

## Best Practices

- **Shift security left**: Involve `security-architect` during design, not after implementation
- **Threat model early**: Run STRIDE analysis before writing code for sensitive features
- **Automate security testing**: Integrate SAST/DAST into CI/CD with DevOps agents
- **Compliance is continuous**: Use `security-architect` for ongoing compliance validation, not just audits
- **Defense in depth**: Layer security controls across network, application, data, and identity
