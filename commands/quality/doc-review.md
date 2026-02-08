---
description: Technical documentation quality review and improvement recommendations
args: <doc-path> [--type adr|readme|api|architecture|runbook] [--level basic|standard|comprehensive]
tools: Task, Read, Write
model: sonnet
---

# /doc-review - Technical Documentation Quality Review

Reviews and provides improvement recommendations for technical documentation through the `technical-documentation-specialist` agent, ensuring clarity, completeness, consistency, and adherence to best practices.

## Purpose

This command validates technical documentation quality by reviewing:
- Architecture Decision Records (ADRs) for proper structure and rationale
- README files for completeness and accessibility
- API documentation for accuracy and usability
- Architecture diagrams for clarity and standards compliance
- Runbooks for operational readiness
- Technical specifications for precision and testability

## Usage

```bash
# Review specific documentation
/doc-review docs/adr/001-microservices-architecture.md --type adr

# Review README with comprehensive analysis
/doc-review README.md --level comprehensive

# Review API documentation
/doc-review docs/api/payment-service.md --type api

# Review all documentation in directory
/doc-review docs/ --type architecture --level standard
```

## Documentation Types

### Architecture Decision Records (ADR)
```bash
/doc-review docs/adr/003-database-choice.md --type adr
```

**Validation Criteria**:
- [ ] **Title**: Clear, concise, action-oriented (e.g., "Use PostgreSQL for transactional data")
- [ ] **Status**: Defined (Proposed, Accepted, Deprecated, Superseded by ADR-XXX)
- [ ] **Context**: Business and technical context clearly explained
- [ ] **Decision**: Specific choice documented with alternatives considered
- [ ] **Consequences**: Positive and negative implications identified
- [ ] **Rationale**: Why this decision over alternatives
- [ ] **References**: Links to related ADRs, specs, or discussions

**Common Issues Detected**:
- Missing "why" rationale (only describes "what" was decided)
- No alternatives considered or compared
- Consequences section only lists positives
- Stale status (accepted but implementation abandoned)

### README Documentation
```bash
/doc-review README.md --type readme --level comprehensive
```

**Validation Criteria**:
- [ ] **Overview**: Clear project purpose and value proposition
- [ ] **Installation**: Step-by-step setup instructions with prerequisites
- [ ] **Usage**: Basic examples and common workflows
- [ ] **Configuration**: Environment variables and config options documented
- [ ] **Architecture**: High-level system overview (optional C4 diagram)
- [ ] **Contributing**: Guidelines for contributors
- [ ] **Troubleshooting**: Common issues and solutions
- [ ] **Links**: References to detailed docs, API specs, licenses

**Common Issues Detected**:
- Assumes prior knowledge without defining prerequisites
- Outdated installation instructions
- No usage examples or only "hello world"
- Missing troubleshooting section

### API Documentation
```bash
/doc-review docs/api/user-service.yaml --type api
```

**Validation Criteria**:
- [ ] **OpenAPI/GraphQL Spec**: Valid schema with examples
- [ ] **Authentication**: Auth requirements clearly documented
- [ ] **Endpoints**: All routes with descriptions, parameters, responses
- [ ] **Error Handling**: Error codes with meanings and recovery guidance
- [ ] **Rate Limits**: Throttling policies documented
- [ ] **Versioning**: API version strategy explained
- [ ] **Examples**: Request/response examples for common use cases
- [ ] **SDKs**: Client library references if available

**Common Issues Detected**:
- Missing error response documentation
- No examples for complex request bodies
- Authentication flows unclear
- Versioning strategy not explained

### Architecture Documentation
```bash
/doc-review docs/architecture/system-overview.md --type architecture
```

**Validation Criteria**:
- [ ] **C4 Diagrams**: Context, Container, Component, Code (as appropriate)
- [ ] **Diagram Standards**: Clear notation, legend, consistent styling
- [ ] **Component Descriptions**: Purpose and responsibilities documented
- [ ] **Integration Points**: External system dependencies shown
- [ ] **Data Flow**: Request/response flows illustrated
- [ ] **Technology Choices**: Stack documented with rationale
- [ ] **Non-Functional Requirements**: Performance, security, scalability addressed
- [ ] **Deployment Model**: Infrastructure and scaling documented

**Common Issues Detected**:
- Diagrams not updated after system changes
- No legend or notation explanation
- Missing integration points with external systems
- Technology stack not documented

### Runbook Documentation
```bash
/doc-review docs/runbooks/incident-response.md --type runbook
```

**Validation Criteria**:
- [ ] **Trigger Conditions**: When to use this runbook
- [ ] **Prerequisites**: Required access, tools, permissions
- [ ] **Step-by-Step Procedures**: Clear, actionable instructions
- [ ] **Validation Checks**: How to verify each step succeeded
- [ ] **Rollback Procedures**: How to undo changes if needed
- [ ] **Escalation Path**: Who to contact for help
- [ ] **Common Issues**: Known problems and solutions
- [ ] **Testing Verification**: Runbook tested in staging

**Common Issues Detected**:
- Steps assume expert knowledge
- No validation checks between steps
- Missing rollback procedures
- Untested procedures

## Review Levels

### Basic
- **Focus**: Critical issues only
- **Scope**: Structure, completeness, accuracy
- **Output**: High-priority fixes

### Standard (Default)
- **Focus**: Comprehensive quality review
- **Scope**: Structure, clarity, consistency, examples
- **Output**: Prioritized improvement recommendations

### Comprehensive
- **Focus**: Deep analysis with best practices
- **Scope**: All criteria + accessibility, SEO, maintenance
- **Output**: Detailed report with examples and templates

## Output Format

```markdown
## Documentation Review: [Document Name]

### Overall Score: 85/100 (Good)

### Critical Issues (Fix Immediately)
- Missing authentication section in API docs
- Outdated installation commands (references v1.x, current is v2.x)

### Important Improvements (Recommended)
- Add troubleshooting section with common errors
- Include request/response examples for POST endpoints
- Update architecture diagram to reflect microservices split

### Enhancements (Nice to Have)
- Add visual diagrams for complex workflows
- Expand code examples with error handling
- Include links to related documentation

### Compliance Checklist
- [x] Follows template structure
- [x] Contains required sections
- [ ] Up-to-date with current implementation
- [x] Examples are testable
- [ ] Cross-references are valid

### Recommendations
1. **Immediate**: Fix authentication section, update versions
2. **This Sprint**: Add troubleshooting, update examples
3. **Backlog**: Enhanced diagrams, expanded examples
```

## Quality Gates

Documentation passes review when:

- [ ] **Critical issues**: 0 remaining
- [ ] **Required sections**: 100% present
- [ ] **Accuracy**: Code examples tested and working
- [ ] **Clarity**: Technical reviewer can follow without assistance
- [ ] **Currency**: Content matches current implementation
- [ ] **Accessibility**: Appropriate audience level (beginner/intermediate/expert)

## Integration with Workflows

### With `/review` Command
```bash
# Combined code and documentation review
/review PR-123 --include-docs

# This triggers:
# 1. /review for code quality
# 2. /doc-review for any changed .md or .yaml files
```

### With `/feature-development` Command
```bash
# Documentation validation as quality gate
/feature-development "payment integration"

# Automatically triggers /doc-review for:
# - Updated README sections
# - New API documentation
# - Architecture decision records
```

## Best Practices

1. **Review Early**: Validate documentation during development, not after
2. **Test Examples**: Ensure all code examples are executable and current
3. **Keep Updated**: Re-review documentation after major changes
4. **Audience Awareness**: Match detail level to target audience
5. **Link Generously**: Connect related docs for discoverability

## Examples

### ADR Review
```bash
/doc-review docs/adr/005-event-sourcing.md --type adr

# Checks for:
# - Clear decision statement
# - Alternatives considered (CRUD, Event Sourcing, CQRS)
# - Consequences documented (complexity, debugging, scalability)
# - Implementation guidance
```

### README Improvement
```bash
/doc-review README.md --level comprehensive

# Provides:
# - Structure recommendations
# - Missing sections identified
# - Example improvements
# - Accessibility enhancements
```

### API Documentation Validation
```bash
/doc-review docs/api/openapi.yaml --type api

# Validates:
# - Schema correctness
# - Complete endpoint documentation
# - Error response coverage
# - Example quality and accuracy
```

## Related Commands

- `/review` - Code quality review (can include documentation)
- `/feature-development` - Full feature pipeline with doc requirements
- `/workflow` - Complete DDD workflow with documentation gates

## Agent Coordination

This command delegates to `technical-documentation-specialist` (Tier 06-integration) which may coordinate with:
- `code-reviewer` for code example validation
- `api-platform-engineer` for API documentation accuracy
- `research-librarian` for documentation best practices

## Troubleshooting

**Too many findings**: Start with `--level basic` to focus on critical issues

**Documentation passes but users confused**: Request comprehensive review with audience consideration

**Examples don't work**: Use `/review` on code examples to validate correctness

**Stale documentation**: Set up automated doc review in CI/CD with `/doc-review` checks