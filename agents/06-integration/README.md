# Tier 06: Integration (Research & Documentation)

Integration agents bridge the gap between knowledge discovery and technical communication. They handle research, requirements analysis, user-facing documentation, and documentation quality assurance across the agent ecosystem.

## When to Use Integration Agents

Use these agents when you need to:
- **Research authoritative sources** (RFCs, specs, vendor docs, standards)
- **Define requirements** with user stories and acceptance criteria
- **Create user-facing documentation** (guides, tutorials, API references)
- **Review and improve documentation quality** across project artifacts
- **Translate business needs** into actionable technical specifications

## Available Agents

### [research-librarian](research-librarian.md)
Research specialist for discovering, vetting, and summarizing authoritative sources. Prioritizes primary sources (RFCs, vendor docs, standards) and produces concise findings with citations and handoff links.

**Use when:** Exploratory research, comparative analysis, finding authoritative specs, vetting unknown URLs, standards research (GDPR, OAuth, Kubernetes benchmarks).

### [product-owner](product-owner.md)
Product ownership specialist for requirements analysis, user story creation, backlog management, and acceptance criteria definition. Bridges business needs and technical implementation.

**Use when:** Writing user stories, defining acceptance criteria (Given-When-Then), backlog prioritization (RICE, MoSCoW), feature roadmap planning, sprint planning.

### [tech-writer](tech-writer.md)
Technical documentation specialist for end-user guides, API documentation, developer onboarding, tutorials, and troubleshooting guides. Expert in documentation-as-code and content strategy.

**Use when:** Creating user guides, API references, quickstart tutorials, troubleshooting docs, setting up documentation platforms (Docusaurus, GitBook).

### [technical-documentation-specialist](technical-documentation-specialist.md)
Documentation quality reviewer for ADRs, READMEs, API specs, architecture diagrams, and runbooks. Enhances structure, clarity, and consistency without creating domain content.

**Use when:** Reviewing documentation quality, improving ADR clarity, standardizing README structure, enhancing API doc completeness, creating documentation style guides.

## Quick Selection Guide

| If you need to... | Use this agent |
|-------------------|----------------|
| Research specs and standards | **research-librarian** |
| Write user stories and requirements | **product-owner** |
| Create user-facing documentation | **tech-writer** |
| Review and improve existing docs | **technical-documentation-specialist** |

## Common Combinations

**New Feature Lifecycle:**
1. `research-librarian` --> Discover relevant standards and best practices
2. `product-owner` --> Define user stories and acceptance criteria
3. Development agents --> Implement the feature
4. `tech-writer` --> Create user-facing documentation
5. `technical-documentation-specialist` --> Review doc quality

**Documentation Overhaul:**
1. `technical-documentation-specialist` --> Audit existing docs for gaps
2. `tech-writer` --> Create or rewrite documentation
3. `technical-documentation-specialist` --> Final quality review

**Requirements Gathering:**
1. `research-librarian` --> Research domain standards and constraints
2. `product-owner` --> Translate findings into user stories
3. `workflow-validator` (Tier 00) --> Validate requirements completeness

## Best Practices

- **Research before building**: Use `research-librarian` to ground decisions in authoritative sources
- **Requirements before code**: Use `product-owner` to define clear acceptance criteria upfront
- **Write docs with code**: Use `tech-writer` during implementation, not after
- **Separate creation from review**: Use `tech-writer` to create, `technical-documentation-specialist` to review
