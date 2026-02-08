---
description: Design and implement REST/GraphQL APIs with OpenAPI specifications and validation
tools: Task, Read, Write, MultiEdit, WebFetch
model: sonnet
args: [endpoint] [spec-type]
---

# API Design & Implementation Command

Delegates to api-platform-engineer for comprehensive API design, implementation, and governance.

## Usage Patterns

### REST API Design
```
/api design users/auth REST
/api implement orders/checkout OpenAPI
/api validate payments/webhooks
```

### GraphQL API Design
```
/api schema user-management GraphQL
/api resolvers content-delivery
/api federation microservices GraphQL
```

### API Governance
```
/api review security-patterns
/api standards REST-guidelines
/api versioning strategy
```

## Command Logic

1. **Context Gathering**: Analyze existing API patterns and project structure
2. **Agent Delegation**: Route to api-platform-engineer with enriched context
3. **Specification Generation**: Create OpenAPI/GraphQL schemas with examples
4. **Implementation Guidance**: Generate idiomatic implementation patterns
5. **Validation**: Security, performance, and standards compliance
6. **Documentation**: Generate comprehensive API documentation

## Delegation Pattern

```yaml
Agent: api-platform-engineer
Context:
  - Project architecture and existing APIs
  - Security requirements and authentication patterns
  - Performance targets and scaling requirements
  - Team conventions and coding standards

Deliverables:
  - API specification (OpenAPI 3.1 or GraphQL SDL)
  - Implementation scaffolding
  - Security and validation rules
  - Testing strategies and examples
  - Documentation and usage guides
```

## Integration Points

- **security-architect**: Security review and threat modeling
- **performance-optimization-specialist**: Performance validation
- **test-engineer**: API testing strategy and automation
- **backend-architect**: Service integration patterns

## Quality Criteria

- **Specification Completeness**: 100% endpoint coverage with examples
- **Security Standards**: Authentication, authorization, input validation
- **Performance Targets**: <200ms P95 response time
- **Documentation Quality**: Complete usage examples and error scenarios