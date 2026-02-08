---
name: openapi-spec-generation
description: Use when generating or refining OpenAPI 3.1 specifications with consistent schemas, examples, security definitions, and error contracts.
trigger_keywords: [openapi spec generation, swagger generation, openapi 3.1, api contract generation, schema-first api]
---

# OpenAPI Spec Generation

Use this skill to create consistent, implementation-ready API contracts.

## When to Use This Skill

- Defining new REST APIs before implementation
- Normalizing inconsistent endpoint documentation
- Generating machine-readable contracts for SDK and testing
- Building error and auth standards across services

## Core Concepts

- **Contract-first** beats endpoint-by-endpoint drift.
- **Reusable components** reduce duplication and inconsistency.
- **Examples are mandatory** for request and response payloads.
- **Error schema uniformity** enables predictable clients.

## Implementation Patterns

```yaml
openapi: 3.1.0
info:
  title: Orders API
  version: 1.0.0
paths:
  /orders/{id}:
    get:
      operationId: getOrderById
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Order found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Order'
        '404':
          description: Order not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Problem'
components:
  schemas:
    Problem:
      type: object
      required: [type, title, status]
      properties:
        type: { type: string }
        title: { type: string }
        status: { type: integer }
```

## Validation Checklist

- Spec validates with OpenAPI tooling in CI
- Operations have stable `operationId` values
- Error responses follow shared schema conventions
- Auth and rate-limit behavior are documented
