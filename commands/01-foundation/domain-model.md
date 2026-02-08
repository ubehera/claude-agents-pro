---
description: Strategic DDD domain modeling with event storming and bounded context identification
args: <domain-context> [--technique event-storming|context-mapping|aggregate-design] [--output artifacts|glossary|diagrams]
tools: Task, Read, Write
model: sonnet
---

# /domain-model - Strategic Domain-Driven Design Modeling

Facilitates strategic Domain-Driven Design activities through the `domain-modeling-expert` agent, including event storming, bounded context identification, context mapping, and ubiquitous language development.

## Purpose

This command helps translate business requirements into well-structured domain models by:
- Facilitating event storming sessions (Big Picture, Process-Level, Design-Level)
- Identifying and defining bounded contexts with clear boundaries
- Creating context maps showing relationships and integration patterns
- Developing ubiquitous language and domain glossaries
- Designing aggregates with proper invariants and transactional boundaries

## Usage

```bash
# Full domain analysis
/domain-model "e-commerce platform" --technique event-storming --output artifacts

# Bounded context identification
/domain-model "payment processing" --technique context-mapping

# Aggregate design for specific context
/domain-model "order management" --technique aggregate-design
```

## Techniques

### Event Storming
```bash
/domain-model "user management" --technique event-storming
```
Facilitates event storming sessions to discover:
- Domain events (past-tense: OrderPlaced, PaymentProcessed)
- Commands triggering events (PlaceOrder, ProcessPayment)
- Aggregates owning state changes
- Policies (automation rules)
- Read models and projections

**Output**: Event flow diagrams, bounded context candidates, initial aggregate identification

### Context Mapping
```bash
/domain-model "entire system" --technique context-mapping
```
Identifies bounded contexts and their relationships:
- Upstream/downstream dependencies
- Integration patterns (ACL, OHS, Conformist, Customer-Supplier)
- Shared Kernel identification
- Published Language definition

**Output**: Context map diagram, integration strategy, translation layer recommendations

### Aggregate Design
```bash
/domain-model "order context" --technique aggregate-design
```
Designs aggregates within a bounded context:
- Aggregate root selection
- Invariant identification and enforcement
- Transactional boundary definition
- Entity and value object composition

**Output**: Aggregate specifications, invariant rules, consistency boundaries

## Output Formats

### Artifacts (Default)
Produces tangible deliverables:
- Markdown documents with event flows
- Mermaid diagrams for visualization
- Context maps and integration patterns
- Architecture decision records (ADRs)

### Glossary
Focuses on ubiquitous language:
- Term definitions aligned with domain experts
- Context-specific vocabulary
- Concept relationships and hierarchies
- Ambiguity resolution

### Diagrams
Visual representations:
- Event flow diagrams
- Bounded context maps
- Aggregate relationship diagrams
- C4 context and container diagrams

## Integration with DDD Workflow

This command maps to **Phase 2 (Domain Modeling)** in the DDD workflow:

```yaml
Phase_1_Requirements: User provides business requirements
  ↓
Phase_2_Domain_Modeling: /domain-model identifies contexts & aggregates
  ↓
Phase_3_Architecture: System design based on bounded contexts
  ↓
Phase_4_API: API contracts aligned with domain model
```

## Quality Gates

Before proceeding from domain modeling to architecture:

- [ ] **Bounded contexts identified**: Each context has clear responsibilities
- [ ] **Ubiquitous language defined**: Glossary with 80%+ domain coverage
- [ ] **Context relationships mapped**: Integration patterns documented
- [ ] **Aggregates designed**: Root entities, invariants, boundaries defined
- [ ] **Events cataloged**: Core domain events identified and named
- [ ] **Validation**: Domain experts have reviewed and approved model

## Examples

### E-Commerce Platform
```bash
/domain-model "e-commerce platform" --technique event-storming

# Output includes:
# - Bounded Contexts: Catalog, Orders, Payments, Shipping, Inventory
# - Key Events: ProductAdded, OrderPlaced, PaymentProcessed, OrderShipped
# - Aggregates: Product, Order, Payment, Shipment
# - Context Map: Relationships and integration patterns
```

### Microservices Decomposition
```bash
/domain-model "monolithic application" --technique context-mapping

# Output includes:
# - Bounded context boundaries for service extraction
# - Upstream/downstream relationships
# - Shared data identification
# - Migration strategy recommendations
```

### Complex Aggregate Design
```bash
/domain-model "order management" --technique aggregate-design

# Output includes:
# - Order aggregate root with OrderLine entities
# - Invariants: total calculation, status transitions
# - Value objects: Money, Address, OrderStatus
# - Consistency boundaries and transaction scope
```

## Best Practices

1. **Start with Event Storming**: Begin with big-picture event storming to discover the domain
2. **Iterate on Context Boundaries**: Refine bounded contexts through multiple sessions
3. **Validate with Domain Experts**: Ensure ubiquitous language matches expert terminology
4. **Document Decisions**: Capture rationale for context boundaries in ADRs
5. **Keep It Simple**: Start with obvious contexts, split later if needed

## Related Commands

- `/api` - Create API contracts aligned with domain model
- `/workflow` - Execute full DDD workflow including domain modeling
- `/feature-development` - Implement features within established bounded contexts

## Agent Coordination

This command delegates to `domain-modeling-expert` (Tier 01-foundation) which may coordinate with:
- `system-design-specialist` for architectural implications
- `research-librarian` for DDD patterns and best practices
- `api-platform-engineer` for API contract alignment

## Troubleshooting

**Too many bounded contexts**: Start coarser, split later when complexity demands it

**Unclear boundaries**: Run event storming first to discover natural aggregation points

**Conflicting terminology**: Document context-specific meanings in glossary, use context maps to show translations

**Aggregate too large**: Look for transactional boundaries and consider splitting into multiple aggregates