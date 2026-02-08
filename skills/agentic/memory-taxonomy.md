---
name: memory-taxonomy
description: Load when designing or managing agent memory systems including knowledge graphs, session continuity, and context management
trigger_keywords: [agent memory, knowledge graph, session continuity, memory management, entity storage, context persistence, memory taxonomy, memory patterns]
---

# Agent Memory Taxonomy Skill

Patterns for designing and managing agent memory systems that persist knowledge across sessions and enable context-aware interactions.

## Overview

Agent memory transforms stateless LLM interactions into continuous, context-aware workflows. This skill covers memory categories, storage patterns, and maintenance strategies.

**When to Use**:
- Designing memory systems for agent frameworks
- Managing knowledge graph entities and relations
- Implementing session continuity across conversations
- Deciding what to remember vs what to forget

## Memory Categories

### 1. Episodic Memory (What Happened)

Facts about specific events, decisions, and interactions.

```yaml
Entity Types:
  - ArchitecturalDecision: "Chose PostgreSQL over MongoDB for ACID compliance"
  - ReviewOutcome: "PR #45 approved with minor suggestions on error handling"
  - WorkflowPhase: "Phase 3 completed: API contracts finalized"

Storage Pattern:
  - Store with timestamp for temporal queries
  - Include rationale (why, not just what)
  - Link to related entities (depends_on, follows)

Lifecycle:
  - Active: <6 months, referenced in current work
  - Archive: 6-12 months, retained for context
  - Prune: >12 months, unless explicitly important
```

### 2. Semantic Memory (What We Know)

General knowledge about projects, patterns, and domains.

```yaml
Entity Types:
  - Project: "trading-platform: Python/FastAPI, PostgreSQL, React frontend"
  - CodePattern: "Repository pattern used for all data access layers"
  - DomainModel: "Order aggregate: root entity with line items, status machine"

Storage Pattern:
  - Update rather than append (keep current)
  - Use observations for evolving knowledge
  - Cross-reference with related projects/patterns

Lifecycle:
  - Evergreen: Updated when new information arrives
  - Supersede: Old observations replaced by newer ones
  - Never auto-prune: Requires explicit invalidation
```

### 3. Procedural Memory (How To Do Things)

Learned workflows, preferences, and effective approaches.

```yaml
Entity Types:
  - UserPreference: "Prefers conventional commits, TypeScript strict mode"
  - Implementation: "Auth implemented with JWT + refresh tokens, 1h/7d expiry"
  - QualityValidation: "Coverage threshold: 85% for production, 70% for prototype"

Storage Pattern:
  - Encode as reusable patterns with context
  - Include failure modes ("X didn't work because Y")
  - Link to successful implementations

Lifecycle:
  - Active: Referenced in current workflows
  - Updated: When preferences or approaches change
  - Retained: Failure patterns (what not to do) kept indefinitely
```

## Entity Design Patterns

### Entity Creation

```yaml
Good Entity:
  name: "AuthStrategy-TradingPlatform"
  entityType: "ArchitecturalDecision"
  observations:
    - "JWT for stateless auth, refresh tokens for session extension"
    - "Access token: 1h expiry, refresh token: 7d"
    - "Rationale: Stateless for API scaling, refresh for UX"
    - "Decision date: 2024-01-15"

Bad Entity:
  name: "auth"                    # Too vague
  entityType: "Decision"          # Non-standard type
  observations:
    - "JWT"                       # No context
```

### Relation Design

```yaml
Directional Relations (use active voice):
  depends_on:      A depends on B (dependency)
  implements:      A implements B (spec → code)
  uses:            A uses B (consumption)
  extends:         A extends B (inheritance)
  follows:         A follows B (sequential)
  contributes_to:  A contributes to B (part of larger whole)
  validates:       A validates B (testing/review)

Example Graph:
  UserService ──depends_on──→ AuthModule
  AuthModule ──implements──→ JWTStrategy
  LoginEndpoint ──uses──→ AuthModule
  AuthTests ──validates──→ AuthModule
  Phase2 ──follows──→ Phase1
  APIDesign ──contributes_to──→ TradingPlatform
```

### Observation Hygiene

```yaml
When to Add Observations:
  ✅ Key decision with rationale
  ✅ Learned pattern or anti-pattern
  ✅ User preference discovered
  ✅ Phase outcome or milestone
  ✅ Error resolution (what worked)

When NOT to Add:
  ❌ Transient state ("currently running tests")
  ❌ Obvious facts ("JavaScript is a language")
  ❌ Duplicate of existing observation
  ❌ Raw data without interpretation
```

## Memory Maintenance

### Periodic Cleanup

```yaml
Session Start:
  1. Search for project-relevant entities
  2. Check for stale observations (>6 months without update)
  3. Remove superseded decisions (new ADR replaces old)

After Major Milestone:
  1. Store milestone entity with outcomes
  2. Update project entity with current state
  3. Archive completed phase entities
  4. Create forward-looking observations for next phase

Context Window Pressure (>50 messages):
  1. Immediately store critical decisions in memory
  2. Create TodoWrite for remaining work
  3. Summarize session learnings as observations
```

### Conflict Resolution

```yaml
When Contradictory Observations Exist:
  1. Check timestamps — newer observation likely supersedes
  2. Check context — may be valid in different contexts
  3. If truly contradictory:
     - Delete the outdated observation
     - Add clarifying observation to the surviving one
     - Example: "Supersedes previous decision to use MongoDB (see AuthStrategy-v1)"
```

## Query Patterns

```yaml
Project Context:
  search_nodes({ query: "trading platform architecture" })
  → Returns: Project, ArchitecturalDecision, DomainModel entities

Decision History:
  search_nodes({ query: "authentication decision JWT" })
  → Returns: ArchitecturalDecision entities with auth-related observations

Pattern Recall:
  search_nodes({ query: "repository pattern data access" })
  → Returns: CodePattern entities with implementation details

User Preferences:
  open_nodes({ names: ["UserPreferences-Umank"] })
  → Returns: Specific entity with all preference observations
```

## Best Practices

1. **Name entities uniquely** — include project or domain context in name
2. **Use standard entity types** — stick to the defined taxonomy
3. **Active voice relations** — "A depends_on B", not "B is depended on by A"
4. **Rationale always** — store WHY, not just WHAT
5. **Update over append** — modify observations rather than duplicating
6. **Prune proactively** — remove stale entities before they pollute search results

---

**Skill Type**: Agentic — Memory
**Complexity**: Moderate
**Typical Usage**: Agent memory system design, knowledge graph management, session continuity
