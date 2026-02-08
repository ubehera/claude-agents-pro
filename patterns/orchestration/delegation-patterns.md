# Agent Delegation Patterns

## Core Delegation Patterns

### 1. Direct Delegation
**Pattern**: Single agent handles complete task
```yaml
Task:
  description: "Design REST API for user management"
  subagent_type: "api-platform-engineer"
  context: "User registration, authentication, profile management"
```

**When to use**:
- Task fits clearly within one agent's expertise
- No complex dependencies or coordination needed
- Quality requirements can be met by single specialist

### 2. Sequential Delegation
**Pattern**: Chain of dependent tasks across multiple agents
```yaml
Step_1:
  description: "Design system architecture for e-commerce platform"
  subagent_type: "system-design-specialist"

Step_2:
  description: "Create API specifications based on system architecture"
  subagent_type: "api-platform-engineer"
  dependencies: [Step_1]

Step_3:
  description: "Implement security measures for the APIs"
  subagent_type: "security-architect"
  dependencies: [Step_1, Step_2]
```

**When to use**:
- Tasks have clear dependencies
- Output of one task directly feeds into next
- Sequential validation is required

### 3. Parallel Delegation
**Pattern**: Independent tasks executed simultaneously
```yaml
Frontend_Task:
  description: "Design React component library"
  subagent_type: "frontend-expert"

Backend_Task:
  description: "Implement microservices architecture"
  subagent_type: "full-stack-architect"

Security_Task:
  description: "Perform security threat modeling"
  subagent_type: "security-architect"

# All can execute in parallel
```

**When to use**:
- Tasks are independent and don't require each other's outputs
- Time-to-completion needs optimization
- Different aspects of same project can be developed concurrently

### 4. Review & Validation Chain
**Pattern**: Primary work + quality validation
```yaml
Primary_Work:
  description: "Develop machine learning pipeline"
  subagent_type: "machine-learning-engineer"

Performance_Review:
  description: "Optimize ML pipeline performance"
  subagent_type: "performance-optimization-specialist"
  dependencies: [Primary_Work]

Security_Review:
  description: "Review ML pipeline for security vulnerabilities"
  subagent_type: "security-architect"
  dependencies: [Primary_Work]

Quality_Gate:
  description: "Final quality assessment and integration"
  subagent_type: "orchestration-coordinator"
  dependencies: [Performance_Review, Security_Review]
```

**When to use**:
- High-quality output requirements
- Cross-domain validation needed
- Multiple expertise areas must validate work

### 5. Iterative Refinement
**Pattern**: Cycles of development and improvement
```yaml
Iteration_1:
  description: "Create initial API design"
  subagent_type: "api-platform-engineer"

Review_1:
  description: "Review API design for system integration"
  subagent_type: "system-design-specialist"
  dependencies: [Iteration_1]

Iteration_2:
  description: "Refine API design based on review feedback"
  subagent_type: "api-platform-engineer"
  dependencies: [Review_1]
  context: "Incorporate feedback from system design review"
```

**When to use**:
- Complex problems requiring multiple refinement cycles
- Feedback loops essential for quality
- Incremental improvement approach preferred

## Context Preparation Patterns

### 1. Rich Context Pattern
**Description**: Provide comprehensive background information
```yaml
Task:
  description: "Optimize database queries for analytics dashboard"
  subagent_type: "performance-optimization-specialist"
  context: |
    Database: PostgreSQL 15
    Current performance: 5-10 second query times
    Data volume: 50M records, growing 1M/month
    User requirements: <2 second response time
    Existing indexes: user_id, created_at, status
    Query patterns: time-series aggregations, user filtering
    Infrastructure: AWS RDS db.r6g.2xlarge

    Previous optimization attempts:
    - Added composite indexes (limited improvement)
    - Query plan analysis shows sequential scans
    - Memory usage at 70% of allocated
```

### 2. Constraint-Based Context
**Description**: Define clear boundaries and limitations
```yaml
Task:
  description: "Design microservices architecture"
  subagent_type: "system-design-specialist"
  constraints:
    budget: "$50K/month infrastructure cost"
    compliance: "SOC2, HIPAA requirements"
    performance: "<100ms p95 latency"
    team_size: "5 developers, 2 DevOps"
    timeline: "6 months to MVP"
    existing_tech: "PostgreSQL, Redis, AWS"
    must_support: "10K concurrent users"
```

### 3. Example-Driven Context
**Description**: Provide concrete examples and references
```yaml
Task:
  description: "Create React component design system"
  subagent_type: "frontend-expert"
  examples:
    design_reference: "Material-UI, Ant Design patterns"
    brand_guidelines: "Primary: #1976d2, Secondary: #dc004e"
    component_examples:
      - "Button with loading states, sizes, variants"
      - "Data table with sorting, filtering, pagination"
      - "Form inputs with validation and error states"
    accessibility: "WCAG 2.1 AA compliance required"
```

## Error Recovery Patterns

### 1. Graceful Fallback
**Pattern**: Automatic fallback to more general agent
```yaml
Primary_Attempt:
  description: "Optimize Rust performance for real-time system"
  subagent_type: "rust-systems-engineer"

Fallback_On_Error:
  description: "General performance optimization guidance"
  subagent_type: "performance-optimization-specialist"
  trigger: "rust-systems-engineer unavailable or error"
```

### 2. Expert Escalation
**Pattern**: Escalate to higher-tier agent for complex issues
```yaml
Specialist_Attempt:
  description: "Debug distributed system performance issue"
  subagent_type: "performance-optimization-specialist"

Expert_Escalation:
  description: "Complex distributed system architecture analysis"
  subagent_type: "system-design-specialist"
  trigger: "performance issue requires architectural changes"
```

### 3. Multi-Agent Consultation
**Pattern**: Get multiple perspectives on difficult problems
```yaml
Problem_Analysis:
  description: "Diagnose intermittent production failures"

Perspectives:
  - subagent_type: "devops-automation-expert"
    focus: "Infrastructure and deployment issues"
  - subagent_type: "performance-optimization-specialist"
    focus: "Performance bottlenecks and resource issues"
  - subagent_type: "security-architect"
    focus: "Security-related failure modes"

Synthesis:
  description: "Synthesize findings into root cause analysis"
  subagent_type: "orchestration-coordinator"
  dependencies: [all perspectives]
```

## Quality Gate Patterns

### 1. Progressive Quality Gates
**Pattern**: Multiple quality checkpoints
```yaml
Development:
  description: "Build feature implementation"
  subagent_type: "full-stack-architect"

Code_Review:
  description: "Review code quality and patterns"
  subagent_type: "code-reviewer"
  quality_threshold: 7.5

Security_Review:
  description: "Security vulnerability assessment"
  subagent_type: "security-architect"
  quality_threshold: 8.0

Performance_Review:
  description: "Performance impact analysis"
  subagent_type: "performance-optimization-specialist"
  quality_threshold: 7.8

Final_Gate:
  description: "Final integration and deployment readiness"
  subagent_type: "devops-automation-expert"
  dependencies: [Code_Review, Security_Review, Performance_Review]
  quality_threshold: 8.2
```

### 2. Domain Cross-Validation
**Pattern**: Experts from different domains validate work
```yaml
Primary_Work:
  description: "Design OAuth 2.0 implementation"
  subagent_type: "security-architect"

API_Validation:
  description: "Validate OAuth implementation against API design principles"
  subagent_type: "api-platform-engineer"
  focus: "API usability and developer experience"

System_Validation:
  description: "Validate OAuth integration with overall system architecture"
  subagent_type: "system-design-specialist"
  focus: "Scalability and system integration"
```

## Communication Patterns

### 1. Structured Handoffs
**Description**: Clear format for passing work between agents
```yaml
Handoff_Format:
  summary: "Brief description of work completed"
  deliverables:
    - "Specific artifacts created"
    - "Documentation provided"
  next_steps: "Recommended actions for receiving agent"
  context: "Important background information"
  concerns: "Issues or risks to be aware of"
  validation: "How to verify the work quality"
```

### 2. Progress Reporting
**Description**: Regular status updates during long workflows
```yaml
Status_Report:
  task_id: "unique_identifier"
  progress: "percentage_complete"
  current_phase: "description_of_current_work"
  blockers: "issues_preventing_progress"
  next_milestone: "upcoming_completion_target"
  quality_metrics: "current_quality_scores"
  estimated_completion: "time_remaining"
```

### 3. Collaborative Decision Making
**Description**: Multiple agents contribute to decisions
```yaml
Decision_Framework:
  question: "Should we use REST or GraphQL for this API?"

Perspectives:
  - agent: "api-platform-engineer"
    recommendation: "GraphQL for flexibility"
    reasoning: "Complex data requirements, frontend flexibility needs"

  - agent: "performance-optimization-specialist"
    recommendation: "REST for simplicity"
    reasoning: "Better caching, simpler optimization, lower complexity"

  - agent: "system-design-specialist"
    recommendation: "Hybrid approach"
    reasoning: "REST for simple operations, GraphQL for complex queries"

Final_Decision:
  coordinator: "orchestration-coordinator"
  decision: "Based on project constraints and team expertise"
  rationale: "Synthesis of all perspectives with project context"
```

---

These patterns provide a comprehensive framework for effective agent delegation and coordination, ensuring optimal task distribution and high-quality collaborative outcomes.
### 6. Incident Recovery Loop
**Pattern**: Coordinate rapid mitigation and follow-up actions
```yaml
Incident_Command:
  description: "Lead response for checkout errors"
  subagent_type: "sre-incident-responder"

Diagnostics:
  description: "Diagnose service degradation"
  subagent_type: "error-diagnostician"
  dependencies: [Incident_Command]

Mitigation:
  description: "Roll back deployment and restore service"
  subagent_type: "devops-automation-expert"
  dependencies: [Incident_Command, Diagnostics]

Postmortem:
  description: "Capture lessons and action items"
  subagent_type: "observability-engineer"
  dependencies: [Mitigation]
```

**When to use**:
- High-severity incidents with customer impact
- Need coordinated response, rollback, and learning cycle
- Desire to codify incident metrics (MTTD/MTTR, error budgeting)
