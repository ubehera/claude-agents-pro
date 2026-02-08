# Tier 03: Specialist Agents

Specialist agents bring deep domain expertise in infrastructure, cloud, databases, DevOps, and operational excellence. Use them for platform-level decisions and specialized technical domains.

## When to Use Specialist Agents

Use these agents when you need to:
- **Design cloud infrastructure** with best practices
- **Set up CI/CD pipelines** and deployment automation
- **Architect databases** for scale and performance
- **Build observability** into systems
- **Respond to incidents** effectively
- **Design backend services** and APIs

## Available Agents

### [aws-cloud-architect](aws-cloud-architect.md)
AWS expert for CloudFormation, CDK, EC2, Lambda, ECS/EKS, S3, RDS, DynamoDB, VPC, IAM, Well-Architected Framework, and cloud migration strategies.

**Use when:** AWS infrastructure design, cloud architecture, production deployments.

### [azure-cloud-architect](azure-cloud-architect.md)
Azure expert for landing zones, Bicep/ARM, AKS, Azure Functions, Entra ID governance, and enterprise cloud migration.

**Use when:** Azure infrastructure design, platform governance, hybrid cloud execution.

### [gcp-cloud-architect](gcp-cloud-architect.md)
Google Cloud expert for GKE, Cloud Run, BigQuery, Vertex AI, data platform architecture, and organization policy design.

**Use when:** GCP platform design, data-intensive systems, AI-enabled cloud deployments.

### [backend-architect](backend-architect.md)
Principal backend architect for service design, REST/GraphQL APIs, event-driven systems, microservices decomposition, messaging, and resilience patterns.

**Use when:** End-to-end backend planning, implementation blueprints, modernization efforts.

### [database-architect](database-architect.md)
Senior database architect for relational modeling (PostgreSQL, MySQL), distributed data stores (DynamoDB, MongoDB), migration strategy, and performance tuning.

**Use when:** Schema design, query optimization, multi-region planning, compliance-ready storage.

### [data-pipeline-engineer](data-pipeline-engineer.md)
Data engineering expert for Apache Spark, Airflow, Kafka, ETL/ELT pipelines, data lakes, streaming processing, and data quality.

**Use when:** Data pipeline architecture, stream processing, data warehousing, feature engineering.

### [devops-automation-expert](devops-automation-expert.md)
DevOps expert for CI/CD pipelines (GitHub Actions, GitLab CI, Jenkins), infrastructure as code (Terraform, Ansible), GitOps, Kubernetes, and automation.

**Use when:** Pipeline setup, automation, deployment, infrastructure management, DevOps transformation.

### [full-stack-architect](full-stack-architect.md)
Full-stack expert for React, Next.js, Vue, Angular, Node.js, TypeScript, state management, API integration, authentication, and cloud deployment.

**Use when:** Web application architecture, frontend/backend development, modern JavaScript/TypeScript projects.

### [kubernetes-architect](kubernetes-architect.md)
Kubernetes specialist for cluster design, workload management, service mesh, security policies, and cloud-native patterns.

**Use when:** K8s cluster design, microservices deployment, container orchestration, service mesh implementation.

### [observability-engineer](observability-engineer.md)
Observability architect covering metrics (Prometheus, Datadog), logging (ELK, Loki), tracing (OpenTelemetry, Jaeger), SLO/SLA design, and alerting.

**Use when:** Instrumentation strategy, dashboard design, alert hygiene, reliability insights.

### [sre-incident-responder](sre-incident-responder.md)
Site reliability incident responder for high-severity production events—detection, triage, mitigation, post-incident analysis, and runbook automation.

**Use when:** On-call readiness, incident war room leadership, resilience upgrades.

### [terraform-expert](terraform-expert.md)
Infrastructure as Code specialist for Terraform modules, state management, provider configurations, and multi-cloud deployments.

**Use when:** IaC development, Terraform module design, infrastructure automation.

### [event-driven-architect](event-driven-architect.md)
Event-driven architecture specialist for message brokers, event sourcing, CQRS, and asynchronous system design.

**Use when:** Event-driven system design, messaging patterns, eventual consistency.

### [chaos-engineer](chaos-engineer.md)
Chaos engineering specialist for designing and executing resilience testing, failure injection, game days, and system reliability validation.

**Use when:** Chaos experiments, disaster recovery testing, building resilient systems.

## Quick Selection Guide

| If you need to... | Use this agent |
|-------------------|----------------|
| Design AWS infrastructure | **aws-cloud-architect** |
| Design Azure infrastructure | **azure-cloud-architect** |
| Design GCP infrastructure | **gcp-cloud-architect** |
| Build backend services | **backend-architect** |
| Design database schemas | **database-architect** |
| Build data pipelines | **data-pipeline-engineer** |
| Set up CI/CD | **devops-automation-expert** |
| Build full-stack apps | **full-stack-architect** |
| Deploy to Kubernetes | **kubernetes-architect** |
| Add observability | **observability-engineer** |
| Handle incidents | **sre-incident-responder** |
| Write Terraform | **terraform-expert** |
| Design event systems | **event-driven-architect** |
| Test system resilience | **chaos-engineer** |

## Common Combinations

**Production Infrastructure:**
1. `aws-cloud-architect` → Cloud design
2. `azure-cloud-architect` / `gcp-cloud-architect` → Alternative cloud providers
3. `kubernetes-architect` → Container orchestration
4. `terraform-expert` → IaC implementation
5. `observability-engineer` → Monitoring setup
6. `devops-automation-expert` → CI/CD pipelines

**Microservices Platform:**
1. `backend-architect` → Service design
2. `event-driven-architect` → Async communication
3. `database-architect` → Data per service
4. `kubernetes-architect` → Deployment

**Data Platform:**
1. `data-pipeline-engineer` → ETL/streaming
2. `database-architect` → Storage layer
3. `observability-engineer` → Data quality monitoring

## Best Practices

- **Start with architecture**: Use architects before implementation
- **Combine for platforms**: Infrastructure requires multiple specialists
- **Include observability early**: Don't add monitoring as an afterthought
- **Plan for incidents**: Involve `sre-incident-responder` in design reviews
