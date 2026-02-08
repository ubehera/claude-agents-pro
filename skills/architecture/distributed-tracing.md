---
name: distributed-tracing
description: Implement distributed tracing with Jaeger and Tempo to track requests across microservices and identify performance bottlenecks. Use when debugging microservices, analyzing request flows, or implementing observability.
trigger_keywords: [distributed tracing, jaeger, tempo, opentelemetry, span, trace, observability, zipkin, trace context]
---

# Distributed Tracing

Track requests across distributed systems to understand latency, dependencies, and failure points.

## When to Use

- Debug latency issues
- Understand service dependencies
- Identify bottlenecks
- Trace error propagation
- Analyze request paths

## Core Concepts

- **Trace Context Propagation**: W3C Trace Context headers (`traceparent`, `tracestate`) carry trace ID and span ID across service boundaries - all services must extract/inject these headers for end-to-end visibility
- **Span Hierarchy**: A trace is a directed acyclic graph of spans - parent spans create child spans, enabling visualization of request flow and identification of parallel vs. sequential operations
- **Baggage Items**: Key-value pairs propagated alongside trace context for cross-cutting concerns (user ID, tenant ID) - use sparingly as they add overhead to every request
- **Sampling Strategies**: Head-based sampling decides at trace start (simple, predictable), tail-based sampling decides after trace completes (captures errors/slow traces) - production typically uses 1-10% head sampling with 100% error sampling
- **Instrumentation Layers**: Auto-instrumentation captures HTTP/DB/messaging spans automatically; manual instrumentation adds business-specific spans for domain operations

## Key Concepts

- **Trace**: End-to-end request journey
- **Span**: Single operation within a trace
- **Context**: Metadata propagated between services
- **Tags**: Key-value pairs for filtering

## Quick Start (OpenTelemetry Python)

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(
    JaegerExporter(agent_host_name="jaeger", agent_port=6831)
))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("my_operation") as span:
    span.set_attribute("user.id", "123")
    # ... business logic
```

## Context Propagation

```python
from opentelemetry.propagate import inject
headers = {}
inject(headers)  # Injects trace context
response = requests.get('http://downstream/api', headers=headers)
```

## Sampling Strategies

- **Probabilistic**: Sample 1% of traces (`param: 0.01`)
- **Rate Limiting**: Max 100 traces/second (`param: 100`)
- **Adaptive**: Based on trace ID (deterministic)

## Best Practices

1. Sample appropriately (1-10% in production)
2. Add meaningful tags (user_id, request_id)
3. Propagate context across all service boundaries
4. Log exceptions in spans
5. Use consistent naming for operations
6. Monitor tracing overhead (<1% CPU)
