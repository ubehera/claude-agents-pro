---
name: microservices-patterns
description: Design microservices architectures with service boundaries, event-driven communication, and resilience patterns. Use when building distributed systems, decomposing monoliths, or implementing microservices.
trigger_keywords: [microservices, saga pattern, circuit breaker, service mesh, distributed system, event driven, bounded context, strangler fig]
---

# Microservices Patterns

## Core Concepts

- **Bounded Context**: Each microservice owns a single domain with clear boundaries - services communicate through well-defined APIs, never share databases, and maintain data autonomy
- **Eventual Consistency**: Accept that data across services will be temporarily inconsistent - design for idempotency, use compensation patterns, and communicate state changes via events
- **Bulkhead Pattern**: Isolate failures by partitioning resources (thread pools, connection pools, instances) - prevents cascading failures where one service exhausts shared resources
- **Service Discovery**: Services register themselves and discover others dynamically - use DNS-based (Kubernetes Services), client-side (Eureka), or service mesh (Istio) discovery patterns
- **Contract-First Design**: Define API contracts (OpenAPI, gRPC protobuf, AsyncAPI) before implementation - enables parallel development, contract testing, and prevents breaking changes

## Core Patterns

### Service Decomposition
- **By Business Capability**: OrderService, PaymentService, InventoryService
- **By Subdomain (DDD)**: Bounded contexts map to services
- **Strangler Fig**: Gradually extract from monolith

### Communication
- **Synchronous**: REST, gRPC, GraphQL
- **Asynchronous**: Kafka, RabbitMQ, SQS

### Data Management
- **Database Per Service**: Each service owns its data
- **Saga Pattern**: Distributed transactions with compensating actions

## Saga Pattern (Orchestrated)

```python
class OrderFulfillmentSaga:
    steps = [
        SagaStep("create_order", action=create_order, compensation=cancel_order),
        SagaStep("reserve_inventory", action=reserve, compensation=release),
        SagaStep("process_payment", action=charge, compensation=refund),
    ]

    async def execute(self, order_data):
        completed = []
        for step in self.steps:
            result = await step.action(context)
            if not result.success:
                await self.compensate(completed)
                return SagaResult(status=FAILED)
            completed.append(step)
        return SagaResult(status=COMPLETED)
```

## Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.state = CLOSED  # CLOSED → OPEN → HALF_OPEN

    async def call(self, func, *args):
        if self.state == OPEN:
            if self._should_attempt_reset():
                self.state = HALF_OPEN
            else:
                raise CircuitBreakerOpenError()
        try:
            result = await func(*args)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise
```

## Best Practices

1. **Service Boundaries**: Align with business capabilities
2. **Database Per Service**: No shared databases
3. **API Contracts**: Versioned, backward compatible
4. **Async When Possible**: Events over direct calls
5. **Circuit Breakers**: Fail fast on service failures
6. **Distributed Tracing**: Track requests across services

## Common Pitfalls

- Distributed Monolith (tightly coupled services)
- Chatty Services (too many inter-service calls)
- Shared Databases
- No Circuit Breakers → cascade failures
- Synchronous Everything
