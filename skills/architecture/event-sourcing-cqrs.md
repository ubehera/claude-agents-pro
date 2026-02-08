---
name: event-sourcing-cqrs
description: Load when user needs event sourcing, CQRS, or event store patterns for distributed systems
trigger_keywords: [event sourcing, cqrs, event store, command query, event stream, projection, aggregate event, domain event, event replay, read model]
---

# Event Sourcing & CQRS Skill

Event Sourcing and Command Query Responsibility Segregation patterns for systems requiring full audit trails, temporal queries, and separate read/write models.

## Overview

Event Sourcing stores state as a sequence of events rather than current values. CQRS separates read and write models for independent scaling and optimization.

**When to Use**:
- Audit trail requirements (finance, healthcare, compliance)
- Temporal queries ("what was the state at time X?")
- Complex domain with many state transitions
- Read/write workloads with very different patterns

**When NOT to Use**:
- Simple CRUD applications
- Prototype/MVP stage
- Team unfamiliar with eventual consistency
- Low-complexity domains

## Event Sourcing Fundamentals

### Event Store

```typescript
// Events are immutable facts that happened
interface DomainEvent {
  eventId: string;
  aggregateId: string;
  aggregateType: string;
  eventType: string;
  version: number;
  timestamp: Date;
  payload: Record<string, unknown>;
  metadata: { userId: string; correlationId: string };
}

// Example: Order aggregate events
type OrderEvent =
  | { eventType: 'OrderCreated'; payload: { customerId: string; items: LineItem[] } }
  | { eventType: 'ItemAdded'; payload: { item: LineItem } }
  | { eventType: 'ItemRemoved'; payload: { itemId: string } }
  | { eventType: 'OrderSubmitted'; payload: { submittedAt: Date } }
  | { eventType: 'OrderCancelled'; payload: { reason: string } }
  | { eventType: 'PaymentReceived'; payload: { amount: number; method: string } }
  | { eventType: 'OrderShipped'; payload: { trackingNumber: string } };
```

### Aggregate Reconstruction

```typescript
class OrderAggregate {
  private state: OrderState = { status: 'draft', items: [], total: 0 };
  private version = 0;

  // Rebuild state by replaying events
  static fromEvents(events: OrderEvent[]): OrderAggregate {
    const aggregate = new OrderAggregate();
    for (const event of events) {
      aggregate.apply(event);
    }
    return aggregate;
  }

  // Command → validate → emit event
  addItem(item: LineItem): OrderEvent {
    if (this.state.status !== 'draft') {
      throw new Error('Cannot modify submitted order');
    }
    const event: OrderEvent = {
      eventType: 'ItemAdded',
      payload: { item },
    };
    this.apply(event);
    return event;
  }

  // Event handler — pure state transition
  private apply(event: OrderEvent): void {
    this.version++;
    switch (event.eventType) {
      case 'OrderCreated':
        this.state = { status: 'draft', items: event.payload.items, total: 0 };
        break;
      case 'ItemAdded':
        this.state.items.push(event.payload.item);
        this.state.total = this.state.items.reduce((sum, i) => sum + i.price * i.quantity, 0);
        break;
      case 'OrderSubmitted':
        this.state.status = 'submitted';
        break;
      case 'OrderCancelled':
        this.state.status = 'cancelled';
        break;
    }
  }
}
```

## CQRS Pattern

```
Command Side (Write):              Query Side (Read):
  ┌──────────┐                       ┌──────────────┐
  │ Command  │                       │ Read Model   │
  │ Handler  │                       │ (Projection) │
  └────┬─────┘                       └──────┬───────┘
       │                                     │
  ┌────▼─────┐    Events    ┌────────┐  ┌───▼────┐
  │ Event    │──────────────▶│ Event  │──▶│ Query  │
  │ Store    │              │ Bus    │  │ DB     │
  └──────────┘              └────────┘  └────────┘

Write: Command → Aggregate → Events → Event Store
Read:  Events → Projection → Read-optimized DB → Query
```

### Write Side — Command Handler

```typescript
class OrderCommandHandler {
  constructor(
    private eventStore: EventStore,
    private eventBus: EventBus,
  ) {}

  async handle(command: SubmitOrderCommand): Promise<void> {
    // 1. Load aggregate from event stream
    const events = await this.eventStore.getEvents(command.orderId);
    const order = OrderAggregate.fromEvents(events);

    // 2. Execute command (validates business rules)
    const newEvent = order.submit();

    // 3. Persist event (optimistic concurrency via version)
    await this.eventStore.append(command.orderId, newEvent, order.version);

    // 4. Publish for read model updates
    await this.eventBus.publish(newEvent);
  }
}
```

### Read Side — Projection

```typescript
class OrderSummaryProjection {
  constructor(private queryDb: QueryDatabase) {}

  // React to events, update read model
  async handle(event: OrderEvent & { aggregateId: string }): Promise<void> {
    switch (event.eventType) {
      case 'OrderCreated':
        await this.queryDb.insert('order_summaries', {
          orderId: event.aggregateId,
          customerId: event.payload.customerId,
          status: 'draft',
          itemCount: event.payload.items.length,
          total: 0,
        });
        break;

      case 'OrderSubmitted':
        await this.queryDb.update('order_summaries',
          { orderId: event.aggregateId },
          { status: 'submitted', submittedAt: event.payload.submittedAt },
        );
        break;

      case 'OrderShipped':
        await this.queryDb.update('order_summaries',
          { orderId: event.aggregateId },
          { status: 'shipped', trackingNumber: event.payload.trackingNumber },
        );
        break;
    }
  }
}
```

## Event Store Options

| Store | Type | Best For |
|-------|------|----------|
| EventStoreDB | Purpose-built | Production event sourcing |
| PostgreSQL | Relational | Teams already on Postgres |
| DynamoDB | NoSQL | AWS-native, high scale |
| Kafka | Stream | Event streaming + sourcing |

## Best Practices

1. **Events are immutable** — never modify or delete past events
2. **Snapshots for performance** — periodically snapshot aggregate state to avoid replaying all events
3. **Idempotent projections** — handle replayed events gracefully
4. **Versioned events** — schema evolution with upcasting
5. **Eventual consistency** — read models are async; design UI accordingly
6. **Small aggregates** — fewer events per stream = faster reconstruction

---

**Skill Type**: Architecture — Event Sourcing
**Complexity**: Complex
**Typical Usage**: Event-sourced systems, CQRS implementation, audit trail architecture
