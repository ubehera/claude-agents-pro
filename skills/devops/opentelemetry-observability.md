---
name: opentelemetry-observability
description: Comprehensive observability with OpenTelemetry for distributed tracing, metrics, and logging. Includes instrumentation, exporters, collectors, and integration with Jaeger, Prometheus, and Grafana. Use when implementing modern observability for microservices.
trigger_keywords: [opentelemetry, otel, distributed tracing, jaeger, tempo, traces, spans, metrics, observability, telemetry, instrumentation, collector, prometheus, grafana]
---

# OpenTelemetry & Distributed Observability

Production-grade observability with OpenTelemetry for distributed tracing, metrics, and logging across microservices.

## Core Concepts

### Three Pillars of Observability

**1. Traces** - Request flow across services
**2. Metrics** - System and application measurements
**3. Logs** - Discrete event records

**OpenTelemetry** provides:
- Vendor-neutral instrumentation
- Automatic and manual instrumentation
- Context propagation across services
- Exporters for popular backends (Jaeger, Prometheus, etc.)

### Architecture

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  Service A   │──────▶│  Service B   │──────▶│  Service C   │
└──────┬───────┘       └──────┬───────┘       └──────┬───────┘
       │                      │                      │
       │ (traces/metrics)     │                      │
       ▼                      ▼                      ▼
┌────────────────────────────────────────────────────────────┐
│              OpenTelemetry Collector                       │
│  ┌──────────┐  ┌───────────┐  ┌─────────────┐            │
│  │Receivers │─▶│Processors │─▶│  Exporters  │            │
│  └──────────┘  └───────────┘  └─────────────┘            │
└────────┬──────────────┬──────────────┬─────────────────────┘
         │              │              │
         ▼              ▼              ▼
    ┌────────┐    ┌──────────┐   ┌────────┐
    │ Jaeger │    │Prometheus│   │  Loki  │
    └────────┘    └──────────┘   └────────┘
         │              │              │
         └──────────────┴──────────────┘
                       │
                       ▼
                 ┌──────────┐
                 │ Grafana  │
                 └──────────┘
```

## Application Instrumentation

### Node.js / TypeScript

```typescript
// instrumentation.ts
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';

const sdk = new NodeSDK({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'user-service',
    [SemanticResourceAttributes.SERVICE_VERSION]: '1.0.0',
    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: process.env.NODE_ENV,
  }),

  // Trace exporter
  traceExporter: new OTLPTraceExporter({
    url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318/v1/traces',
  }),

  // Metric exporter
  metricReader: new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter({
      url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318/v1/metrics',
    }),
    exportIntervalMillis: 60000,
  }),

  // Auto-instrumentation for common libraries
  instrumentations: [
    getNodeAutoInstrumentations({
      '@opentelemetry/instrumentation-http': {
        enabled: true,
      },
      '@opentelemetry/instrumentation-express': {
        enabled: true,
      },
      '@opentelemetry/instrumentation-pg': {
        enabled: true,
      },
      '@opentelemetry/instrumentation-redis': {
        enabled: true,
      },
    }),
  ],
});

sdk.start();

// Graceful shutdown
process.on('SIGTERM', () => {
  sdk.shutdown()
    .then(() => console.log('Tracing terminated'))
    .catch((error) => console.error('Error terminating tracing', error))
    .finally(() => process.exit(0));
});
```

**Manual instrumentation:**

```typescript
// app.ts
import { trace, context, SpanStatusCode } from '@opentelemetry/api';
import { SemanticAttributes } from '@opentelemetry/semantic-conventions';

const tracer = trace.getTracer('user-service', '1.0.0');

// Custom span
export async function getUserById(userId: string) {
  return tracer.startActiveSpan('getUserById', async (span) => {
    try {
      span.setAttributes({
        [SemanticAttributes.DB_SYSTEM]: 'postgresql',
        'user.id': userId,
      });

      const user = await db.query('SELECT * FROM users WHERE id = $1', [userId]);

      span.addEvent('user_fetched', {
        'user.email': user.email,
      });

      return user;
    } catch (error) {
      span.recordException(error);
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error.message,
      });
      throw error;
    } finally {
      span.end();
    }
  });
}

// Express middleware
app.use((req, res, next) => {
  const span = trace.getActiveSpan();
  if (span) {
    span.setAttributes({
      'http.method': req.method,
      'http.url': req.url,
      'http.user_agent': req.headers['user-agent'],
    });
  }
  next();
});
```

### Python (FastAPI)

```python
# instrumentation.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

# Configure resource
resource = Resource(attributes={
    ResourceAttributes.SERVICE_NAME: "payment-service",
    ResourceAttributes.SERVICE_VERSION: "1.0.0",
    ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("ENV", "development"),
})

# Configure tracer provider
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Auto-instrument FastAPI
FastAPIInstrumentor.instrument_app(app)
SQLAlchemyInstrumentor().instrument(engine=engine)
RedisInstrumentor().instrument()

# Manual instrumentation
tracer = trace.get_tracer(__name__)

@app.post("/payment")
async def process_payment(payment: Payment):
    with tracer.start_as_current_span("process_payment") as span:
        span.set_attribute("payment.amount", payment.amount)
        span.set_attribute("payment.currency", payment.currency)

        try:
            # Validate payment
            with tracer.start_as_current_span("validate_payment"):
                await validate_payment(payment)

            # Process with payment gateway
            with tracer.start_as_current_span("charge_customer") as charge_span:
                charge_span.set_attribute("payment.gateway", "stripe")
                result = await stripe.charge(payment)
                charge_span.add_event("payment_charged", {
                    "transaction_id": result.id
                })

            return {"status": "success", "transaction_id": result.id}

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
```

### Go

```go
// instrumentation.go
package main

import (
    "context"
    "log"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
    "go.opentelemetry.io/otel/sdk/resource"
    sdktrace "go.opentelemetry.io/otel/sdk/trace"
    semconv "go.opentelemetry.io/otel/semconv/v1.20.0"
    "go.opentelemetry.io/otel/trace"

    "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
    "go.opentelemetry.io/contrib/instrumentation/github.com/gin-gonic/gin/otelgin"
)

func initTracer() (*sdktrace.TracerProvider, error) {
    exporter, err := otlptracehttp.New(context.Background(),
        otlptracehttp.WithEndpoint("localhost:4318"),
        otlptracehttp.WithInsecure(),
    )
    if err != nil {
        return nil, err
    }

    tp := sdktrace.NewTracerProvider(
        sdktrace.WithBatcher(exporter),
        sdktrace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceName("order-service"),
            semconv.ServiceVersion("1.0.0"),
            attribute.String("environment", "production"),
        )),
    )

    otel.SetTracerProvider(tp)
    return tp, nil
}

// Gin middleware
func main() {
    tp, _ := initTracer()
    defer tp.Shutdown(context.Background())

    router := gin.Default()
    router.Use(otelgin.Middleware("order-service"))

    router.GET("/orders/:id", getOrder)
    router.Run(":8080")
}

// Manual instrumentation
func getOrder(c *gin.Context) {
    tracer := otel.Tracer("order-service")
    ctx, span := tracer.Start(c.Request.Context(), "getOrder",
        trace.WithAttributes(
            attribute.String("order.id", c.Param("id")),
        ),
    )
    defer span.End()

    // Database query with tracing
    order, err := fetchOrderFromDB(ctx, c.Param("id"))
    if err != nil {
        span.RecordError(err)
        span.SetStatus(codes.Error, err.Error())
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }

    span.AddEvent("order_fetched", trace.WithAttributes(
        attribute.Float64("order.total", order.Total),
    ))

    c.JSON(200, order)
}
```

## OpenTelemetry Collector

**Collector configuration:**

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

  # Prometheus receiver for scraping
  prometheus:
    config:
      scrape_configs:
        - job_name: 'otel-collector'
          scrape_interval: 10s
          static_configs:
            - targets: ['localhost:8888']

processors:
  # Batch spans for efficiency
  batch:
    timeout: 10s
    send_batch_size: 1024

  # Add resource attributes
  resource:
    attributes:
      - key: cluster.name
        value: production-cluster
        action: upsert

  # Memory limiter to prevent OOM
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128

  # Sampling (optional)
  probabilistic_sampler:
    sampling_percentage: 100  # Sample 100% in production
    hash_seed: 22

exporters:
  # Jaeger for traces
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  # Tempo for traces (alternative)
  otlp/tempo:
    endpoint: tempo:4317
    tls:
      insecure: true

  # Prometheus for metrics
  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: otel
    const_labels:
      environment: production

  # Loki for logs
  loki:
    endpoint: http://loki:3100/loki/api/v1/push

  # Logging exporter for debugging
  logging:
    loglevel: info

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, resource]
      exporters: [jaeger, otlp/tempo, logging]

    metrics:
      receivers: [otlp, prometheus]
      processors: [memory_limiter, batch, resource]
      exporters: [prometheus, logging]

    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch, resource]
      exporters: [loki, logging]

  extensions: [health_check, pprof, zpages]

extensions:
  health_check:
    endpoint: :13133
  pprof:
    endpoint: :1777
  zpages:
    endpoint: :55679
```

**Deploy collector:**

```yaml
# kubernetes/otel-collector.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: observability
data:
  otel-collector-config.yaml: |
    # Configuration above

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otel-collector
  namespace: observability
spec:
  replicas: 2
  selector:
    matchLabels:
      app: otel-collector
  template:
    metadata:
      labels:
        app: otel-collector
    spec:
      containers:
      - name: otel-collector
        image: otel/opentelemetry-collector-contrib:0.89.0
        args:
        - --config=/conf/otel-collector-config.yaml
        ports:
        - containerPort: 4317  # OTLP gRPC
        - containerPort: 4318  # OTLP HTTP
        - containerPort: 8889  # Prometheus metrics
        - containerPort: 13133 # Health check
        volumeMounts:
        - name: config
          mountPath: /conf
        resources:
          limits:
            memory: 1Gi
            cpu: 1000m
          requests:
            memory: 512Mi
            cpu: 500m
      volumes:
      - name: config
        configMap:
          name: otel-collector-config
---
apiVersion: v1
kind: Service
metadata:
  name: otel-collector
  namespace: observability
spec:
  selector:
    app: otel-collector
  ports:
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  - name: otlp-http
    port: 4318
    targetPort: 4318
  - name: metrics
    port: 8889
    targetPort: 8889
```

## Context Propagation

**Cross-service tracing:**

```typescript
// Service A (initiator)
import { propagation, context } from '@opentelemetry/api';

async function callServiceB(data: any) {
  const span = trace.getActiveSpan();

  // Inject trace context into HTTP headers
  const headers = {};
  propagation.inject(context.active(), headers);

  const response = await fetch('http://service-b/api/process', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers,  // W3C Trace Context headers
    },
    body: JSON.stringify(data),
  });

  return response.json();
}

// Service B (receiver)
app.use((req, res, next) => {
  // Extract trace context from headers
  const extractedContext = propagation.extract(context.active(), req.headers);

  context.with(extractedContext, () => {
    const span = trace.getActiveSpan();
    span.setAttributes({
      'http.method': req.method,
      'http.route': req.route?.path,
    });
    next();
  });
});
```

## Metrics with OpenTelemetry

```typescript
import { MeterProvider, PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';

const metricReader = new PeriodicExportingMetricReader({
  exporter: new OTLPMetricExporter({
    url: 'http://localhost:4318/v1/metrics',
  }),
  exportIntervalMillis: 60000,
});

const meterProvider = new MeterProvider({
  readers: [metricReader],
});

const meter = meterProvider.getMeter('user-service');

// Counter
const requestCounter = meter.createCounter('http_requests_total', {
  description: 'Total HTTP requests',
});

requestCounter.add(1, { method: 'GET', route: '/users', status: '200' });

// Histogram
const requestDuration = meter.createHistogram('http_request_duration_ms', {
  description: 'HTTP request duration',
  unit: 'ms',
});

const start = Date.now();
// ... handle request
requestDuration.record(Date.now() - start, { method: 'GET', route: '/users' });

// Gauge (async)
const activeConnections = meter.createObservableGauge('active_connections', {
  description: 'Number of active connections',
});

activeConnections.addCallback((result) => {
  result.observe(getActiveConnectionCount());
});
```

## Grafana Dashboards

**Trace visualization query:**

```json
{
  "datasource": "Tempo",
  "queryType": "traceql",
  "query": "{ span.http.status_code >= 500 } | rate() > 0.01"
}
```

**Service map:**

```json
{
  "datasource": "Tempo",
  "queryType": "serviceMap",
  "query": "{ service.name=\"user-service\" }"
}
```

## Best Practices

1. **Sampling Strategy**
   - Production: 100% for errors, 10% for success
   - Development: 100%
   - Use tail-based sampling for important traces

2. **Attribute Guidelines**
   - Use semantic conventions
   - Avoid high-cardinality attributes
   - Don't include PII

3. **Performance**
   - Use batch processors
   - Configure memory limits
   - Monitor collector resource usage

4. **Instrumentation**
   - Start with auto-instrumentation
   - Add custom spans for business logic
   - Include meaningful events and attributes

## Quality Standards

- **Coverage**: All services instrumented with traces
- **Performance**: <5% overhead from instrumentation
- **Reliability**: Collector HA with 99.9% uptime
- **Usability**: Trace search <2s, service map updated real-time

## Related Skills

- `prometheus-configuration` - For metrics
- `kubernetes-advanced-patterns` - For deployment
- `ci-cd-patterns` - For automation

---

**Skill Type**: DevOps - Observability
**Complexity**: Advanced
**Typical Usage**: Microservices observability, distributed tracing, performance monitoring
**Prerequisites**: Basic observability concepts, microservices architecture
