---
name: load-testing
description: Load when user needs performance and load testing patterns with k6, Artillery, or Locust
trigger_keywords: [load testing, stress testing, performance testing, k6, artillery, locust, benchmark, throughput, rps, concurrent users, spike test, soak test]
---

# Load Testing Skill

Performance and load testing patterns using k6, Artillery, and Locust for validating system capacity and identifying bottlenecks.

## Overview

Load testing verifies that systems handle expected (and unexpected) traffic levels. Different test types reveal different issues.

**When to Use**:
- Before production launch (capacity planning)
- After architecture changes (regression detection)
- Capacity planning for traffic spikes
- Identifying bottlenecks under load

## Test Types

| Type | Purpose | Pattern |
|------|---------|---------|
| **Smoke** | Verify system works under minimal load | 1-5 VUs, 1 min |
| **Load** | Validate expected traffic levels | Target VUs, 10-30 min |
| **Stress** | Find breaking point | Ramp beyond capacity |
| **Spike** | Test sudden traffic bursts | Rapid ramp to peak |
| **Soak** | Find memory leaks over time | Moderate load, 1-4 hours |

## k6 (Grafana) — JavaScript

```javascript
// load-test.js — comprehensive load test
import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const loginDuration = new Trend('login_duration');

export const options = {
  scenarios: {
    // Smoke test
    smoke: {
      executor: 'constant-vus',
      vus: 3,
      duration: '1m',
      tags: { test_type: 'smoke' },
    },
    // Load test — ramp up to target
    load: {
      executor: 'ramping-vus',
      startTime: '1m',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up
        { duration: '5m', target: 50 },   // Sustain
        { duration: '2m', target: 0 },    // Ramp down
      ],
      tags: { test_type: 'load' },
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],  // 95% under 500ms
    http_req_failed: ['rate<0.01'],                    // <1% errors
    errors: ['rate<0.05'],                             // Custom error rate
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export default function () {
  group('API Flow', () => {
    // Login
    const loginStart = Date.now();
    const loginRes = http.post(`${BASE_URL}/api/auth/login`, JSON.stringify({
      email: `user${__VU}@example.com`,
      password: 'testpass123',
    }), { headers: { 'Content-Type': 'application/json' } });

    loginDuration.add(Date.now() - loginStart);
    check(loginRes, { 'login successful': (r) => r.status === 200 });
    errorRate.add(loginRes.status !== 200);

    if (loginRes.status !== 200) return;
    const token = loginRes.json('token');

    // List items
    const listRes = http.get(`${BASE_URL}/api/items`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    check(listRes, {
      'list status 200': (r) => r.status === 200,
      'has items': (r) => r.json('items').length > 0,
    });

    // Create item
    const createRes = http.post(`${BASE_URL}/api/items`, JSON.stringify({
      name: `item-${Date.now()}`,
    }), {
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    check(createRes, { 'create status 201': (r) => r.status === 201 });

    sleep(1); // Think time between requests
  });
}
```

Run: `k6 run --out json=results.json load-test.js`

## Artillery — YAML-Based

```yaml
# artillery.yml
config:
  target: "http://localhost:3000"
  phases:
    - duration: 60       # Warm up
      arrivalRate: 5
      name: "Warm up"
    - duration: 300      # Sustained load
      arrivalRate: 50
      name: "Sustained load"
    - duration: 60       # Spike
      arrivalRate: 200
      name: "Spike"
  defaults:
    headers:
      Content-Type: "application/json"
  ensure:
    thresholds:
      - http.response_time.p95: 500
      - http.response_time.p99: 1000
  plugins:
    metrics-by-endpoint:
      useOnlyRequestNames: true

scenarios:
  - name: "Browse and Purchase"
    weight: 70        # 70% of traffic
    flow:
      - get:
          url: "/api/products"
          capture:
            - json: "$.products[0].id"
              as: "productId"
      - think: 2
      - get:
          url: "/api/products/{{ productId }}"
      - think: 1
      - post:
          url: "/api/cart"
          json:
            productId: "{{ productId }}"
            quantity: 1

  - name: "Search"
    weight: 30        # 30% of traffic
    flow:
      - get:
          url: "/api/search?q=laptop"
      - think: 3
      - get:
          url: "/api/search?q=laptop&sort=price&page=2"
```

Run: `artillery run artillery.yml --output report.json && artillery report report.json`

## Locust — Python

```python
# locustfile.py
from locust import HttpUser, task, between, tag

class WebUser(HttpUser):
    wait_time = between(1, 3)  # Think time

    def on_start(self):
        """Login on start."""
        response = self.client.post("/api/auth/login", json={
            "email": f"user{self.environment.runner.user_count}@test.com",
            "password": "testpass123",
        })
        self.token = response.json().get("token", "")

    @task(3)  # Weight: 3x more likely than other tasks
    @tag("read")
    def list_items(self):
        self.client.get("/api/items", headers={
            "Authorization": f"Bearer {self.token}",
        })

    @task(1)
    @tag("write")
    def create_item(self):
        self.client.post("/api/items", json={
            "name": f"item-{self.environment.runner.user_count}",
        }, headers={
            "Authorization": f"Bearer {self.token}",
        })
```

Run: `locust -f locustfile.py --host=http://localhost:3000 --users=100 --spawn-rate=10`

## Key Metrics

| Metric | Target | Red Flag |
|--------|--------|----------|
| P95 Response Time | <500ms | >1000ms |
| P99 Response Time | <1000ms | >3000ms |
| Error Rate | <1% | >5% |
| Throughput (RPS) | Meets SLO | Declining under load |
| CPU Utilization | <70% | >90% sustained |
| Memory | Stable | Growing (leak) |

## Best Practices

1. **Test in production-like environment** — same infra, same data volume
2. **Realistic user flows** — not just single endpoints
3. **Think times** — simulate real user pauses between actions
4. **Gradual ramps** — don't start at peak; observe system behavior as load increases
5. **Custom metrics** — track business-specific metrics (checkout time, search latency)
6. **Automate in CI** — run smoke/load tests on every deploy

---

**Skill Type**: Testing — Load Testing
**Complexity**: Moderate
**Typical Usage**: Capacity planning, performance regression detection, bottleneck identification
