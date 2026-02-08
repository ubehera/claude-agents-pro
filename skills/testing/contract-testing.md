---
name: contract-testing
description: Load when user needs consumer-driven contract testing patterns with Pact or similar frameworks
trigger_keywords: [contract testing, pact, consumer driven contract, provider verification, contract test, api contract, schema contract, cdc]
---

# Contract Testing Skill

Consumer-driven contract testing patterns for verifying API compatibility between services without end-to-end tests.

## Overview

Contract testing verifies that two services (consumer and provider) can communicate correctly by testing each side independently against a shared contract. This catches integration bugs early without requiring both services to be running.

**When to Use**:
- Microservices communicating via HTTP/gRPC/messaging
- Multiple teams owning different services
- API changes that could break downstream consumers
- Replacing flaky E2E integration tests

## Core Concepts

### Contract Testing vs Integration Testing

```
Integration Test:
  Consumer → [network] → Provider → [database]
  Pros: Tests real behavior
  Cons: Slow, flaky, requires full environment

Contract Test:
  Consumer → [mock provider from contract] → Verify expectations
  Provider → [verify contract against real implementation]
  Pros: Fast, isolated, catches breaking changes early
  Cons: Doesn't test actual network behavior
```

### Pact — Consumer-Driven Contracts

#### Consumer Side (JavaScript/TypeScript)

```typescript
import { PactV3, MatchersV3 } from '@pact-foundation/pact';

const provider = new PactV3({
  consumer: 'OrderService',
  provider: 'UserService',
});

describe('UserService Contract', () => {
  it('returns user by ID', async () => {
    // Define the contract (what consumer expects)
    await provider
      .given('user 123 exists')
      .uponReceiving('a request for user 123')
      .withRequest({
        method: 'GET',
        path: '/api/users/123',
        headers: { Accept: 'application/json' },
      })
      .willRespondWith({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: MatchersV3.like({
          id: '123',
          name: MatchersV3.string('John Doe'),
          email: MatchersV3.email('john@example.com'),
          plan: MatchersV3.regex('free|pro|enterprise', 'pro'),
        }),
      })
      .executeTest(async (mockServer) => {
        // Test consumer code against the mock
        const client = new UserClient(mockServer.url);
        const user = await client.getUser('123');

        expect(user.id).toBe('123');
        expect(user.name).toBeDefined();
        expect(user.email).toContain('@');
      });
  });

  it('handles user not found', async () => {
    await provider
      .given('user 999 does not exist')
      .uponReceiving('a request for non-existent user')
      .withRequest({ method: 'GET', path: '/api/users/999' })
      .willRespondWith({
        status: 404,
        body: MatchersV3.like({ error: 'User not found' }),
      })
      .executeTest(async (mockServer) => {
        const client = new UserClient(mockServer.url);
        await expect(client.getUser('999')).rejects.toThrow('not found');
      });
  });
});
```

#### Provider Side (Verification)

```typescript
import { Verifier } from '@pact-foundation/pact';

describe('UserService Provider Verification', () => {
  it('validates contract with OrderService', async () => {
    const verifier = new Verifier({
      providerBaseUrl: 'http://localhost:3001',
      pactUrls: ['./pacts/OrderService-UserService.json'],
      // OR from Pact Broker:
      // pactBrokerUrl: 'https://pact-broker.example.com',
      // provider: 'UserService',

      stateHandlers: {
        'user 123 exists': async () => {
          await db.users.create({ id: '123', name: 'John Doe', email: 'john@example.com', plan: 'pro' });
        },
        'user 999 does not exist': async () => {
          await db.users.deleteMany({ where: { id: '999' } });
        },
      },
    });

    await verifier.verifyProvider();
  });
});
```

### Python — Pact Consumer

```python
import pytest
from pact import Consumer, Provider

@pytest.fixture
def pact():
    pact = Consumer('OrderService').has_pact_with(
        Provider('UserService'),
        pact_dir='./pacts',
    )
    pact.start_service()
    yield pact
    pact.stop_service()

def test_get_user(pact):
    expected = {'id': '123', 'name': 'John', 'email': 'john@ex.com'}

    (pact
     .given('user 123 exists')
     .upon_receiving('a request for user 123')
     .with_request('GET', '/api/users/123')
     .will_respond_with(200, body=Like(expected)))

    with pact:
        result = UserClient(pact.uri).get_user('123')
        assert result['id'] == '123'
```

## Pact Broker Workflow

```
1. Consumer writes tests → generates pact file
2. Pact file published to Pact Broker
3. Provider CI pulls pacts → verifies against real implementation
4. Results published back to broker
5. can-i-deploy check before releasing either service

CI Pipeline:
  Consumer CI:
    - Run consumer pact tests
    - Publish pact to broker: pact-broker publish ./pacts --consumer-app-version=$GIT_SHA
    - can-i-deploy: pact-broker can-i-deploy --pacticipant=OrderService --version=$GIT_SHA

  Provider CI:
    - Verify pacts from broker
    - Publish verification: results auto-published
    - can-i-deploy: pact-broker can-i-deploy --pacticipant=UserService --version=$GIT_SHA
```

## Best Practices

1. **Consumer drives the contract** — providers verify, not define
2. **Test behavior, not implementation** — use matchers (like, regex) over exact values
3. **Provider states** — set up test data for each scenario
4. **Pact Broker** for team coordination — central contract repository
5. **can-i-deploy** gate — prevent releasing incompatible versions
6. **Keep contracts minimal** — only test fields the consumer actually uses

---

**Skill Type**: Testing — Contract Testing
**Complexity**: Moderate
**Typical Usage**: Microservice API compatibility, consumer-driven contracts
