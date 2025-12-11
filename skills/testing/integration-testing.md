---
name: integration-testing
description: Load when implementing integration tests for APIs, databases, services, or cross-component interactions requiring real dependencies
trigger_keywords: [integration test, api testing, database testing, service integration, contract testing, test containers, mock server, supertest]
---

# Integration Testing Patterns

Production-grade integration testing for APIs, databases, and service boundaries with real dependencies and comprehensive coverage.

## Overview

Integration tests validate interactions between components, services, and external systems. Unlike unit tests (isolated) and E2E tests (full system), integration tests focus on component boundaries and contracts.

**When to Use**:
- Testing API endpoints with real database
- Validating service-to-service communication
- Testing database queries and transactions
- Verifying external API integrations
- Contract testing between services

**Test Scope**:
- API request/response cycles
- Database operations (CRUD, transactions)
- Message queue producers/consumers
- Authentication and authorization flows
- Data transformation pipelines

## The Testing Pyramid Position

```
        /\
       /E2E\         ← Full system
      /─────\
     /Integr\        ← 20%: Component boundaries ← YOU ARE HERE
    /────────\
   /Unit Tests\      ← Isolated logic
  /────────────\
```

## API Integration Testing

### FastAPI Testing (Python)

```python
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import pytest

from app.main import app
from app.database import Base, get_db

# Test database setup
@pytest.fixture(scope="function")
def db_session():
    """Create isolated test database for each test"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)

    yield session

    session.rollback()
    session.close()

@pytest.fixture
def client(db_session):
    """Test client with overridden database dependency"""
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

# Test API endpoints
def test_create_user(client, db_session):
    """Test user creation endpoint with database"""
    response = client.post(
        "/users",
        json={"email": "test@example.com", "name": "Test User"}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data

    # Verify database state
    user = db_session.query(User).filter_by(email="test@example.com").first()
    assert user is not None
    assert user.name == "Test User"

def test_get_user_not_found(client):
    """Test 404 handling"""
    response = client.get("/users/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

def test_duplicate_email_rejected(client):
    """Test unique constraint validation"""
    user_data = {"email": "duplicate@example.com", "name": "User One"}

    # First creation succeeds
    response1 = client.post("/users", json=user_data)
    assert response1.status_code == 201

    # Second creation fails
    response2 = client.post("/users", json=user_data)
    assert response2.status_code == 400
    assert "already exists" in response2.json()["detail"].lower()
```

### Express.js Testing (Node.js)

```typescript
import request from 'supertest';
import { app } from '../src/app';
import { setupTestDatabase, teardownTestDatabase } from './helpers/db';

describe('User API Integration Tests', () => {
    beforeAll(async () => {
        await setupTestDatabase();
    });

    afterAll(async () => {
        await teardownTestDatabase();
    });

    beforeEach(async () => {
        // Clear data between tests
        await db.query('DELETE FROM users');
    });

    describe('POST /api/users', () => {
        it('should create user with valid data', async () => {
            const response = await request(app)
                .post('/api/users')
                .send({
                    email: 'test@example.com',
                    name: 'Test User',
                })
                .expect(201);

            expect(response.body).toMatchObject({
                email: 'test@example.com',
                name: 'Test User',
                id: expect.any(Number),
            });

            // Verify in database
            const user = await db.query(
                'SELECT * FROM users WHERE email = $1',
                ['test@example.com']
            );
            expect(user.rows).toHaveLength(1);
        });

        it('should validate required fields', async () => {
            const response = await request(app)
                .post('/api/users')
                .send({ name: 'Test User' })  // Missing email
                .expect(400);

            expect(response.body.errors).toContainEqual(
                expect.objectContaining({
                    field: 'email',
                    message: 'Email is required',
                })
            );
        });
    });

    describe('GET /api/users/:id', () => {
        it('should return user by id', async () => {
            // Setup: Create test user
            const user = await createTestUser({
                email: 'test@example.com',
                name: 'Test User',
            });

            const response = await request(app)
                .get(`/api/users/${user.id}`)
                .expect(200);

            expect(response.body).toMatchObject({
                id: user.id,
                email: 'test@example.com',
                name: 'Test User',
            });
        });
    });
});
```

## Database Integration Testing

### Transaction Testing

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

@pytest.fixture(scope="function")
def db_session():
    """Test database with automatic rollback"""
    engine = create_engine("postgresql://test:test@localhost/testdb")
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    # Rollback all changes after test
    session.close()
    transaction.rollback()
    connection.close()

def test_user_creation_transaction(db_session):
    """Test that user creation is transactional"""
    user = User(email="test@example.com", name="Test User")
    db_session.add(user)
    db_session.commit()

    # Verify user exists
    found = db_session.query(User).filter_by(email="test@example.com").first()
    assert found is not None
    assert found.name == "Test User"

def test_rollback_on_constraint_violation(db_session):
    """Test transaction rollback on error"""
    # Create first user
    user1 = User(email="test@example.com", name="User One")
    db_session.add(user1)
    db_session.commit()

    # Attempt to create duplicate (should fail)
    user2 = User(email="test@example.com", name="User Two")
    db_session.add(user2)

    with pytest.raises(IntegrityError):
        db_session.commit()

    # Session should be rolled back
    db_session.rollback()

    # First user should still exist
    users = db_session.query(User).all()
    assert len(users) == 1
    assert users[0].name == "User One"
```

### Test Containers (Docker)

```python
import pytest
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def postgres_container():
    """Start PostgreSQL container for tests"""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres

@pytest.fixture
def db_engine(postgres_container):
    """Create database engine connected to test container"""
    connection_url = postgres_container.get_connection_url()
    engine = create_engine(connection_url)

    # Create tables
    Base.metadata.create_all(engine)

    yield engine

    # Drop tables after tests
    Base.metadata.drop_all(engine)
    engine.dispose()

def test_with_real_postgres(db_engine):
    """Test against real PostgreSQL instance"""
    session = Session(db_engine)

    user = User(email="test@example.com", name="Test User")
    session.add(user)
    session.commit()

    # Test PostgreSQL-specific features
    result = session.execute(
        text("SELECT * FROM users WHERE email ILIKE :email"),
        {"email": "%TEST%"}
    )
    assert result.rowcount == 1
```

## Service Integration Testing

### Mocking External APIs

```typescript
// Using MSW (Mock Service Worker)
import { rest } from 'msw';
import { setupServer } from 'msw/node';

const server = setupServer(
    rest.get('https://api.stripe.com/v1/customers/:id', (req, res, ctx) => {
        return res(
            ctx.json({
                id: req.params.id,
                email: 'customer@example.com',
                name: 'Test Customer',
            })
        );
    }),

    rest.post('https://api.stripe.com/v1/charges', (req, res, ctx) => {
        return res(
            ctx.json({
                id: 'ch_test123',
                status: 'succeeded',
                amount: 1000,
            })
        );
    })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

test('payment processing with mocked Stripe', async () => {
    const result = await processPayment({
        customerId: 'cus_test',
        amount: 1000,
    });

    expect(result.status).toBe('succeeded');
    expect(result.chargeId).toBe('ch_test123');
});

test('handles Stripe API errors', async () => {
    server.use(
        rest.post('https://api.stripe.com/v1/charges', (req, res, ctx) => {
            return res(
                ctx.status(400),
                ctx.json({
                    error: {
                        type: 'card_error',
                        message: 'Your card was declined',
                    },
                })
            );
        })
    );

    await expect(
        processPayment({ customerId: 'cus_test', amount: 1000 })
    ).rejects.toThrow('Your card was declined');
});
```

### Contract Testing (Pact)

```typescript
import { Pact } from '@pact-foundation/pact';
import { UserService } from '../src/services/user';

const provider = new Pact({
    consumer: 'WebApp',
    provider: 'UserAPI',
    port: 1234,
});

describe('User API Contract', () => {
    beforeAll(() => provider.setup());
    afterAll(() => provider.finalize());
    afterEach(() => provider.verify());

    it('should get user by id', async () => {
        await provider.addInteraction({
            state: 'user 123 exists',
            uponReceiving: 'a request for user 123',
            withRequest: {
                method: 'GET',
                path: '/api/users/123',
            },
            willRespondWith: {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
                body: {
                    id: 123,
                    email: 'user@example.com',
                    name: 'Test User',
                },
            },
        });

        const userService = new UserService('http://localhost:1234');
        const user = await userService.getUser(123);

        expect(user).toMatchObject({
            id: 123,
            email: 'user@example.com',
        });
    });
});
```

## Message Queue Testing

### Testing Kafka Consumers/Producers

```python
import pytest
from kafka import KafkaProducer, KafkaConsumer
from testcontainers.kafka import KafkaContainer

@pytest.fixture(scope="module")
def kafka_container():
    """Start Kafka container for tests"""
    with KafkaContainer() as kafka:
        yield kafka

def test_event_processing(kafka_container):
    """Test event producer and consumer"""
    bootstrap_servers = kafka_container.get_bootstrap_server()

    # Producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Consumer
    consumer = KafkaConsumer(
        'test-topic',
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='earliest',
    )

    # Send event
    event = {'type': 'USER_CREATED', 'userId': 123}
    producer.send('test-topic', event)
    producer.flush()

    # Consume event
    messages = []
    for message in consumer:
        messages.append(message.value)
        break  # Get first message

    assert len(messages) == 1
    assert messages[0] == event
```

## Authentication Integration Testing

```typescript
describe('Authentication Flow Integration', () => {
    it('should complete login flow with JWT', async () => {
        const response = await request(app)
            .post('/auth/login')
            .send({
                email: 'user@example.com',
                password: 'SecurePassword123!',
            })
            .expect(200);

        expect(response.body).toHaveProperty('token');
        const { token } = response.body;

        // Verify token is valid
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        expect(decoded).toMatchObject({
            email: 'user@example.com',
            exp: expect.any(Number),
        });

        // Use token to access protected endpoint
        const protectedResponse = await request(app)
            .get('/api/profile')
            .set('Authorization', `Bearer ${token}`)
            .expect(200);

        expect(protectedResponse.body.email).toBe('user@example.com');
    });

    it('should reject invalid tokens', async () => {
        await request(app)
            .get('/api/profile')
            .set('Authorization', 'Bearer invalid-token')
            .expect(401);
    });

    it('should refresh expired tokens', async () => {
        // Create expired token
        const expiredToken = jwt.sign(
            { email: 'user@example.com' },
            process.env.JWT_SECRET,
            { expiresIn: '-1h' }  // Expired 1 hour ago
        );

        const response = await request(app)
            .post('/auth/refresh')
            .send({ token: expiredToken })
            .expect(200);

        expect(response.body).toHaveProperty('token');

        // New token should be valid
        const decoded = jwt.verify(response.body.token, process.env.JWT_SECRET);
        expect(decoded.email).toBe('user@example.com');
    });
});
```

## Best Practices

### Test Data Management

```python
# Use fixtures for reusable test data
@pytest.fixture
def sample_users(db_session):
    """Create sample users for tests"""
    users = [
        User(email="alice@example.com", name="Alice"),
        User(email="bob@example.com", name="Bob"),
        User(email="charlie@example.com", name="Charlie"),
    ]
    db_session.add_all(users)
    db_session.commit()
    return users

def test_list_users(client, sample_users):
    """Test user listing with sample data"""
    response = client.get("/users")
    assert response.status_code == 200
    assert len(response.json()) == 3
```

### Cleanup Between Tests

```typescript
// Clear database between tests
beforeEach(async () => {
    await db.query('TRUNCATE TABLE users CASCADE');
    await db.query('TRUNCATE TABLE orders CASCADE');
});

// Or use transactions for automatic rollback
let transaction;

beforeEach(async () => {
    transaction = await db.transaction();
});

afterEach(async () => {
    await transaction.rollback();
});
```

### Environment Configuration

```python
# Separate test configuration
@pytest.fixture(scope="session")
def app_config():
    """Test-specific configuration"""
    return {
        "DATABASE_URL": "postgresql://test:test@localhost/test_db",
        "REDIS_URL": "redis://localhost:6379/1",  # Separate Redis DB
        "DEBUG": True,
        "TESTING": True,
    }
```

## Common Pitfalls

❌ **Shared State**: Tests affecting each other through shared database
✅ **Fix**: Use transactions or clear data between tests

❌ **Slow Tests**: Running full database migrations for each test
✅ **Fix**: Use in-memory databases or Docker containers

❌ **Flaky Tests**: Tests depending on external service availability
✅ **Fix**: Mock external APIs or use test containers

❌ **Missing Cleanup**: Test data polluting subsequent tests
✅ **Fix**: Use teardown fixtures or database transactions

❌ **Over-Mocking**: Mocking so much it's not really integration testing
✅ **Fix**: Test with real dependencies for critical paths

## Quality Standards

- **Test Speed**: Integration suite should complete in <5 minutes
- **Isolation**: Each test should be independent
- **Coverage**: Focus on API contracts and service boundaries
- **Reliability**: <1% flaky test rate
- **Real Dependencies**: Use actual databases, message queues when possible

---

**Skill Type**: Testing - Integration
**Complexity**: Moderate
**Typical Usage**: Activated when testing API endpoints, database operations, or service interactions
**Performance**: Optimized with test containers, transactions, and smart mocking
