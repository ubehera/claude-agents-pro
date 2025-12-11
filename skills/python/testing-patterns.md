---
name: testing-patterns
description: Load when user needs pytest fixtures, mocking, parametrize, coverage, property-based testing, or advanced testing patterns for Python applications
trigger_keywords: [pytest, testing, fixture, mock, unittest, parametrize, coverage, property based testing, hypothesis, test suite, integration test, unit test, test automation]
---

# Testing Patterns Skill

Production-grade Python testing strategies with pytest, hypothesis, and comprehensive test automation patterns.

## Overview

Modern Python testing using pytest ecosystem for unit, integration, and property-based testing. Enables high-confidence refactoring and rapid development cycles.

**When to Use**:
- Building test suites for libraries and applications
- Implementing CI/CD quality gates
- Practicing TDD or BDD workflows
- Ensuring code reliability and maintainability

## Core Concepts

### Pytest Basics

**Simple Test Structure**:
```python
# test_calculator.py
def add(x: int, y: int) -> int:
    return x + y

def test_add_positive_numbers():
    """Test addition of positive integers"""
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-1, -1) == -2

def test_add_mixed_signs():
    assert add(10, -5) == 5

# Run: pytest test_calculator.py
```

**Test Organization**:
```python
# tests/
#   conftest.py         # Shared fixtures
#   unit/
#     test_models.py
#     test_services.py
#   integration/
#     test_api.py
#     test_database.py

# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

### Fixtures

**Basic Fixtures**:
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

@pytest.fixture
def db_session() -> Session:
    """Provide database session for tests"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)

    yield session

    session.close()

@pytest.fixture
def sample_user(db_session: Session) -> User:
    """Create test user"""
    user = User(email="test@example.com", name="Test User")
    db_session.add(user)
    db_session.commit()
    return user

def test_user_creation(sample_user: User):
    assert sample_user.email == "test@example.com"
    assert sample_user.name == "Test User"
```

**Fixture Scopes**:
```python
@pytest.fixture(scope="function")  # Default: new instance per test
def temp_file():
    file = open("temp.txt", "w")
    yield file
    file.close()
    os.remove("temp.txt")

@pytest.fixture(scope="class")  # Shared across test class
def database():
    db = Database()
    db.connect()
    yield db
    db.disconnect()

@pytest.fixture(scope="module")  # Shared across test module
def app():
    app = create_app()
    yield app

@pytest.fixture(scope="session")  # Shared across entire test session
def docker_services():
    """Start Docker containers once for all tests"""
    subprocess.run(["docker-compose", "up", "-d"])
    yield
    subprocess.run(["docker-compose", "down"])
```

**Fixture Composition**:
```python
@pytest.fixture
def database():
    db = Database("test.db")
    db.connect()
    yield db
    db.disconnect()

@pytest.fixture
def populated_database(database):
    """Builds on database fixture"""
    database.execute("INSERT INTO users VALUES (1, 'Alice')")
    database.execute("INSERT INTO users VALUES (2, 'Bob')")
    return database

def test_user_count(populated_database):
    count = populated_database.query("SELECT COUNT(*) FROM users")
    assert count == 2
```

### Parametrized Tests

**Simple Parametrization**:
```python
import pytest

@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
    (5, 25),
])
def test_square(input: int, expected: int):
    assert input ** 2 == expected

# Generates 4 separate tests
```

**Multiple Parameters**:
```python
@pytest.mark.parametrize("base,exponent,expected", [
    (2, 3, 8),
    (3, 2, 9),
    (5, 0, 1),
    (10, 1, 10),
])
def test_power(base: int, exponent: int, expected: int):
    assert base ** exponent == expected
```

**Parametrize with IDs**:
```python
@pytest.mark.parametrize("email,valid", [
    ("user@example.com", True),
    ("invalid.email", False),
    ("user@", False),
    ("@example.com", False),
], ids=["valid", "no_at_sign", "no_domain", "no_local"])
def test_email_validation(email: str, valid: bool):
    assert validate_email(email) == valid

# Test output shows readable IDs:
# test_email_validation[valid] PASSED
# test_email_validation[no_at_sign] PASSED
```

**Combining Parametrize Decorators**:
```python
@pytest.mark.parametrize("x", [1, 2, 3])
@pytest.mark.parametrize("y", [10, 20])
def test_combinations(x: int, y: int):
    """Generates 6 tests (3 x 2 combinations)"""
    assert x + y > 0
```

### Mocking

**Mock with unittest.mock**:
```python
from unittest.mock import Mock, patch, MagicMock

# Mock objects
def test_api_call():
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": 1, "name": "Test"}

    with patch("requests.get", return_value=mock_response):
        result = fetch_user(1)
        assert result["name"] == "Test"

# Mock class methods
def test_database_save():
    with patch("database.Database.save") as mock_save:
        user = User(name="Alice")
        user.save()

        mock_save.assert_called_once()
        mock_save.assert_called_with(user)
```

**pytest-mock Plugin**:
```python
def test_user_creation(mocker):
    """Using pytest-mock for cleaner syntax"""
    mock_db = mocker.patch("app.database.save")

    create_user("alice@example.com", "Alice")

    mock_db.assert_called_once()
    args, kwargs = mock_db.call_args
    assert args[0].email == "alice@example.com"
```

**Spy on Real Objects**:
```python
def test_cache_hit(mocker):
    """Verify cache is used but don't mock logic"""
    cache = Cache()
    spy = mocker.spy(cache, "get")

    # Real implementation runs
    result = cache.get("key")

    # But we can verify it was called
    spy.assert_called_once_with("key")
```

### Property-Based Testing (Hypothesis)

**Basic Property Tests**:
```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_addition_commutative(x: int, y: int):
    """Addition is commutative: x + y == y + x"""
    assert x + y == y + x

@given(st.lists(st.integers()))
def test_reverse_twice(items: list[int]):
    """Reversing a list twice returns original"""
    assert list(reversed(list(reversed(items)))) == items

@given(st.text())
def test_string_length(s: str):
    """String length is non-negative"""
    assert len(s) >= 0
```

**Custom Strategies**:
```python
from hypothesis import given, strategies as st
from dataclasses import dataclass

@dataclass
class User:
    email: str
    age: int

# Custom strategy for User objects
user_strategy = st.builds(
    User,
    email=st.emails(),
    age=st.integers(min_value=18, max_value=120)
)

@given(user_strategy)
def test_user_age(user: User):
    """All generated users have valid age"""
    assert 18 <= user.age <= 120
    assert "@" in user.email
```

**Stateful Testing**:
```python
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize

class DatabaseStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.database = {}

    @rule(key=st.text(), value=st.integers())
    def insert(self, key: str, value: int):
        self.database[key] = value

    @rule(key=st.text())
    def delete(self, key: str):
        self.database.pop(key, None)

    @rule()
    def check_invariants(self):
        # Database invariants always hold
        assert isinstance(self.database, dict)

TestDatabase = DatabaseStateMachine.TestCase
```

## Advanced Patterns

### Test Markers

```python
import pytest

@pytest.mark.slow
def test_large_computation():
    """Skip in fast test runs"""
    ...

@pytest.mark.integration
def test_database_integration():
    """Integration test requiring database"""
    ...

@pytest.mark.skip(reason="API not yet implemented")
def test_future_feature():
    ...

@pytest.mark.skipif(sys.platform == "win32", reason="Unix only")
def test_unix_specific():
    ...

@pytest.mark.xfail(reason="Known bug #123")
def test_known_failure():
    ...

# Run specific markers:
# pytest -m "not slow"           # Skip slow tests
# pytest -m "integration"        # Only integration tests
# pytest -m "not (slow or integration)"  # Fast unit tests only
```

### Snapshot Testing (pytest-snapshot)

```python
def test_api_response(snapshot):
    """Compare API response to saved snapshot"""
    response = api.get_user(123)
    snapshot.assert_match(response)

# First run creates snapshot file
# Subsequent runs compare against saved snapshot
# Update snapshots: pytest --snapshot-update
```

### Coverage Analysis

```python
# pytest.ini
[pytest]
addopts =
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

# Run with coverage
# pytest --cov=src --cov-report=html

# Coverage report shows:
# - Line coverage per file
# - Branch coverage
# - Missing lines
```

### Async Testing

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async code"""
    result = await fetch_data()
    assert result["status"] == "ok"

@pytest.fixture
async def async_client():
    """Async fixture"""
    async with httpx.AsyncClient() as client:
        yield client

@pytest.mark.asyncio
async def test_api_call(async_client):
    response = await async_client.get("/users/1")
    assert response.status_code == 200
```

## Production-Ready Examples

### FastAPI Testing

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_user():
    """Test user creation endpoint"""
    response = client.post(
        "/users",
        json={"email": "test@example.com", "name": "Test User"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data

def test_get_user():
    # Create user first
    create_response = client.post("/users", json={...})
    user_id = create_response.json()["id"]

    # Fetch user
    response = client.get(f"/users/{user_id}")
    assert response.status_code == 200

def test_user_not_found():
    response = client.get("/users/99999")
    assert response.status_code == 404
```

### Database Testing with Fixtures

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.models import Base, User

@pytest.fixture(scope="function")
def db_session():
    """Isolated database for each test"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)

    yield session

    session.rollback()
    session.close()

@pytest.fixture
def test_users(db_session):
    """Populate test data"""
    users = [
        User(email="alice@example.com", name="Alice"),
        User(email="bob@example.com", name="Bob"),
    ]
    db_session.add_all(users)
    db_session.commit()
    return users

def test_query_users(db_session, test_users):
    users = db_session.query(User).all()
    assert len(users) == 2
    assert users[0].name == "Alice"
```

### Mock External APIs

```python
import pytest
import responses

@responses.activate
def test_github_api():
    """Mock HTTP responses"""
    responses.add(
        responses.GET,
        "https://api.github.com/users/octocat",
        json={"login": "octocat", "id": 1},
        status=200
    )

    user = fetch_github_user("octocat")
    assert user["login"] == "octocat"

# Or with pytest-httpserver
def test_api_mock(httpserver):
    httpserver.expect_request("/users/1").respond_with_json(
        {"id": 1, "name": "Test"}
    )

    response = fetch_user(httpserver.url_for("/users/1"))
    assert response["name"] == "Test"
```

### Test Organization Example

```python
# tests/conftest.py
import pytest
from app import create_app, db

@pytest.fixture(scope="session")
def app():
    """Create app for testing"""
    app = create_app({"TESTING": True})
    return app

@pytest.fixture(scope="function")
def client(app):
    """Test client"""
    return app.test_client()

@pytest.fixture(scope="function")
def db_session(app):
    """Database session"""
    with app.app_context():
        db.create_all()
        yield db
        db.drop_all()

# tests/unit/test_models.py
def test_user_model():
    user = User(email="test@example.com")
    assert user.email == "test@example.com"

# tests/integration/test_api.py
def test_create_user_endpoint(client, db_session):
    response = client.post("/users", json={...})
    assert response.status_code == 201
```

## Best Practices

### 1. AAA Pattern (Arrange-Act-Assert)
```python
def test_user_creation():
    # Arrange: Set up test data
    email = "test@example.com"
    name = "Test User"

    # Act: Execute the code under test
    user = create_user(email, name)

    # Assert: Verify the outcome
    assert user.email == email
    assert user.name == name
```

### 2. One Assertion Per Test (When Possible)
```python
# ❌ Multiple unrelated assertions
def test_user():
    user = create_user("test@example.com", "Test")
    assert user.email == "test@example.com"
    assert user.is_active == True
    assert user.created_at is not None

# ✅ Focused tests
def test_user_email():
    user = create_user("test@example.com", "Test")
    assert user.email == "test@example.com"

def test_user_active_by_default():
    user = create_user("test@example.com", "Test")
    assert user.is_active == True
```

### 3. Test Data Builders
```python
from dataclasses import dataclass

@dataclass
class UserBuilder:
    email: str = "test@example.com"
    name: str = "Test User"
    age: int = 30
    is_active: bool = True

    def with_email(self, email: str) -> "UserBuilder":
        self.email = email
        return self

    def inactive(self) -> "UserBuilder":
        self.is_active = False
        return self

    def build(self) -> User:
        return User(**self.__dict__)

# Usage
def test_inactive_user():
    user = UserBuilder().inactive().build()
    assert not user.is_active
```

### 4. Avoid Test Interdependence
```python
# ❌ Tests depend on execution order
def test_create_user():
    global user_id
    user_id = create_user("test@example.com")

def test_get_user():
    user = get_user(user_id)  # ❌ Depends on previous test

# ✅ Independent tests with fixtures
@pytest.fixture
def user_id():
    return create_user("test@example.com")

def test_get_user(user_id):
    user = get_user(user_id)  # ✅ Self-contained
```

## Common Pitfalls

❌ **Testing implementation instead of behavior**
```python
def test_internal_cache():
    cache = Cache()
    cache._internal_dict["key"] = "value"  # ❌ Testing internals
    assert cache._internal_dict["key"] == "value"
```
✅ Test public interface
```python
def test_cache_storage():
    cache = Cache()
    cache.set("key", "value")  # ✅ Test behavior
    assert cache.get("key") == "value"
```

❌ **Overly complex fixtures**
```python
@pytest.fixture
def everything():
    # ❌ Fixture does too much
    db = setup_database()
    users = create_users(db)
    orders = create_orders(users)
    return db, users, orders
```
✅ Composable fixtures
```python
@pytest.fixture
def db():
    return setup_database()

@pytest.fixture
def users(db):
    return create_users(db)

@pytest.fixture
def orders(users):
    return create_orders(users)
```

❌ **Not cleaning up resources**
```python
@pytest.fixture
def temp_file():
    file = open("temp.txt", "w")
    return file
    # ❌ File never closed, leaks resources
```
✅ Proper cleanup with yield
```python
@pytest.fixture
def temp_file():
    file = open("temp.txt", "w")
    yield file
    file.close()
    os.remove("temp.txt")
```

## Quality Standards

- **Coverage**: >85% line coverage for production code
- **Test Speed**: Unit tests <1s total, integration tests <10s
- **Test Isolation**: Each test runs independently
- **Readability**: Tests serve as documentation
- **Maintainability**: DRY principle with fixtures and helpers

---

**Skill Type**: Python - Testing
**Complexity**: Moderate
**Typical Usage**: Activated when Python specialists need testing patterns
**Performance**: Fast test execution with proper fixture scoping and mocking
