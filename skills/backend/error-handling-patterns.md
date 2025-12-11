---
name: error-handling-patterns
description: Load when user needs error handling strategies, exceptions, Result types, retry logic, circuit breakers, or graceful degradation patterns
trigger_keywords: [error handling, exception, try catch, result type, retry, circuit breaker, exponential backoff, error propagation, graceful degradation, fault tolerance]
---

# Error Handling Patterns

Production-grade error handling strategies for building resilient applications that gracefully handle failures and provide excellent debugging experiences across Python, TypeScript, Rust, and Go.

## Core Concepts

### Error Handling Philosophies

**Exceptions vs Result Types:**
- **Exceptions**: Traditional try-catch, disrupts control flow (Python, TypeScript, Java)
- **Result Types**: Explicit success/failure, functional approach (Rust, Go-style)
- **Error Codes**: C-style, requires discipline (legacy systems)
- **Option/Maybe Types**: For nullable values (Rust, functional languages)

**When to Use Each:**
- **Exceptions**: Unexpected errors, exceptional conditions
- **Result Types**: Expected errors, validation failures, explicit error handling
- **Panics/Crashes**: Unrecoverable errors, programming bugs

### Error Categories

**Recoverable Errors:**
- Network timeouts (retry with backoff)
- Missing files (provide defaults or fallback)
- Invalid user input (return validation errors)
- API rate limits (queue and retry)

**Unrecoverable Errors:**
- Out of memory (crash and restart)
- Stack overflow (programming bug)
- Null pointer/undefined (programming bug)

## Python Error Handling

### Custom Exception Hierarchy

```python
from datetime import datetime
from typing import Optional, Dict, Any

class ApplicationError(Exception):
    """Base exception for all application errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.code = code or "INTERNAL_ERROR"
        self.details = details or {}
        self.timestamp = datetime.utcnow()

class ValidationError(ApplicationError):
    """Raised when validation fails."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, code="VALIDATION_ERROR", **kwargs)
        if field:
            self.details["field"] = field

class NotFoundError(ApplicationError):
    """Raised when resource not found."""
    def __init__(self, resource: str, identifier: str, **kwargs):
        super().__init__(
            f"{resource} not found",
            code="NOT_FOUND",
            details={"resource": resource, "id": identifier},
            **kwargs
        )

class ExternalServiceError(ApplicationError):
    """Raised when external service fails."""
    def __init__(self, message: str, service: str, **kwargs):
        super().__init__(message, code="EXTERNAL_SERVICE_ERROR", **kwargs)
        self.details["service"] = service

# Usage
def get_user(user_id: str) -> User:
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise NotFoundError("User", user_id)
    return user
```

### Context Managers for Cleanup

```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def database_transaction(session) -> Generator:
    """Ensure transaction is committed or rolled back."""
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Usage
with database_transaction(db.session) as session:
    user = User(name="Alice")
    session.add(user)
    # Automatic commit or rollback
```

### Retry with Exponential Backoff

```python
import time
from functools import wraps
from typing import TypeVar, Callable, Type, Tuple

T = TypeVar('T')

def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt < max_attempts - 1:
                        sleep_time = backoff_factor ** attempt
                        time.sleep(sleep_time)
                        continue
                    raise
            raise RuntimeError("Retry logic failed")  # Should never reach
        return wrapper
    return decorator

# Usage
@retry(max_attempts=3, exceptions=(NetworkError, TimeoutError))
def fetch_data(url: str) -> dict:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()
```

## TypeScript Error Handling

### Custom Error Classes

```typescript
// Custom error classes
class ApplicationError extends Error {
    constructor(
        message: string,
        public code: string,
        public statusCode: number = 500,
        public details?: Record<string, any>
    ) {
        super(message);
        this.name = this.constructor.name;
        Error.captureStackTrace(this, this.constructor);
    }
}

class ValidationError extends ApplicationError {
    constructor(message: string, details?: Record<string, any>) {
        super(message, 'VALIDATION_ERROR', 400, details);
    }
}

class NotFoundError extends ApplicationError {
    constructor(resource: string, id: string) {
        super(
            `${resource} not found`,
            'NOT_FOUND',
            404,
            { resource, id }
        );
    }
}

class NetworkError extends ApplicationError {
    constructor(message: string, statusCode?: number) {
        super(message, 'NETWORK_ERROR', statusCode || 500);
    }
}

// Usage
function getUser(id: string): User {
    const user = users.find(u => u.id === id);
    if (!user) {
        throw new NotFoundError('User', id);
    }
    return user;
}
```

### Result Type Pattern

```typescript
// Result type for explicit error handling
type Result<T, E = Error> =
    | { ok: true; value: T }
    | { ok: false; error: E };

// Helper functions
function Ok<T>(value: T): Result<T, never> {
    return { ok: true, value };
}

function Err<E>(error: E): Result<never, E> {
    return { ok: false, error };
}

// Usage
function parseJSON<T>(json: string): Result<T, SyntaxError> {
    try {
        const value = JSON.parse(json) as T;
        return Ok(value);
    } catch (error) {
        return Err(error as SyntaxError);
    }
}

// Consuming Result
const result = parseJSON<User>(userJson);
if (result.ok) {
    console.log(result.value.name);
} else {
    console.error('Parse failed:', result.error.message);
}

// Chaining Results
function chain<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => Result<U, E>
): Result<U, E> {
    return result.ok ? fn(result.value) : result;
}
```

### Async Error Handling

```typescript
// Async/await with proper error handling
async function fetchUserOrders(userId: string): Promise<Order[]> {
    try {
        const user = await getUser(userId);
        const orders = await getOrders(user.id);
        return orders;
    } catch (error) {
        if (error instanceof NotFoundError) {
            return [];  // Return empty array for not found
        }
        if (error instanceof NetworkError) {
            // Retry logic
            return retryFetchOrders(userId);
        }
        // Re-throw unexpected errors
        throw error;
    }
}

// Promise error handling
function fetchData(url: string): Promise<Data> {
    return fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new NetworkError(`HTTP ${response.status}`, response.status);
            }
            return response.json();
        })
        .catch(error => {
            console.error('Fetch failed:', error);
            throw error;
        });
}
```

## Rust Error Handling

### Result and Option Types

```rust
use std::fs::File;
use std::io::{self, Read};

// Result type for operations that can fail
fn read_file(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;  // ? operator propagates errors
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

// Custom error types
#[derive(Debug)]
enum AppError {
    Io(io::Error),
    Parse(std::num::ParseIntError),
    NotFound(String),
    Validation(String),
}

impl From<io::Error> for AppError {
    fn from(error: io::Error) -> Self {
        AppError::Io(error)
    }
}

// Using custom error type
fn read_number_from_file(path: &str) -> Result<i32, AppError> {
    let contents = read_file(path)?;  // Auto-converts io::Error
    let number = contents.trim().parse()
        .map_err(AppError::Parse)?;   // Explicitly convert ParseIntError
    Ok(number)
}

// Option for nullable values
fn find_user(id: &str) -> Option<User> {
    users.iter().find(|u| u.id == id).cloned()
}

// Combining Option and Result
fn get_user_age(id: &str) -> Result<u32, AppError> {
    find_user(id)
        .ok_or_else(|| AppError::NotFound(id.to_string()))
        .map(|user| user.age)
}
```

## Go Error Handling

### Explicit Error Returns

```go
import (
    "errors"
    "fmt"
)

// Basic error handling
func getUser(id string) (*User, error) {
    user, err := db.QueryUser(id)
    if err != nil {
        return nil, fmt.Errorf("failed to query user: %w", err)
    }
    if user == nil {
        return nil, errors.New("user not found")
    }
    return user, nil
}

// Custom error types
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed for %s: %s", e.Field, e.Message)
}

// Sentinel errors for comparison
var (
    ErrNotFound     = errors.New("not found")
    ErrUnauthorized = errors.New("unauthorized")
    ErrInvalidInput = errors.New("invalid input")
)

// Error checking
user, err := getUser("123")
if err != nil {
    if errors.Is(err, ErrNotFound) {
        // Handle not found
    } else {
        // Handle other errors
    }
}

// Error wrapping and unwrapping
func processUser(id string) error {
    user, err := getUser(id)
    if err != nil {
        return fmt.Errorf("process user failed: %w", err)
    }
    // Process user
    return nil
}

// Unwrap errors
err := processUser("123")
if err != nil {
    var valErr *ValidationError
    if errors.As(err, &valErr) {
        fmt.Printf("Validation error: %s\n", valErr.Field)
    }
}
```

## Universal Patterns

### Pattern 1: Circuit Breaker

Prevent cascading failures in distributed systems.

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, TypeVar

T = TypeVar('T')

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: timedelta = timedelta(seconds=60),
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None

    def call(self, func: Callable[[], T]) -> T:
        """Execute function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            if datetime.now() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0

    def on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
circuit_breaker = CircuitBreaker()

def fetch_data():
    return circuit_breaker.call(lambda: external_api.get_data())
```

### Pattern 2: Error Aggregation

Collect multiple errors instead of failing on first error.

```typescript
class ErrorCollector {
    private errors: Error[] = [];

    add(error: Error): void {
        this.errors.push(error);
    }

    hasErrors(): boolean {
        return this.errors.length > 0;
    }

    getErrors(): Error[] {
        return [...this.errors];
    }

    throw(): never {
        if (this.errors.length === 1) {
            throw this.errors[0];
        }
        throw new AggregateError(
            this.errors,
            `${this.errors.length} errors occurred`
        );
    }
}

// Usage: Validate multiple fields
function validateUser(data: any): User {
    const errors = new ErrorCollector();

    if (!data.email) {
        errors.add(new ValidationError('Email is required'));
    } else if (!isValidEmail(data.email)) {
        errors.add(new ValidationError('Email is invalid'));
    }

    if (!data.name || data.name.length < 2) {
        errors.add(new ValidationError('Name must be at least 2 characters'));
    }

    if (!data.age || data.age < 18) {
        errors.add(new ValidationError('Age must be 18 or older'));
    }

    if (errors.hasErrors()) {
        errors.throw();
    }

    return data as User;
}
```

### Pattern 3: Graceful Degradation

Provide fallback functionality when errors occur.

```python
from typing import Optional, Callable, TypeVar

T = TypeVar('T')

def with_fallback(
    primary: Callable[[], T],
    fallback: Callable[[], T],
    log_error: bool = True
) -> T:
    """Try primary function, fall back to fallback on error."""
    try:
        return primary()
    except Exception as e:
        if log_error:
            logger.error(f"Primary function failed: {e}")
        return fallback()

# Usage
def get_user_profile(user_id: str) -> UserProfile:
    return with_fallback(
        primary=lambda: fetch_from_cache(user_id),
        fallback=lambda: fetch_from_database(user_id)
    )

# Multiple fallbacks
def get_exchange_rate(currency: str) -> float:
    return (
        try_function(lambda: api_provider_1.get_rate(currency))
        or try_function(lambda: api_provider_2.get_rate(currency))
        or try_function(lambda: cache.get_rate(currency))
        or DEFAULT_RATE
    )

def try_function(func: Callable[[], Optional[T]]) -> Optional[T]:
    try:
        return func()
    except Exception:
        return None
```

## Best Practices

### 1. Fail Fast - Validate Early

```python
def process_order(order_id: str) -> Order:
    """Process order with comprehensive error handling."""
    # Validate input immediately
    if not order_id:
        raise ValidationError("Order ID is required")

    # Fetch order
    order = db.get_order(order_id)
    if not order:
        raise NotFoundError("Order", order_id)

    # Process payment with error wrapping
    try:
        payment_result = payment_service.charge(order.total)
    except PaymentServiceError as e:
        logger.error(f"Payment failed for order {order_id}: {e}")
        raise ExternalServiceError(
            "Payment processing failed",
            service="payment_service",
            details={"order_id": order_id, "amount": order.total}
        ) from e

    # Update order
    order.status = "completed"
    order.payment_id = payment_result.id
    db.save(order)

    return order
```

### 2. Preserve Context

```python
# ✅ Include stack traces, metadata, timestamps
try:
    result = risky_operation()
except Exception as e:
    logger.error(
        "Operation failed",
        extra={
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "context": {"user_id": user.id, "operation": "payment"}
        },
        exc_info=True  # Include stack trace
    )
    raise
```

### 3. Meaningful Messages

```python
# ❌ BAD: Generic message
raise Exception("Error occurred")

# ✅ GOOD: Specific, actionable message
raise ValidationError(
    f"Email format is invalid. Expected format: user@domain.com, got: {email}",
    field="email"
)
```

### 4. Handle at Right Level

```python
# Catch where you can meaningfully handle
def api_handler():
    try:
        result = business_logic()
    except ValidationError as e:
        # Can handle - return 400 to client
        return {"error": str(e)}, 400
    except NotFoundError as e:
        # Can handle - return 404 to client
        return {"error": str(e)}, 404
    except Exception as e:
        # Cannot handle - log and return 500
        logger.exception("Unexpected error")
        return {"error": "Internal server error"}, 500
```

### 5. Clean Up Resources

```python
# ✅ Use context managers
with open("file.txt") as f:
    data = f.read()
    # File automatically closed

# ✅ Or try-finally
connection = None
try:
    connection = db.connect()
    result = connection.query("SELECT ...")
finally:
    if connection:
        connection.close()
```

## Common Pitfalls

### ❌ Catching Too Broadly

```python
# BAD: Hides bugs
try:
    result = operation()
except Exception:
    pass  # Swallows all errors

# GOOD: Catch specific errors
try:
    result = operation()
except NetworkError as e:
    logger.warning(f"Network error, retrying: {e}")
    result = retry_operation()
```

### ❌ Logging and Re-throwing

```python
# BAD: Creates duplicate log entries
try:
    result = operation()
except Exception as e:
    logger.error(f"Error: {e}")
    raise  # Same error logged twice

# GOOD: Log OR re-throw, not both
try:
    result = operation()
except Exception:
    # Transform and raise with context
    raise ApplicationError("Operation failed") from e
```

### ❌ Not Cleaning Up

```python
# BAD: Connection leak
def fetch_data():
    conn = db.connect()
    return conn.query("SELECT ...")  # Connection never closed

# GOOD: Cleanup guaranteed
def fetch_data():
    with db.connect() as conn:
        return conn.query("SELECT ...")
```

## Quality Standards

- **Fail Fast**: Validate input at entry points
- **Preserve Context**: Include stack traces, timestamps, metadata
- **Meaningful Messages**: Explain what happened and how to fix
- **Appropriate Logging**: Error = log, expected failure = don't spam
- **Handle at Right Level**: Catch where you can meaningfully handle
- **Clean Up Resources**: Use try-finally, context managers, defer
- **Type-Safe Errors**: Use typed errors when possible

---

**Skill Type**: Backend - Error Handling
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when implementing robust error handling strategies
**Languages**: Python, TypeScript, Rust, Go
