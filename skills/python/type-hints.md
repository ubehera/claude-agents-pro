---
name: type-hints
description: Load when user needs PEP 484, generics, protocols, TypedDict, overloads, or advanced type hinting patterns for Python static type checking
trigger_keywords: [type hint, typing, mypy, protocol, generic, typedict, typeddict, type annotation, overload, literal, union, optional, callable, type guard, type narrowing, type checking]
---

# Type Hints Skill

Advanced Python type hinting patterns for production-grade static type safety with mypy, Pyright, and type checkers.

## Overview

Type hints (PEP 484, 526, 544, 589, 604) enable static analysis, IDE intelligence, and runtime validation. Use for API contracts, library interfaces, and complex business logic.

**When to Use**:
- Libraries and public APIs requiring strong contracts
- Codebases with mypy --strict enforcement
- Complex data transformations needing type safety
- Team projects requiring self-documenting interfaces

## Core Concepts

### Type System Hierarchy

**Built-in Types**:
```python
from typing import Any, Optional, Union, Literal, Final

# Basic types
age: int = 30
price: float = 99.99
name: str = "Alice"
is_active: bool = True

# Optional (shorthand for Union[T, None])
middle_name: Optional[str] = None  # Python <3.10
middle_name: str | None = None     # Python 3.10+

# Union types
id_value: Union[int, str] = "abc123"  # Python <3.10
id_value: int | str = "abc123"        # Python 3.10+

# Literal types (exact values)
Direction = Literal["north", "south", "east", "west"]
Status = Literal[200, 404, 500]

# Final (constant)
MAX_RETRIES: Final[int] = 3
```

**Collection Types**:
```python
from typing import List, Dict, Set, Tuple, Sequence, Mapping

# Homogeneous collections
user_ids: list[int] = [1, 2, 3]
prices: set[float] = {19.99, 29.99}
settings: dict[str, bool] = {"debug": True}

# Heterogeneous tuples
coordinates: tuple[float, float] = (10.5, 20.3)
record: tuple[int, str, bool] = (1, "Alice", True)

# Variable-length tuples
numbers: tuple[int, ...] = (1, 2, 3, 4, 5)

# Abstract collections (prefer for function parameters)
def process_items(items: Sequence[str]) -> None:  # Accepts list, tuple
    ...

def get_config(config: Mapping[str, Any]) -> str:  # Accepts dict, MappingProxy
    ...
```

### Generics

**Generic Functions**:
```python
from typing import TypeVar, Generic, Sequence

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

def first(items: Sequence[T]) -> T | None:
    """Type-safe first element extraction"""
    return items[0] if items else None

def swap_pair(pair: tuple[K, V]) -> tuple[V, K]:
    """Swap tuple elements preserving types"""
    return pair[1], pair[0]

# Bounded TypeVar (constrains to specific types)
NumericT = TypeVar('NumericT', int, float)

def add(x: NumericT, y: NumericT) -> NumericT:
    return x + y  # type: ignore (arithmetic not proven safe)
```

**Generic Classes**:
```python
from typing import Generic, TypeVar
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    """Generic Result container"""
    value: T | None
    error: str | None

    @property
    def is_ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> T:
        if self.error:
            raise ValueError(self.error)
        assert self.value is not None
        return self.value

# Usage
user_result: Result[User] = fetch_user(123)
if user_result.is_ok:
    user = user_result.unwrap()  # type: User
```

### Protocols (Structural Subtyping)

**Duck Typing with Type Safety**:
```python
from typing import Protocol, runtime_checkable

class Drawable(Protocol):
    """Anything with a draw() method"""
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Rectangle:
    def draw(self) -> None:
        print("Drawing rectangle")

def render(shape: Drawable) -> None:
    """Accepts any object with draw() method"""
    shape.draw()

# No inheritance required - structural typing
render(Circle())      # ✅ OK
render(Rectangle())   # ✅ OK
```

**Runtime Checkable Protocols**:
```python
@runtime_checkable
class Closeable(Protocol):
    def close(self) -> None: ...

class FileHandler:
    def close(self) -> None:
        print("Closing file")

handler = FileHandler()
assert isinstance(handler, Closeable)  # ✅ Runtime check passes
```

### TypedDict

**Structured Dictionaries**:
```python
from typing import TypedDict, NotRequired, Required

# Basic TypedDict
class UserDict(TypedDict):
    id: int
    name: str
    email: str

# With optional fields (Python 3.11+)
class UserProfile(TypedDict):
    id: int
    name: str
    email: str
    bio: NotRequired[str]
    avatar_url: NotRequired[str]

# Total=False makes all fields optional
class PartialUser(TypedDict, total=False):
    id: int
    name: str
    email: str

# Mixed required/optional
class MixedUser(TypedDict):
    id: Required[int]      # Required even with total=False
    name: Required[str]
    nickname: NotRequired[str]

# Usage
def create_user(data: UserDict) -> int:
    # Type checker ensures all required keys present
    user_id = data["id"]
    return user_id

user: UserDict = {"id": 1, "name": "Alice", "email": "alice@example.com"}
```

### Function Overloads

**Type-Safe Function Variants**:
```python
from typing import overload, Literal

@overload
def get_user(id: int) -> User: ...

@overload
def get_user(id: int, include_deleted: Literal[True]) -> User | None: ...

def get_user(id: int, include_deleted: bool = False) -> User | None:
    """
    Fetch user by ID
    - If include_deleted=False, always returns User (raises if not found)
    - If include_deleted=True, may return None
    """
    user = db.query(User).filter_by(id=id)
    if not include_deleted:
        user = user.filter_by(deleted=False)

    result = user.first()
    if result is None and not include_deleted:
        raise ValueError(f"User {id} not found")
    return result

# Type checker understands different return types
active_user = get_user(123)              # type: User
maybe_user = get_user(123, True)         # type: User | None
```

## Advanced Patterns

### Callable Types

```python
from typing import Callable, ParamSpec, TypeVar
from functools import wraps

# Simple callable
Validator = Callable[[str], bool]

def validate_email(email: str) -> bool:
    return "@" in email

validators: list[Validator] = [validate_email]

# Callable with specific signature
Transform = Callable[[int, str], dict[str, Any]]

# ParamSpec for decorator type preservation (Python 3.10+)
P = ParamSpec('P')
R = TypeVar('R')

def log_calls(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator preserving exact signature"""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_calls
def add(x: int, y: int) -> int:
    return x + y

# Type checker knows: add(1, 2) -> int
```

### Type Guards

**Custom Type Narrowing**:
```python
from typing import TypeGuard

def is_str_list(items: list[Any]) -> TypeGuard[list[str]]:
    """Narrow type to list[str] if check passes"""
    return all(isinstance(item, str) for item in items)

data: list[Any] = ["a", "b", "c"]

if is_str_list(data):
    # Type checker knows data is list[str] here
    first: str = data[0]
    joined: str = ",".join(data)
```

### Covariance and Contravariance

```python
from typing import TypeVar, Generic

# Covariant (Producer)
T_co = TypeVar('T_co', covariant=True)

class Producer(Generic[T_co]):
    def get(self) -> T_co: ...

# Animal -> Dog relationship preserved
class Animal: pass
class Dog(Animal): pass

dog_producer: Producer[Dog] = ...
animal_producer: Producer[Animal] = dog_producer  # ✅ OK (covariant)

# Contravariant (Consumer)
T_contra = TypeVar('T_contra', contravariant=True)

class Consumer(Generic[T_contra]):
    def consume(self, item: T_contra) -> None: ...

animal_consumer: Consumer[Animal] = ...
dog_consumer: Consumer[Dog] = animal_consumer  # ✅ OK (contravariant)
```

### Self Type (Python 3.11+)

```python
from typing import Self

class Builder:
    def __init__(self) -> None:
        self._value = 0

    def add(self, x: int) -> Self:
        """Returns same type as class (enables chaining)"""
        self._value += x
        return self

    def multiply(self, x: int) -> Self:
        self._value *= x
        return self

    def build(self) -> int:
        return self._value

# Works with subclasses
class AdvancedBuilder(Builder):
    def power(self, exp: int) -> Self:
        self._value **= exp
        return self

result = AdvancedBuilder().add(5).power(2).build()  # type: int
```

## Production-Ready Examples

### FastAPI with Type Safety

```python
from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field

app = FastAPI()

class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
    age: int | None = Field(None, ge=0, le=150)

class User(BaseModel):
    id: int
    email: EmailStr
    name: str
    age: int | None
    is_active: bool = True

async def get_current_user() -> User:
    # Simulate auth
    return User(id=1, email="test@example.com", name="Test")

@app.post("/users", response_model=User, status_code=201)
async def create_user(
    data: UserCreate,
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Type-safe user creation"""
    # Type checker validates all fields
    new_user = User(
        id=123,
        email=data.email,
        name=data.name,
        age=data.age
    )
    return new_user
```

### SQLAlchemy 2.0 Type Safety

```python
from typing import Optional
from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, Session

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    bio: Mapped[Optional[str]] = mapped_column(String(500))

def get_user_by_email(session: Session, email: str) -> User | None:
    """Type-safe query with proper return type"""
    stmt = select(User).where(User.email == email)
    return session.scalar(stmt)
```

### Type-Safe Configuration

```python
from typing import Literal
from pydantic import BaseSettings, PostgresDsn, RedisDsn, validator

Environment = Literal["development", "staging", "production"]

class Settings(BaseSettings):
    # Type-safe environment
    environment: Environment = "development"
    debug: bool = False

    # Database
    database_url: PostgresDsn
    pool_size: int = 10

    # Redis
    redis_url: RedisDsn
    redis_max_connections: int = 50

    # Security
    secret_key: str
    algorithm: Literal["HS256", "HS512", "RS256"] = "HS256"
    access_token_expire_minutes: int = 30

    @validator("debug", always=True)
    def debug_production_check(cls, v: bool, values: dict) -> bool:
        if values.get("environment") == "production" and v:
            raise ValueError("Debug cannot be True in production")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False

# Usage
settings = Settings()  # Type-checked at runtime
```

## Best Practices

### 1. Start with `mypy --strict`
```ini
# mypy.ini
[mypy]
python_version = 3.11
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_generics = True
```

### 2. Prefer Protocols over ABCs
```python
# ❌ Rigid inheritance
from abc import ABC, abstractmethod

class Storage(ABC):
    @abstractmethod
    def save(self, data: bytes) -> None: ...

# ✅ Flexible structural typing
from typing import Protocol

class Storage(Protocol):
    def save(self, data: bytes) -> None: ...
```

### 3. Use `Sequence`/`Mapping` for Function Parameters
```python
# ❌ Overly specific
def process(items: list[str]) -> None: ...

# ✅ Accepts list, tuple, etc.
from typing import Sequence

def process(items: Sequence[str]) -> None: ...
```

### 4. Leverage TypedDict for API Contracts
```python
from typing import TypedDict

class APIResponse(TypedDict):
    success: bool
    data: dict[str, Any]
    error: str | None

def call_api() -> APIResponse:
    return {"success": True, "data": {}, "error": None}
```

### 5. Type Narrow with `assert` or Type Guards
```python
def process_user(user: User | None) -> str:
    assert user is not None  # Type narrowed to User
    return user.name

# Or use type guard
if user is None:
    raise ValueError("User required")
# Type narrowed to User below
return user.name
```

## Common Pitfalls

❌ **Using `Any` liberally** (defeats type safety)
```python
def process(data: Any) -> Any:  # ❌ No type safety
    return data
```
✅ Use generics or specific types
```python
T = TypeVar('T')
def process(data: T) -> T:  # ✅ Preserves type
    return data
```

❌ **Forgetting `-> None` on procedures**
```python
def save_user(user: User):  # ❌ Implicit Any return
    db.save(user)
```
✅ Explicit `None` return
```python
def save_user(user: User) -> None:  # ✅ Clear no return value
    db.save(user)
```

❌ **Not handling `None` in Optional types**
```python
def get_name(user: User | None) -> str:
    return user.name  # ❌ Type error: user might be None
```
✅ Type narrowing
```python
def get_name(user: User | None) -> str:
    if user is None:
        return "Unknown"
    return user.name  # ✅ Type checker knows user is not None
```

❌ **Mutating function parameters without indicating**
```python
def add_item(items: list[str], item: str) -> None:  # ❌ Unclear mutation
    items.append(item)
```
✅ Return new collection or document mutation
```python
def add_item(items: list[str], item: str) -> list[str]:
    """Returns new list with item added"""
    return items + [item]
```

## Quality Standards

- **Type Coverage**: 100% of public APIs and business logic
- **Mypy Compliance**: `mypy --strict` passes with zero errors
- **Protocol Usage**: Prefer protocols over abstract base classes
- **Documentation**: Type hints serve as executable documentation
- **IDE Support**: Full autocomplete and refactoring support

---

**Skill Type**: Python - Type System
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when Python specialists need type safety patterns
**Performance**: Zero runtime overhead (type hints are erased at runtime)
