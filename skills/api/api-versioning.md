---
name: api-versioning
description: Load when user needs URL versioning, header versioning, deprecation strategies, breaking changes, or API versioning patterns
trigger_keywords: [api versioning, version, deprecation, breaking change, backwards compatibility, semantic versioning, url versioning, header versioning, api evolution]
---

# API Versioning Skill

Production-grade API versioning strategies including URL versioning, header-based versioning, deprecation workflows, and managing breaking changes.

## Overview

API versioning enables controlled evolution while maintaining backward compatibility for existing clients. Critical for public APIs and long-lived integrations.

**When to Use**:
- Public or partner APIs with external clients
- Breaking changes required (schema, behavior, security)
- Long-term API contracts (mobile apps, third-party integrations)
- Multiple API generations in production simultaneously

## Core Concepts

### Versioning Strategies

**URL Versioning** (Most Common):
```
GET /v1/users
GET /v2/users
```
- ✅ Simple, explicit, discoverable
- ✅ Easy caching (different URLs)
- ❌ Pollutes URL space

**Header Versioning**:
```
GET /users
Accept: application/vnd.myapi.v2+json
```
- ✅ Clean URLs
- ✅ Fine-grained versioning per resource
- ❌ Less discoverable, harder to test

**Query Parameter Versioning**:
```
GET /users?version=2
```
- ✅ Simple to implement
- ❌ Caching issues
- ❌ Not recommended for production

**Content Negotiation**:
```
GET /users
Accept: application/vnd.myapi+json;version=2
```
- ✅ Standards-based (HTTP)
- ❌ Complex client implementation

## URL Versioning (Recommended)

### FastAPI Implementation

**Version-Specific Routers**:
```python
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

app = FastAPI()

# V1 Models
class UserV1(BaseModel):
    id: int
    name: str
    email: str

# V2 Models (added role field)
class UserV2(BaseModel):
    id: int
    name: str
    email: str
    role: str = "user"  # New field with default

# V1 Router
router_v1 = APIRouter(prefix="/v1")

@router_v1.get("/users/{user_id}", response_model=UserV1)
async def get_user_v1(user_id: int) -> UserV1:
    user = await fetch_user(user_id)
    return UserV1(id=user.id, name=user.name, email=user.email)

# V2 Router
router_v2 = APIRouter(prefix="/v2")

@router_v2.get("/users/{user_id}", response_model=UserV2)
async def get_user_v2(user_id: int) -> UserV2:
    user = await fetch_user(user_id)
    return UserV2(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role
    )

# Mount routers
app.include_router(router_v1, tags=["v1"])
app.include_router(router_v2, tags=["v2"])
```

**Shared Logic with Version Adapters**:
```python
from typing import Union

# Domain model (internal)
class User:
    id: int
    name: str
    email: str
    role: str
    created_at: datetime

# Version adapters
def to_v1(user: User) -> UserV1:
    """Convert internal model to V1 API response"""
    return UserV1(id=user.id, name=user.name, email=user.email)

def to_v2(user: User) -> UserV2:
    """Convert internal model to V2 API response"""
    return UserV2(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role
    )

@router_v1.get("/users/{user_id}", response_model=UserV1)
async def get_user_v1(user_id: int) -> UserV1:
    user = await user_service.get(user_id)  # Shared service
    return to_v1(user)

@router_v2.get("/users/{user_id}", response_model=UserV2)
async def get_user_v2(user_id: int) -> UserV2:
    user = await user_service.get(user_id)  # Same service
    return to_v2(user)
```

## Header-Based Versioning

### Custom Header

```python
from fastapi import FastAPI, Header, HTTPException
from typing import Annotated

app = FastAPI()

async def get_api_version(
    api_version: Annotated[str, Header(alias="X-API-Version")] = "1"
) -> str:
    """Extract API version from header"""
    if api_version not in ["1", "2"]:
        raise HTTPException(400, "Unsupported API version")
    return api_version

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    version: str = Depends(get_api_version)
) -> Union[UserV1, UserV2]:
    user = await fetch_user(user_id)

    if version == "1":
        return to_v1(user)
    else:
        return to_v2(user)

# Client request:
# GET /users/123
# X-API-Version: 2
```

### Content Negotiation (Accept Header)

```python
from fastapi import Request, HTTPException

@app.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    accept = request.headers.get("accept", "")

    # Parse Accept header
    if "application/vnd.myapi.v1+json" in accept:
        version = "v1"
    elif "application/vnd.myapi.v2+json" in accept:
        version = "v2"
    else:
        raise HTTPException(406, "Unsupported media type")

    user = await fetch_user(user_id)

    if version == "v1":
        return to_v1(user)
    else:
        return to_v2(user)

# Client request:
# GET /users/123
# Accept: application/vnd.myapi.v2+json
```

## Deprecation Workflow

### Deprecation Headers

```python
from fastapi import Response
from datetime import datetime, timedelta

@router_v1.get("/users/{user_id}", response_model=UserV1)
async def get_user_v1(user_id: int, response: Response) -> UserV1:
    # Add deprecation headers
    sunset_date = datetime.now() + timedelta(days=180)  # 6 months

    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = sunset_date.strftime("%a, %d %b %Y %H:%M:%S GMT")
    response.headers["Link"] = '</v2/users>; rel="successor-version"'

    user = await fetch_user(user_id)
    return to_v1(user)

# Response headers:
# Deprecation: true
# Sunset: Mon, 01 Jun 2025 00:00:00 GMT
# Link: </v2/users>; rel="successor-version"
```

### Deprecation Notice in Response

```python
class DeprecatedResponse(BaseModel):
    data: UserV1
    _deprecated: dict = {
        "message": "This API version is deprecated",
        "sunset_date": "2025-06-01",
        "migration_guide": "https://docs.api.com/v1-to-v2"
    }

@router_v1.get("/users/{user_id}", response_model=DeprecatedResponse)
async def get_user_v1(user_id: int) -> DeprecatedResponse:
    user = await fetch_user(user_id)
    return DeprecatedResponse(data=to_v1(user))
```

### Sunset Timeline

```yaml
Deprecation Workflow:
  Day 0:
    - Announce deprecation (docs, blog, email)
    - Add Deprecation headers
    - Monitor V1 usage metrics

  Day 90:
    - Email active V1 users
    - Add warning logs for V1 calls
    - Publish migration guide

  Day 150:
    - Final warning email
    - Return 426 Upgrade Required for new clients

  Day 180:
    - Sunset V1 (return 410 Gone)
    - Keep read-only V1 for 30 days (optional)

  Day 210:
    - Remove V1 code
```

## Breaking vs Non-Breaking Changes

### Non-Breaking Changes (No version bump)

✅ **Adding optional fields**:
```python
# V2 (non-breaking)
class UserV2(BaseModel):
    id: int
    name: str
    email: str
    role: str = "user"  # Optional with default
```

✅ **Adding new endpoints**:
```python
@router_v2.post("/users/{user_id}/avatar")  # New endpoint
async def upload_avatar(...):
    ...
```

✅ **Relaxing validation**:
```python
# Before: email must be .com
# After: any valid email (more permissive)
```

### Breaking Changes (Requires new version)

❌ **Removing fields**:
```python
# V1
class UserV1(BaseModel):
    id: int
    name: str
    legacy_field: str  # Removed in V2

# V2 (breaking)
class UserV2(BaseModel):
    id: int
    name: str
    # legacy_field removed
```

❌ **Renaming fields**:
```python
# V1
class UserV1(BaseModel):
    name: str

# V2 (breaking)
class UserV2(BaseModel):
    full_name: str  # Renamed from "name"
```

❌ **Changing field types**:
```python
# V1
class UserV1(BaseModel):
    age: str  # "25"

# V2 (breaking)
class UserV2(BaseModel):
    age: int  # 25
```

❌ **Stricter validation**:
```python
# Before: any string
# After: must match regex pattern (more restrictive)
```

❌ **Changing behavior**:
```python
# V1: Sorts by name ascending
# V2: Sorts by created_at descending (breaking)
```

## Advanced Patterns

### Semantic Versioning for APIs

```
MAJOR.MINOR.PATCH

v1.0.0  - Initial release
v1.1.0  - Added optional fields (non-breaking)
v1.1.1  - Bug fix (non-breaking)
v2.0.0  - Removed field (breaking)
```

**URL Mapping**:
```
/v1/users  → Maps to latest v1.x.x (e.g., v1.2.5)
/v2/users  → Maps to latest v2.x.x (e.g., v2.0.1)
```

### Feature Flags for Gradual Rollout

```python
from typing import Optional

async def get_user_with_feature_flag(
    user_id: int,
    enable_new_format: Optional[bool] = False
) -> Union[UserV1, UserV2]:
    """Allow clients to opt-in to new format"""
    user = await fetch_user(user_id)

    if enable_new_format:
        return to_v2(user)
    else:
        return to_v1(user)

# Client can test V2 without full migration:
# GET /users/123?enable_new_format=true
```

### Version Negotiation

```python
from packaging import version as pkg_version

def negotiate_version(requested: str, supported: list[str]) -> str:
    """Select best matching API version"""
    requested_ver = pkg_version.parse(requested)

    # Find highest compatible version
    compatible = [
        v for v in supported
        if pkg_version.parse(v).major == requested_ver.major
    ]

    if not compatible:
        raise HTTPException(400, f"No compatible version for {requested}")

    return max(compatible, key=lambda v: pkg_version.parse(v))

# Client requests v1.0.0 → Gets v1.2.3 (latest v1.x)
```

## Production-Ready Examples

### Multi-Version API with Shared Services

```python
from fastapi import FastAPI
from app.services import UserService

app = FastAPI()

# Shared service layer (version-agnostic)
user_service = UserService()

# V1 API
router_v1 = APIRouter(prefix="/v1", tags=["v1"])

@router_v1.get("/users/{user_id}", response_model=UserV1, deprecated=True)
async def get_user_v1(user_id: int, response: Response) -> UserV1:
    # Add deprecation headers
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "Sun, 01 Jun 2025 00:00:00 GMT"

    # Fetch from shared service
    user = await user_service.get(user_id)
    return to_v1(user)

# V2 API
router_v2 = APIRouter(prefix="/v2", tags=["v2"])

@router_v2.get("/users/{user_id}", response_model=UserV2)
async def get_user_v2(user_id: int) -> UserV2:
    user = await user_service.get(user_id)
    return to_v2(user)

app.include_router(router_v1)
app.include_router(router_v2)
```

### GraphQL Versioning (Field Deprecation)

```python
import strawberry

@strawberry.type
class User:
    id: strawberry.ID
    name: str

    @strawberry.field(deprecation_reason="Use 'fullName' instead")
    def full_name_legacy(self) -> str:
        return self.name

    @strawberry.field
    def full_name(self) -> str:
        return self.name

# Schema introspection shows deprecation
"""
type User {
  fullNameLegacy: String @deprecated(reason: "Use 'fullName' instead")
  fullName: String
}
"""
```

### Client SDK Versioning

```python
# Client library with version support
from myapi_client import Client

# V1 client
client_v1 = Client(version="v1")
user = client_v1.users.get(123)  # Returns UserV1

# V2 client
client_v2 = Client(version="v2")
user = client_v2.users.get(123)  # Returns UserV2
```

## Best Practices

### 1. Version Only When Necessary
```python
# ❌ New version for every change
v3.get("/users")  # Just added a field

# ✅ New version only for breaking changes
v2.get("/users")  # Removed field, changed behavior
```

### 2. Default to Latest Stable Version
```python
# Redirect unversioned requests to latest
@app.get("/users")
async def get_users_unversioned():
    return RedirectResponse(url="/v2/users")
```

### 3. Maintain N-1 Versions Minimum
```yaml
Support Policy:
  Current: v3 (full support)
  Previous: v2 (security updates only)
  Legacy: v1 (deprecated, sunset in 6 months)
```

### 4. Communicate Changes Proactively
```yaml
Communication Timeline:
  - Blog post announcing new version
  - Email to API consumers
  - Deprecation headers 6 months in advance
  - Migration guide with code examples
  - Sunset date in documentation
```

### 5. Monitor Version Usage
```python
from prometheus_client import Counter

api_version_counter = Counter(
    "api_requests_total",
    "API requests by version",
    ["version", "endpoint"]
)

@app.middleware("http")
async def track_api_version(request: Request, call_next):
    version = request.url.path.split("/")[1]  # Extract version
    response = await call_next(request)
    api_version_counter.labels(version=version, endpoint=request.url.path).inc()
    return response
```

## Common Pitfalls

❌ **Versioning everything**
```python
# ❌ Overkill
/v1/health  # Health check doesn't need versioning
```

❌ **No deprecation timeline**
```python
# ❌ Surprise breaking change
# V1 works → V1 removed (no warning)
```

❌ **Micro-versioning**
```python
# ❌ Too granular
/v1.2.3/users
```

❌ **Not documenting breaking changes**
```yaml
# ❌ Undocumented change
# V1: Returns 200 with empty array
# V2: Returns 404 (breaking, not documented)
```

## Quality Standards

- **Support Policy**: Minimum N-1 version support
- **Deprecation Notice**: 6+ months for public APIs
- **Documentation**: Migration guides for each major version
- **Monitoring**: Track version usage and adoption rates
- **Testing**: Integration tests for all supported versions

---

**Skill Type**: API - Versioning
**Complexity**: Moderate
**Typical Usage**: Activated when evolving APIs with breaking changes
**Performance**: Minimal overhead (routing logic only)
