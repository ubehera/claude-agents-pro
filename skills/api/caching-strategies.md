---
name: caching-strategies
description: Load when user needs cache headers, ETags, CDN, cache invalidation, or HTTP caching strategies and patterns
trigger_keywords: [caching, cache, etag, cache control, cdn, cache invalidation, cache strategy, http caching, redis cache, memcached, cache headers, stale while revalidate]
---

# Caching Strategies Skill

Production-grade caching patterns with HTTP cache headers, ETags, CDN integration, Redis/Memcached, and cache invalidation strategies.

## Overview

Caching reduces latency, bandwidth, and server load by serving previously computed responses. Implements multiple cache layers (browser, CDN, application, database) with appropriate invalidation strategies.

**When to Use**:
- High-traffic APIs with frequently accessed data
- Expensive computations or database queries
- Static or infrequently changing content
- Global content delivery (CDN)

## Core Concepts

### Cache Hierarchy

```yaml
Client:
  Browser Cache:
    - Cache-Control headers
    - ETags for validation
    - Service Worker cache

CDN:
  Edge Cache:
    - Geographic distribution
    - Cache-Control: public
    - Surrogate-Control headers

Server:
  Application Cache:
    - Redis/Memcached
    - In-memory caching
    - Database query cache

Database:
  Query Cache:
    - Materialized views
    - Read replicas
```

### Cache-Control Directives

```http
Cache-Control: public, max-age=3600
Cache-Control: private, max-age=300
Cache-Control: no-cache
Cache-Control: no-store
Cache-Control: must-revalidate
Cache-Control: stale-while-revalidate=86400
```

## HTTP Caching (Browser & CDN)

### Cache-Control Headers

```python
from fastapi import FastAPI, Response
from datetime import datetime, timedelta

app = FastAPI()

@app.get("/api/public-data")
async def get_public_data(response: Response):
    """Public data cached for 1 hour"""
    response.headers["Cache-Control"] = "public, max-age=3600"

    return {"data": "public content"}

@app.get("/api/user-data")
async def get_user_data(response: Response):
    """Private user data cached for 5 minutes"""
    response.headers["Cache-Control"] = "private, max-age=300"

    return {"data": "user-specific content"}

@app.get("/api/sensitive")
async def get_sensitive(response: Response):
    """Never cache sensitive data"""
    response.headers["Cache-Control"] = "no-store"

    return {"data": "sensitive information"}
```

### ETags (Conditional Requests)

```python
import hashlib
from fastapi import FastAPI, Request, Response, HTTPException

app = FastAPI()

def generate_etag(content: str) -> str:
    """Generate ETag from content hash"""
    return hashlib.md5(content.encode()).hexdigest()

@app.get("/api/resource")
async def get_resource(request: Request, response: Response):
    # Fetch data
    data = {"id": 123, "value": "content"}
    content = str(data)

    # Generate ETag
    etag = generate_etag(content)

    # Check If-None-Match header (client's cached ETag)
    client_etag = request.headers.get("If-None-Match")

    if client_etag == f'"{etag}"':
        # Content not modified, return 304
        raise HTTPException(status_code=304)

    # Content modified or first request
    response.headers["ETag"] = f'"{etag}"'
    response.headers["Cache-Control"] = "public, max-age=3600, must-revalidate"

    return data

# Client workflow:
# 1. GET /api/resource → Returns data + ETag: "abc123"
# 2. GET /api/resource + If-None-Match: "abc123" → Returns 304 (not modified)
```

### Last-Modified (Weak Validation)

```python
from datetime import datetime

@app.get("/api/document/{doc_id}")
async def get_document(doc_id: int, request: Request, response: Response):
    doc = await db.get_document(doc_id)

    # Last modified timestamp
    last_modified = doc.updated_at.strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Check If-Modified-Since header
    if_modified_since = request.headers.get("If-Modified-Since")

    if if_modified_since == last_modified:
        raise HTTPException(status_code=304)

    response.headers["Last-Modified"] = last_modified
    response.headers["Cache-Control"] = "public, max-age=3600"

    return doc
```

### Stale-While-Revalidate

```python
@app.get("/api/expensive")
async def get_expensive_data(response: Response):
    """
    Serve stale cache for 1 day while revalidating in background
    - Client uses cached data immediately
    - Background fetch updates cache
    """
    response.headers["Cache-Control"] = (
        "public, max-age=3600, stale-while-revalidate=86400"
    )

    # Expensive computation
    data = await compute_expensive_data()

    return data
```

## Application-Level Caching (Redis/Memcached)

### Redis Cache Decorator

```python
import redis
import json
from functools import wraps
from typing import Callable, Any

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache(ttl: int = 300):
    """Cache function result in Redis"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key from function name and arguments
            cache_key = f"cache:{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Cache miss - execute function
            result = await func(*args, **kwargs)

            # Store in cache
            redis_client.setex(cache_key, ttl, json.dumps(result))

            return result

        return wrapper
    return decorator

# Usage
@cache(ttl=3600)
async def get_user_profile(user_id: int) -> dict:
    """Cached for 1 hour"""
    return await db.query(User).filter_by(id=user_id).first()
```

### Cache-Aside Pattern

```python
import redis
import json
from typing import Optional

class CacheAside:
    """Cache-aside (lazy loading) pattern"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get(self, key: str, fetch_fn: Callable, ttl: int = 300) -> Any:
        """
        Get from cache, or fetch and cache if missing
        """
        # 1. Try cache first
        cached = self.redis.get(f"cache:{key}")
        if cached:
            return json.loads(cached)

        # 2. Cache miss - fetch from source
        data = await fetch_fn()

        # 3. Store in cache
        self.redis.setex(f"cache:{key}", ttl, json.dumps(data))

        return data

# Usage
cache_aside = CacheAside(redis_client)

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return await cache_aside.get(
        key=f"user:{user_id}",
        fetch_fn=lambda: fetch_user_from_db(user_id),
        ttl=3600
    )
```

### Write-Through Cache

```python
class WriteThroughCache:
    """Write-through cache pattern"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def set(self, key: str, value: Any, ttl: int = 300):
        """
        Write to database AND cache simultaneously
        """
        # 1. Write to database
        await db.save(key, value)

        # 2. Write to cache
        self.redis.setex(f"cache:{key}", ttl, json.dumps(value))

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache (always fresh)"""
        cached = self.redis.get(f"cache:{key}")
        if cached:
            return json.loads(cached)
        return None

# Usage
cache = WriteThroughCache(redis_client)

@app.post("/users")
async def create_user(user: UserCreate):
    new_user = User(**user.dict())
    await cache.set(f"user:{new_user.id}", new_user.dict(), ttl=3600)
    return new_user
```

### Write-Behind (Write-Back) Cache

```python
import asyncio
from collections import deque

class WriteBehindCache:
    """
    Write-behind cache with async persistence
    - Writes to cache immediately (fast)
    - Persists to database asynchronously (deferred)
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.write_queue = deque()
        self.batch_size = 100
        self.flush_interval = 5  # seconds

    async def set(self, key: str, value: Any, ttl: int = 300):
        """Write to cache immediately, queue for DB"""
        # 1. Immediate cache write
        self.redis.setex(f"cache:{key}", ttl, json.dumps(value))

        # 2. Queue for database write
        self.write_queue.append((key, value))

        # 3. Flush if batch size reached
        if len(self.write_queue) >= self.batch_size:
            await self.flush()

    async def flush(self):
        """Batch write to database"""
        if not self.write_queue:
            return

        batch = []
        while self.write_queue and len(batch) < self.batch_size:
            batch.append(self.write_queue.popleft())

        # Batch insert to database
        await db.bulk_insert(batch)

    async def background_flusher(self):
        """Background task to flush periodically"""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush()
```

## Cache Invalidation Strategies

### Time-Based Expiration (TTL)

```python
# Simple: Set TTL on cache keys
redis_client.setex("user:123", 3600, json.dumps(user_data))  # 1 hour

# FastAPI dependency
from fastapi import Depends

async def get_cached_user(user_id: int, ttl: int = 3600):
    key = f"user:{user_id}"
    cached = redis_client.get(key)

    if cached:
        return json.loads(cached)

    user = await fetch_user(user_id)
    redis_client.setex(key, ttl, json.dumps(user))
    return user
```

### Event-Based Invalidation

```python
from typing import List

class EventBasedCache:
    """Invalidate cache on specific events"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def invalidate_user(self, user_id: int):
        """Invalidate all user-related cache keys"""
        patterns = [
            f"user:{user_id}",
            f"user:{user_id}:profile",
            f"user:{user_id}:posts",
        ]

        for pattern in patterns:
            self.redis.delete(pattern)

    async def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)

# Usage
cache = EventBasedCache(redis_client)

@app.put("/users/{user_id}")
async def update_user(user_id: int, user: UserUpdate):
    # Update database
    updated_user = await db.update(user_id, user)

    # Invalidate cache
    await cache.invalidate_user(user_id)

    return updated_user
```

### Cache Tags (Multi-Level Invalidation)

```python
class TaggedCache:
    """Cache with tags for group invalidation"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def set(self, key: str, value: Any, tags: List[str], ttl: int = 300):
        """Store with tags"""
        # Store value
        self.redis.setex(f"cache:{key}", ttl, json.dumps(value))

        # Store tag associations
        for tag in tags:
            self.redis.sadd(f"tag:{tag}", key)
            self.redis.expire(f"tag:{tag}", ttl)

    async def invalidate_tag(self, tag: str):
        """Invalidate all keys with tag"""
        keys = self.redis.smembers(f"tag:{tag}")

        for key in keys:
            self.redis.delete(f"cache:{key}")

        self.redis.delete(f"tag:{tag}")

# Usage
tagged_cache = TaggedCache(redis_client)

# Cache user with tags
await tagged_cache.set(
    "user:123",
    user_data,
    tags=["user", "org:456"],  # Belongs to user and organization
    ttl=3600
)

# Invalidate all organization cache
await tagged_cache.invalidate_tag("org:456")
```

## Advanced Patterns

### Multi-Layer Cache

```python
from typing import Optional
import asyncio

class MultiLayerCache:
    """L1 (in-memory) + L2 (Redis) cache"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.local_cache = {}  # In-memory L1 cache
        self.local_ttl = 60    # 1 minute local cache

    async def get(self, key: str, fetch_fn: Callable) -> Any:
        # L1: Check in-memory cache
        if key in self.local_cache:
            data, expiry = self.local_cache[key]
            if time.time() < expiry:
                return data

        # L2: Check Redis
        cached = self.redis.get(f"cache:{key}")
        if cached:
            data = json.loads(cached)
            # Populate L1
            self.local_cache[key] = (data, time.time() + self.local_ttl)
            return data

        # Cache miss - fetch from source
        data = await fetch_fn()

        # Store in both layers
        self.redis.setex(f"cache:{key}", 3600, json.dumps(data))  # L2
        self.local_cache[key] = (data, time.time() + self.local_ttl)  # L1

        return data

    def invalidate_local(self):
        """Periodic cleanup of expired L1 cache"""
        now = time.time()
        expired = [k for k, (_, exp) in self.local_cache.items() if exp < now]
        for k in expired:
            del self.local_cache[k]
```

### Cache Warming

```python
async def warm_cache():
    """Pre-populate cache with frequently accessed data"""
    # Identify hot data
    popular_user_ids = await db.query(
        "SELECT user_id FROM analytics WHERE requests > 1000"
    )

    # Warm cache
    for user_id in popular_user_ids:
        user = await fetch_user(user_id)
        redis_client.setex(f"user:{user_id}", 3600, json.dumps(user))

    print(f"Warmed cache for {len(popular_user_ids)} users")

# Run on startup
@app.on_event("startup")
async def startup():
    await warm_cache()
```

### Probabilistic Early Expiration (Thundering Herd Prevention)

```python
import random

async def get_with_early_expiration(key: str, fetch_fn: Callable, ttl: int = 3600):
    """
    Probabilistically recompute before expiration to prevent thundering herd
    """
    cached = redis_client.get(f"cache:{key}")

    if cached:
        # Get TTL remaining
        remaining_ttl = redis_client.ttl(f"cache:{key}")

        # Probability of early recompute increases as TTL decreases
        # P(recompute) = 1 - (remaining_ttl / ttl)
        recompute_probability = 1 - (remaining_ttl / ttl)

        if random.random() < recompute_probability:
            # Early recompute (in background)
            asyncio.create_task(recompute_and_cache(key, fetch_fn, ttl))

        return json.loads(cached)

    # Cache miss - fetch and cache
    data = await fetch_fn()
    redis_client.setex(f"cache:{key}", ttl, json.dumps(data))
    return data

async def recompute_and_cache(key: str, fetch_fn: Callable, ttl: int):
    """Background recomputation"""
    data = await fetch_fn()
    redis_client.setex(f"cache:{key}", ttl, json.dumps(data))
```

## Production-Ready Examples

### Complete FastAPI Cache Middleware

```python
from fastapi import FastAPI, Request, Response
import redis
import hashlib

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class CacheMiddleware:
    def __init__(self, app: FastAPI, redis_client: redis.Redis):
        self.app = app
        self.redis = redis_client
        self.cacheable_methods = {"GET", "HEAD"}

    async def __call__(self, request: Request, call_next):
        # Only cache safe methods
        if request.method not in self.cacheable_methods:
            return await call_next(request)

        # Generate cache key from URL + query params
        cache_key = self.generate_key(request)

        # Check cache
        cached_response = self.redis.get(cache_key)
        if cached_response:
            return Response(
                content=cached_response,
                media_type="application/json",
                headers={"X-Cache": "HIT"}
            )

        # Cache miss - process request
        response = await call_next(request)

        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            # Cache with 5-minute TTL
            self.redis.setex(cache_key, 300, body)

            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers) | {"X-Cache": "MISS"}
            )

        return response

    def generate_key(self, request: Request) -> str:
        """Generate cache key from request"""
        key_parts = [
            request.method,
            str(request.url),
            request.headers.get("Accept", ""),
        ]
        key_string = ":".join(key_parts)
        return f"cache:{hashlib.md5(key_string.encode()).hexdigest()}"

app.add_middleware(CacheMiddleware, redis_client=redis_client)
```

## Best Practices

### 1. Set Appropriate TTLs
```python
# Static content: Long TTL
response.headers["Cache-Control"] = "public, max-age=31536000, immutable"

# Dynamic content: Short TTL
response.headers["Cache-Control"] = "public, max-age=300"

# Private data: Private cache only
response.headers["Cache-Control"] = "private, max-age=60"
```

### 2. Use ETags for Validation
```python
# ✅ Enable conditional requests
response.headers["ETag"] = f'"{generate_etag(content)}"'
response.headers["Cache-Control"] = "public, max-age=3600, must-revalidate"
```

### 3. Invalidate Proactively
```python
# ✅ Invalidate on writes
@app.put("/users/{user_id}")
async def update_user(user_id: int, user: UserUpdate):
    await db.update(user_id, user)
    redis_client.delete(f"cache:user:{user_id}")  # Invalidate cache
    return {"status": "updated"}
```

### 4. Monitor Cache Hit Rates
```python
from prometheus_client import Counter, Gauge

cache_hits = Counter("cache_hits_total", "Cache hits")
cache_misses = Counter("cache_misses_total", "Cache misses")
cache_hit_rate = Gauge("cache_hit_rate", "Cache hit rate")

# Track metrics
if cached:
    cache_hits.inc()
else:
    cache_misses.inc()

# Calculate hit rate
total = cache_hits._value.get() + cache_misses._value.get()
if total > 0:
    cache_hit_rate.set(cache_hits._value.get() / total)
```

### 5. Implement Cache Warming
```python
# ✅ Pre-populate cache on startup
@app.on_event("startup")
async def warm_cache():
    hot_keys = await identify_hot_data()
    for key in hot_keys:
        data = await fetch_data(key)
        redis_client.setex(f"cache:{key}", 3600, json.dumps(data))
```

## Common Pitfalls

❌ **Caching user-specific data with public cache**
```python
# ❌ Leaks data between users
response.headers["Cache-Control"] = "public, max-age=3600"  # BAD
```

❌ **Not invalidating on updates**
```python
# ❌ Stale cache after update
await db.update(user_id, user)
# Forgot to invalidate cache
```

❌ **Thundering herd problem**
```python
# ❌ All requests refetch simultaneously when cache expires
# ✅ Use probabilistic early expiration or cache locking
```

❌ **Caching errors**
```python
# ❌ Cache 500 errors
if response.status_code == 200:  # ✅ Only cache success
    redis_client.setex(cache_key, ttl, response_data)
```

## Quality Standards

- **Hit Rate**: Target >80% for frequently accessed data
- **TTL Strategy**: Match data volatility (static: days, dynamic: minutes)
- **Invalidation**: Event-based for critical data, TTL for less critical
- **Headers**: Always include Cache-Control and ETag headers
- **Monitoring**: Track hit rate, eviction rate, memory usage

---

**Skill Type**: API - Caching
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when optimizing API performance and reducing load
**Performance**: Well-tuned cache reduces latency by 10-100x for cached requests
