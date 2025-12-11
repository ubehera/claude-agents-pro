---
name: rate-limiting
description: Load when user needs token bucket, sliding window, fixed window, distributed rate limiting, or API rate limiting patterns
trigger_keywords: [rate limiting, rate limit, throttling, token bucket, sliding window, fixed window, leaky bucket, api throttle, quota, distributed rate limiting, redis rate limit]
---

# Rate Limiting Skill

Production-grade API rate limiting with token bucket, sliding window, fixed window algorithms, and distributed implementations using Redis.

## Overview

Rate limiting protects APIs from abuse, ensures fair resource allocation, and prevents system overload. Essential for public APIs and multi-tenant systems.

**When to Use**:
- Public APIs requiring usage quotas
- Preventing brute-force attacks
- Fair resource allocation across tenants
- Protecting downstream services from overload

## Core Concepts

### Rate Limiting Algorithms

**Fixed Window**:
```
Time:    [0-60s] [60-120s] [120-180s]
Limit:     100      100       100
Requests:   95       5        103 (3 rejected)
```
- ✅ Simple implementation
- ❌ Burst at window boundaries (190 requests in 1 second)

**Sliding Window**:
```
Time:     [t-60s ... t]
Limit:    100 requests per 60s window
Requests: Smooth distribution
```
- ✅ Prevents burst attacks
- ✅ Fair distribution
- ❌ More complex, memory-intensive

**Token Bucket**:
```
Bucket:   [●●●○○] (3 tokens available, max 5)
Rate:     1 token/second refill
Request:  -1 token, allow if tokens > 0
```
- ✅ Allows bursts (up to bucket capacity)
- ✅ Smooth long-term rate
- ✅ Industry standard

**Leaky Bucket**:
```
Queue:    [Req1, Req2, Req3, ...]
Rate:     Process 10 req/sec constant
Overflow: Reject when queue full
```
- ✅ Constant output rate
- ❌ No burst allowance

## Fixed Window (Simple)

### In-Memory Implementation

```python
import time
from typing import Dict
from dataclasses import dataclass
from threading import Lock

@dataclass
class RateLimitState:
    count: int
    window_start: float

class FixedWindowRateLimiter:
    """Simple fixed-window rate limiter"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: Dict[str, RateLimitState] = {}
        self.lock = Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()

        with self.lock:
            if client_id not in self.clients:
                self.clients[client_id] = RateLimitState(count=1, window_start=now)
                return True

            state = self.clients[client_id]
            elapsed = now - state.window_start

            # Reset window if expired
            if elapsed >= self.window_seconds:
                state.count = 1
                state.window_start = now
                return True

            # Check limit
            if state.count < self.max_requests:
                state.count += 1
                return True

            return False

# Usage
limiter = FixedWindowRateLimiter(max_requests=100, window_seconds=60)

for i in range(105):
    if limiter.is_allowed("user123"):
        print(f"Request {i+1}: Allowed")
    else:
        print(f"Request {i+1}: Rate limited")
```

### FastAPI Middleware

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import time

app = FastAPI()

limiter = FixedWindowRateLimiter(max_requests=100, window_seconds=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Extract client ID (IP, API key, user ID)
    client_id = request.client.host

    if not limiter.is_allowed(client_id):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + 60)
            }
        )

    response = await call_next(request)

    # Add rate limit headers
    state = limiter.clients.get(client_id)
    if state:
        remaining = max(0, limiter.max_requests - state.count)
        response.headers["X-RateLimit-Limit"] = str(limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(state.window_start + limiter.window_seconds))

    return response
```

## Sliding Window (Accurate)

### Redis Implementation

```python
import redis
import time
from typing import Optional

class SlidingWindowRateLimiter:
    """Sliding window rate limiter using Redis sorted sets"""

    def __init__(self, redis_client: redis.Redis, max_requests: int, window_seconds: int):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if request is allowed
        Returns: (is_allowed, remaining_quota)
        """
        key = f"rate_limit:{client_id}"
        now = time.time()
        window_start = now - self.window_seconds

        # Use Redis pipeline for atomicity
        pipe = self.redis.pipeline()

        # Remove old entries outside window
        pipe.zremrangebyscore(key, 0, window_start)

        # Count requests in current window
        pipe.zcard(key)

        # Add current request timestamp
        pipe.zadd(key, {str(now): now})

        # Set expiry on key
        pipe.expire(key, self.window_seconds)

        results = pipe.execute()

        count = results[1]  # zcard result

        if count < self.max_requests:
            remaining = self.max_requests - count - 1
            return True, remaining
        else:
            # Remove the request we just added
            self.redis.zrem(key, str(now))
            return False, 0

# Usage
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
limiter = SlidingWindowRateLimiter(redis_client, max_requests=100, window_seconds=60)

allowed, remaining = limiter.is_allowed("user123")
if allowed:
    print(f"Request allowed. {remaining} requests remaining")
else:
    print("Rate limit exceeded")
```

## Token Bucket (Industry Standard)

### Redis Implementation

```python
import redis
import time
from typing import Optional

class TokenBucketRateLimiter:
    """Token bucket rate limiter using Redis"""

    def __init__(
        self,
        redis_client: redis.Redis,
        capacity: int,
        refill_rate: float,  # tokens per second
    ):
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate

    def is_allowed(self, client_id: str, tokens: int = 1) -> tuple[bool, int]:
        """
        Check if request is allowed and consume tokens
        Returns: (is_allowed, tokens_remaining)
        """
        key = f"token_bucket:{client_id}"
        now = time.time()

        # Lua script for atomic token bucket logic
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        -- Get current state
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now

        -- Calculate refill
        local elapsed = now - last_refill
        local new_tokens = math.min(capacity, tokens + (elapsed * refill_rate))

        -- Check if enough tokens
        if new_tokens >= tokens_requested then
            new_tokens = new_tokens - tokens_requested
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
            return {1, math.floor(new_tokens)}  -- allowed
        else
            redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)
            return {0, math.floor(new_tokens)}  -- denied
        end
        """

        result = self.redis.eval(
            lua_script,
            1,  # num keys
            key,
            self.capacity,
            self.refill_rate,
            tokens,
            now
        )

        is_allowed = bool(result[0])
        remaining = int(result[1])

        return is_allowed, remaining

# Usage
redis_client = redis.Redis(host='localhost', port=6379)
limiter = TokenBucketRateLimiter(
    redis_client,
    capacity=100,      # Max burst of 100 requests
    refill_rate=10.0   # 10 tokens/second = 600 req/min
)

allowed, remaining = limiter.is_allowed("user123", tokens=1)
if allowed:
    print(f"Request allowed. {remaining} tokens remaining")
else:
    print("Rate limit exceeded")
```

### FastAPI Dependency

```python
from fastapi import FastAPI, Depends, HTTPException, Request

app = FastAPI()

# Initialize rate limiter
redis_client = redis.Redis(host='localhost', port=6379)
limiter = TokenBucketRateLimiter(redis_client, capacity=100, refill_rate=10.0)

async def check_rate_limit(request: Request):
    """FastAPI dependency for rate limiting"""
    # Extract client ID (prefer API key > user ID > IP)
    client_id = request.headers.get("X-API-Key") or request.client.host

    allowed, remaining = limiter.is_allowed(client_id, tokens=1)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": "6",  # Wait ~6 seconds for 1 token at 10/sec
                "X-RateLimit-Remaining": "0"
            }
        )

    # Store remaining for response headers
    request.state.rate_limit_remaining = remaining

@app.get("/api/data", dependencies=[Depends(check_rate_limit)])
async def get_data(request: Request):
    return {"data": "value"}

@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    response = await call_next(request)

    # Add rate limit headers
    if hasattr(request.state, "rate_limit_remaining"):
        response.headers["X-RateLimit-Limit"] = "100"
        response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)

    return response
```

## Advanced Patterns

### Multi-Tier Rate Limiting

```python
from typing import List
from dataclasses import dataclass

@dataclass
class RateLimitTier:
    max_requests: int
    window_seconds: int
    name: str

class MultiTierRateLimiter:
    """Multiple rate limit tiers (second, minute, hour, day)"""

    def __init__(self, redis_client: redis.Redis, tiers: List[RateLimitTier]):
        self.redis = redis_client
        self.tiers = tiers

    def is_allowed(self, client_id: str) -> tuple[bool, str]:
        """
        Check all tiers, deny if any exceeded
        Returns: (is_allowed, tier_name_if_blocked)
        """
        for tier in self.tiers:
            limiter = SlidingWindowRateLimiter(
                self.redis,
                tier.max_requests,
                tier.window_seconds
            )
            allowed, _ = limiter.is_allowed(f"{client_id}:{tier.name}")

            if not allowed:
                return False, tier.name

        return True, ""

# Usage
tiers = [
    RateLimitTier(max_requests=10, window_seconds=1, name="per_second"),
    RateLimitTier(max_requests=100, window_seconds=60, name="per_minute"),
    RateLimitTier(max_requests=1000, window_seconds=3600, name="per_hour"),
]

limiter = MultiTierRateLimiter(redis_client, tiers)

allowed, blocked_tier = limiter.is_allowed("user123")
if not allowed:
    print(f"Rate limit exceeded: {blocked_tier}")
```

### Per-Endpoint Rate Limiting

```python
from fastapi import FastAPI, Depends, Request
from functools import wraps

app = FastAPI()

def rate_limit(max_requests: int, window_seconds: int):
    """Decorator for per-endpoint rate limiting"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            client_id = request.client.host
            endpoint = request.url.path

            # Create endpoint-specific limiter
            limiter = FixedWindowRateLimiter(max_requests, window_seconds)
            key = f"{client_id}:{endpoint}"

            if not limiter.is_allowed(key):
                raise HTTPException(429, "Rate limit exceeded")

            return await func(*args, request=request, **kwargs)

        return wrapper
    return decorator

@app.get("/expensive")
@rate_limit(max_requests=10, window_seconds=60)  # 10 req/min
async def expensive_endpoint(request: Request):
    return {"message": "Expensive operation"}

@app.get("/cheap")
@rate_limit(max_requests=100, window_seconds=60)  # 100 req/min
async def cheap_endpoint(request: Request):
    return {"message": "Cheap operation"}
```

### Quota-Based Limiting (API Keys)

```python
from typing import Dict

class QuotaRateLimiter:
    """Per-API-key quota with different limits"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        # In production, load from database
        self.quotas: Dict[str, int] = {
            "free_tier": 100,      # 100 req/hour
            "pro_tier": 1000,      # 1000 req/hour
            "enterprise_tier": 10000,  # 10000 req/hour
        }

    def get_tier(self, api_key: str) -> str:
        """Look up tier for API key"""
        # In production, query database
        tier_map = {
            "sk_free_xxx": "free_tier",
            "sk_pro_xxx": "pro_tier",
            "sk_ent_xxx": "enterprise_tier",
        }
        return tier_map.get(api_key, "free_tier")

    def is_allowed(self, api_key: str) -> tuple[bool, int, int]:
        """
        Returns: (allowed, remaining, limit)
        """
        tier = self.get_tier(api_key)
        limit = self.quotas[tier]

        limiter = SlidingWindowRateLimiter(self.redis, limit, 3600)  # 1 hour
        allowed, remaining = limiter.is_allowed(api_key)

        return allowed, remaining, limit

# FastAPI integration
async def check_api_key_quota(api_key: str = Header(...)):
    quota_limiter = QuotaRateLimiter(redis_client)
    allowed, remaining, limit = quota_limiter.is_allowed(api_key)

    if not allowed:
        raise HTTPException(
            429,
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
            }
        )

    return {"remaining": remaining, "limit": limit}
```

## Production-Ready Examples

### Distributed Rate Limiting with Redis Cluster

```python
from redis.cluster import RedisCluster

class DistributedRateLimiter:
    """Production-grade distributed rate limiter"""

    def __init__(self, redis_cluster: RedisCluster):
        self.redis = redis_cluster

    def is_allowed(
        self,
        client_id: str,
        max_requests: int,
        window_seconds: int
    ) -> dict:
        """
        Check rate limit across distributed Redis cluster
        Returns detailed rate limit info
        """
        key = f"rate_limit:{client_id}"
        now = time.time()
        window_start = now - window_seconds

        # Atomic Lua script
        lua_script = """
        local key = KEYS[1]
        local window_start = tonumber(ARGV[1])
        local max_requests = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local window_seconds = tonumber(ARGV[4])

        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

        -- Count current requests
        local current = redis.call('ZCARD', key)

        if current < max_requests then
            redis.call('ZADD', key, now, now)
            redis.call('EXPIRE', key, window_seconds)
            return {1, max_requests - current - 1, now + window_seconds}
        else
            return {0, 0, now + window_seconds}
        end
        """

        result = self.redis.eval(
            lua_script,
            1,
            key,
            window_start,
            max_requests,
            now,
            window_seconds
        )

        is_allowed = bool(result[0])
        remaining = int(result[1])
        reset_at = int(result[2])

        return {
            "allowed": is_allowed,
            "limit": max_requests,
            "remaining": remaining,
            "reset_at": reset_at,
            "retry_after": window_seconds if not is_allowed else None
        }

# Usage
startup_nodes = [{"host": "redis1", "port": 6379}, {"host": "redis2", "port": 6379}]
redis_cluster = RedisCluster(startup_nodes=startup_nodes)

limiter = DistributedRateLimiter(redis_cluster)
info = limiter.is_allowed("user123", max_requests=100, window_seconds=60)

if info["allowed"]:
    print(f"Allowed. {info['remaining']} remaining")
else:
    print(f"Denied. Retry after {info['retry_after']} seconds")
```

### Complete FastAPI Integration

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import redis

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class RateLimitMiddleware:
    def __init__(self, app: FastAPI, redis_client: redis.Redis):
        self.app = app
        self.limiter = TokenBucketRateLimiter(
            redis_client,
            capacity=100,
            refill_rate=10.0
        )

    async def __call__(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Extract client ID
        api_key = request.headers.get("X-API-Key")
        client_id = api_key or f"ip:{request.client.host}"

        # Check rate limit
        allowed, remaining = self.limiter.is_allowed(client_id)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later."
                },
                headers={
                    "Retry-After": "6",
                    "X-RateLimit-Limit": "100",
                    "X-RateLimit-Remaining": "0",
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = "100"
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

app.add_middleware(RateLimitMiddleware, redis_client=redis_client)
```

## Best Practices

### 1. Use Token Bucket for Public APIs
```python
# ✅ Token bucket allows bursts within limits
limiter = TokenBucketRateLimiter(capacity=100, refill_rate=10.0)
```

### 2. Implement Multiple Tiers
```python
# ✅ Prevent abuse at multiple time scales
tiers = [
    RateLimitTier(10, 1, "per_second"),
    RateLimitTier(100, 60, "per_minute"),
    RateLimitTier(1000, 3600, "per_hour"),
]
```

### 3. Return Informative Headers
```python
# ✅ Help clients implement backoff
headers = {
    "X-RateLimit-Limit": "100",
    "X-RateLimit-Remaining": "23",
    "X-RateLimit-Reset": "1640000000",
    "Retry-After": "30"
}
```

### 4. Use Redis for Distributed Systems
```python
# ✅ Consistent rate limiting across instances
redis_cluster = RedisCluster(...)
limiter = DistributedRateLimiter(redis_cluster)
```

### 5. Differentiate by User Tier
```python
# ✅ Fair resource allocation
quotas = {
    "free": 100,
    "pro": 1000,
    "enterprise": 10000
}
```

## Common Pitfalls

❌ **Fixed window burst vulnerability**
```python
# ❌ 200 requests in 1 second at window boundary
# T=59s: 100 requests (allowed)
# T=60s: 100 requests (allowed, new window)
```

❌ **Not handling distributed race conditions**
```python
# ❌ Multiple servers increment counter simultaneously
# Use Lua scripts or INCR with EXPIRE
```

❌ **Rate limiting expensive operations equally**
```python
# ❌ Same limit for GET /user and POST /export-data
# ✅ Different limits per endpoint
```

❌ **No retry guidance for clients**
```python
# ❌ Just return 429
# ✅ Include Retry-After header
```

## Quality Standards

- **Algorithm**: Token bucket for most cases, sliding window for strict limits
- **Distribution**: Redis/Memcached for multi-instance deployments
- **Headers**: X-RateLimit-* headers on all responses
- **Granularity**: Per-endpoint and per-user tier limits
- **Monitoring**: Track rate limit hits, adjust limits based on metrics

---

**Skill Type**: API - Rate Limiting
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when protecting APIs from abuse and overload
**Performance**: Redis Lua scripts provide atomic operations with <1ms latency
