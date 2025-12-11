---
name: async-patterns
description: Load when user needs async/await, asyncio, coroutines, event loops, async context managers, or concurrent programming patterns in Python
trigger_keywords: [async, await, asyncio, async def, coroutine, event loop, async context manager, gather, create_task, semaphore, concurrent, asynchronous]
---

# Python Async/Await Patterns

Modern Python concurrency using asyncio, async/await syntax, and production-ready patterns for high-performance I/O-bound applications.

## Core Concepts

### When to Use Async

**Use Async For**:
- I/O-bound operations (HTTP requests, database queries, file I/O)
- High-concurrency web servers (10,000+ concurrent connections)
- Real-time systems (WebSocket servers, streaming data)
- Efficient resource utilization with many waiting operations

**Don't Use Async For**:
- CPU-bound operations (use multiprocessing instead)
- Simple scripts with minimal I/O
- Legacy synchronous codebases (refactoring cost > benefit)

### The Event Loop

```python
import asyncio

# Event loop basics
async def main():
    print("Event loop is running")
    await asyncio.sleep(1)
    print("After 1 second")

# Run the event loop
asyncio.run(main())  # Python 3.7+ recommended approach
```

### Coroutines vs Regular Functions

```python
# Regular function (synchronous)
def fetch_data():
    return "data"

# Coroutine function (asynchronous)
async def fetch_data_async():
    await asyncio.sleep(0.1)  # Simulate I/O
    return "data"

# Calling patterns
result = fetch_data()  # Direct call
result = await fetch_data_async()  # Must use await inside async function
```

## Implementation Patterns

### 1. Basic Async HTTP Client

```python
import asyncio
import aiohttp
from typing import List, Dict, Any

async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Fetch a single URL"""
    async with session.get(url) as response:
        return {
            "url": url,
            "status": response.status,
            "data": await response.text()
        }

async def fetch_multiple_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# Usage
urls = ["https://api.example.com/1", "https://api.example.com/2"]
results = asyncio.run(fetch_multiple_urls(urls))
```

### 2. Async Context Managers

```python
import asyncio
from typing import Optional

class AsyncDatabaseConnection:
    """Production-ready async database connection with context manager"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection: Optional[Any] = None

    async def __aenter__(self):
        """Acquire connection"""
        print(f"Connecting to {self.connection_string}")
        await asyncio.sleep(0.1)  # Simulate connection
        self.connection = {"connected": True}
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release connection"""
        print("Closing connection")
        await asyncio.sleep(0.05)  # Simulate cleanup
        self.connection = None
        return False  # Don't suppress exceptions

    async def execute(self, query: str):
        """Execute query"""
        if not self.connection:
            raise RuntimeError("Not connected")
        await asyncio.sleep(0.1)  # Simulate query execution
        return f"Result for: {query}"

# Usage
async def main():
    async with AsyncDatabaseConnection("postgres://localhost") as db:
        result = await db.execute("SELECT * FROM users")
        print(result)
```

### 3. Task Management and Cancellation

```python
import asyncio
from typing import List

async def long_running_task(task_id: int) -> str:
    """Simulated long-running task with cancellation support"""
    try:
        for i in range(10):
            print(f"Task {task_id}: Step {i}")
            await asyncio.sleep(1)
        return f"Task {task_id} completed"
    except asyncio.CancelledError:
        print(f"Task {task_id} was cancelled")
        raise  # Re-raise to properly handle cancellation

async def run_with_timeout():
    """Run tasks with timeout and cancellation"""
    task1 = asyncio.create_task(long_running_task(1))
    task2 = asyncio.create_task(long_running_task(2))

    try:
        # Wait max 5 seconds
        results = await asyncio.wait_for(
            asyncio.gather(task1, task2),
            timeout=5.0
        )
        return results
    except asyncio.TimeoutError:
        print("Tasks timed out, cancelling...")
        task1.cancel()
        task2.cancel()
        # Wait for tasks to acknowledge cancellation
        await asyncio.gather(task1, task2, return_exceptions=True)
        raise
```

### 4. Rate Limiting with Semaphore

```python
import asyncio
from typing import List, Callable, Any

class RateLimiter:
    """Rate limiter using semaphore for concurrent request limiting"""

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(self, coro: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute coroutine with rate limiting"""
        async with self.semaphore:
            return await coro(*args, **kwargs)

async def api_request(url: str, limiter: RateLimiter) -> str:
    """Make rate-limited API request"""
    async def _request():
        print(f"Fetching {url}")
        await asyncio.sleep(0.5)  # Simulate request
        return f"Response from {url}"

    return await limiter.execute(_request)

async def main():
    limiter = RateLimiter(max_concurrent=5)  # Max 5 concurrent requests

    urls = [f"https://api.example.com/{i}" for i in range(20)]
    tasks = [api_request(url, limiter) for url in urls]

    results = await asyncio.gather(*tasks)
    return results
```

### 5. Producer-Consumer Pattern

```python
import asyncio
from typing import Optional
import random

async def producer(queue: asyncio.Queue, producer_id: int, count: int):
    """Produce items and put them in queue"""
    for i in range(count):
        item = f"Item-{producer_id}-{i}"
        await queue.put(item)
        print(f"Producer {producer_id} produced {item}")
        await asyncio.sleep(random.uniform(0.1, 0.5))

    # Signal completion
    await queue.put(None)

async def consumer(queue: asyncio.Queue, consumer_id: int):
    """Consume items from queue"""
    while True:
        item = await queue.get()

        if item is None:
            # Producer finished
            queue.task_done()
            await queue.put(None)  # Pass signal to other consumers
            break

        print(f"Consumer {consumer_id} processing {item}")
        await asyncio.sleep(random.uniform(0.2, 0.7))
        queue.task_done()

async def producer_consumer_example():
    """Run producer-consumer pattern"""
    queue = asyncio.Queue(maxsize=10)

    # Create producers and consumers
    producers = [
        asyncio.create_task(producer(queue, i, 5))
        for i in range(2)
    ]

    consumers = [
        asyncio.create_task(consumer(queue, i))
        for i in range(3)
    ]

    # Wait for producers to finish
    await asyncio.gather(*producers)

    # Wait for queue to be fully processed
    await queue.join()

    # Wait for consumers to finish
    await asyncio.gather(*consumers)
```

### 6. Async Iterator and Generator

```python
import asyncio
from typing import AsyncIterator

async def fetch_page(page: int) -> str:
    """Simulate fetching a page of data"""
    await asyncio.sleep(0.1)
    return f"Page {page} data"

async def paginated_api_fetcher(max_pages: int) -> AsyncIterator[str]:
    """Async generator for paginated data fetching"""
    for page in range(1, max_pages + 1):
        data = await fetch_page(page)
        yield data

async def process_paginated_data():
    """Process data from async generator"""
    async for page_data in paginated_api_fetcher(5):
        print(f"Processing: {page_data}")
        # Process each page as it arrives
```

### 7. Exception Handling in Async Code

```python
import asyncio
from typing import List, Union

async def failing_task(task_id: int) -> str:
    """Task that might fail"""
    await asyncio.sleep(0.1)
    if task_id % 3 == 0:
        raise ValueError(f"Task {task_id} failed")
    return f"Task {task_id} succeeded"

async def run_tasks_with_error_handling() -> List[Union[str, Exception]]:
    """Run multiple tasks and collect both successes and failures"""
    tasks = [failing_task(i) for i in range(10)]

    # gather with return_exceptions=True to capture errors
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} succeeded: {result}")

    return results

# Alternative: Handle each task individually
async def run_tasks_individually():
    """Run tasks with individual error handling"""
    tasks = [failing_task(i) for i in range(10)]
    results = []

    for task in asyncio.as_completed(tasks):
        try:
            result = await task
            results.append(result)
        except ValueError as e:
            print(f"Caught error: {e}")
            results.append(None)

    return results
```

## Production-Ready Patterns

### Complete Async API Client

```python
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class RequestConfig:
    """Configuration for API requests"""
    base_url: str
    timeout: int = 30
    max_concurrent: int = 10
    retry_count: int = 3

class AsyncAPIClient:
    """Production-ready async API client with retry, timeout, and rate limiting"""

    def __init__(self, config: RequestConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

    async def __aenter__(self):
        """Initialize session"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session"""
        if self.session:
            await self.session.close()

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        url = f"{self.config.base_url}{endpoint}"

        for attempt in range(self.config.retry_count):
            try:
                async with self.semaphore:  # Rate limiting
                    async with self.session.request(method, url, **kwargs) as response:
                        response.raise_for_status()
                        return await response.json()

            except aiohttp.ClientError as e:
                if attempt == self.config.retry_count - 1:
                    raise

                # Exponential backoff
                wait_time = 2 ** attempt
                print(f"Request failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise RuntimeError("Max retries exceeded")

    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """GET request"""
        return await self._request_with_retry("GET", endpoint, **kwargs)

    async def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """POST request"""
        return await self._request_with_retry("POST", endpoint, **kwargs)

# Usage
async def main():
    config = RequestConfig(
        base_url="https://api.example.com",
        timeout=10,
        max_concurrent=5
    )

    async with AsyncAPIClient(config) as client:
        # Concurrent requests with rate limiting
        tasks = [
            client.get(f"/users/{i}")
            for i in range(20)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]

        print(f"Successful: {len(successful)}, Failed: {len(failed)}")
```

## Best Practices

### 1. Use asyncio.run() for Top-Level Entry

```python
# ✅ Correct: Python 3.7+
async def main():
    await do_async_work()

if __name__ == "__main__":
    asyncio.run(main())

# ❌ Avoid: Manual event loop management
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
loop.close()
```

### 2. Avoid Blocking Operations in Async Code

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# ❌ BAD: Blocks event loop
async def bad_async_function():
    time.sleep(5)  # Blocks entire event loop!
    return "done"

# ✅ GOOD: Use asyncio.sleep for async delays
async def good_async_function():
    await asyncio.sleep(5)  # Non-blocking
    return "done"

# ✅ GOOD: Offload CPU-bound work to thread pool
async def cpu_bound_work():
    loop = asyncio.get_event_loop()

    def blocking_operation():
        # CPU-intensive work
        return sum(i ** 2 for i in range(10000000))

    result = await loop.run_in_executor(
        ThreadPoolExecutor(),
        blocking_operation
    )
    return result
```

### 3. Properly Handle Task Cancellation

```python
async def cancellable_task():
    """Task with proper cleanup on cancellation"""
    resource = None

    try:
        resource = await acquire_resource()
        await do_work_with_resource(resource)

    except asyncio.CancelledError:
        # Cleanup on cancellation
        if resource:
            await release_resource(resource)
        raise  # Re-raise to signal cancellation

    finally:
        # Cleanup always runs
        if resource:
            await release_resource(resource)
```

### 4. Use Structured Concurrency

```python
import asyncio
from typing import List

async def structured_concurrency_example():
    """Use task groups for structured concurrency (Python 3.11+)"""
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(async_operation_1())
        task2 = tg.create_task(async_operation_2())
        task3 = tg.create_task(async_operation_3())

    # All tasks complete or all are cancelled on exception
    # Results available as task.result()
    results = [task1.result(), task2.result(), task3.result()]
    return results
```

## Common Anti-Patterns

### ❌ Anti-Pattern 1: Mixing Sync and Async

```python
# ❌ BAD: Can't await in sync function
def sync_function():
    result = await async_function()  # SyntaxError!

# ✅ GOOD: Use asyncio.run() or make function async
async def async_function_wrapper():
    result = await async_function()
    return result
```

### ❌ Anti-Pattern 2: Forgetting to Await

```python
# ❌ BAD: Creates coroutine but doesn't execute it
async def fetch_data():
    return "data"

result = fetch_data()  # Returns coroutine, not "data"!

# ✅ GOOD: Always await coroutines
result = await fetch_data()  # Returns "data"
```

### ❌ Anti-Pattern 3: Using time.sleep() Instead of asyncio.sleep()

```python
# ❌ BAD: Blocks entire event loop
async def bad_delay():
    time.sleep(5)  # Freezes all other tasks!

# ✅ GOOD: Non-blocking sleep
async def good_delay():
    await asyncio.sleep(5)  # Other tasks continue
```

### ❌ Anti-Pattern 4: Not Using Context Managers

```python
# ❌ BAD: Manual session management
async def bad_http_request():
    session = aiohttp.ClientSession()
    response = await session.get("https://example.com")
    # Session might not close on exception!
    await session.close()

# ✅ GOOD: Automatic cleanup with context manager
async def good_http_request():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://example.com") as response:
            return await response.text()
```

## Performance Optimization

### Batch Processing

```python
import asyncio
from typing import List, TypeVar, Callable
from itertools import islice

T = TypeVar('T')

async def process_in_batches(
    items: List[T],
    processor: Callable[[T], Any],
    batch_size: int = 100
) -> List[Any]:
    """Process items in batches to control concurrency"""
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[processor(item) for item in batch])
        results.extend(batch_results)

    return results
```

## Quality Standards

- **Proper Cancellation**: All tasks handle asyncio.CancelledError correctly
- **Resource Cleanup**: Use async context managers for resource management
- **Error Handling**: Catch and handle exceptions appropriately
- **No Blocking**: Never use time.sleep() or blocking I/O in async code
- **Type Hints**: Full type annotations for async functions
- **Timeout Protection**: Set timeouts on all external I/O operations

---

**Skill Type**: Python - Concurrency
**Complexity**: Moderate
**Typical Usage**: Activated when Python agents need async/await patterns for I/O-bound operations
**Performance**: Enables 10,000+ concurrent connections vs 100-500 with threading
