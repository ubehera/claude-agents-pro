---
name: performance-profiling
description: Load when user needs cProfile, line_profiler, memory_profiler, py-spy, or performance optimization and profiling patterns for Python applications
trigger_keywords: [profiling, performance, cprofile, line_profiler, memory profiler, py-spy, optimization, benchmark, bottleneck, performance tuning, cpu profiling, memory leak]
---

# Performance Profiling Skill

Production-grade Python performance analysis using cProfile, line_profiler, memory_profiler, and py-spy for CPU and memory optimization.

## Overview

Systematic performance profiling to identify bottlenecks, optimize hot paths, and eliminate memory leaks. Essential for production systems and data-intensive applications.

**When to Use**:
- Application response time exceeds SLOs
- Memory usage grows unbounded
- CPU utilization consistently high
- Before optimizing (measure first, optimize second)

## Core Concepts

### Performance Measurement Hierarchy

1. **cProfile**: Function-level CPU profiling (built-in)
2. **line_profiler**: Line-by-line CPU profiling
3. **memory_profiler**: Line-by-line memory profiling
4. **py-spy**: Sampling profiler for production (no code changes)
5. **tracemalloc**: Memory allocation tracking (built-in)

### Profiling Workflow

```yaml
1. Measure: Establish baseline metrics
2. Profile: Identify bottlenecks
3. Optimize: Target hot paths
4. Verify: Confirm improvements
5. Repeat: Iterate until SLO met
```

## CPU Profiling

### cProfile (Built-in)

**Basic Usage**:
```python
import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

# Profile from command line
# python -m cProfile -o output.prof script.py

# Profile programmatically
profiler = cProfile.Profile()
profiler.enable()

result = slow_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

**Output Analysis**:
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.050    0.050    0.100    0.100 script.py:5(slow_function)
  1000000    0.030    0.000    0.030    0.000 {built-in method builtins.sum}
```

- `ncalls`: Number of calls
- `tottime`: Total time in function (excluding subcalls)
- `cumtime`: Cumulative time (including subcalls)
- `percall`: Time per call

**Context Manager Pattern**:
```python
from contextlib import contextmanager
import cProfile
import pstats

@contextmanager
def profile_context(sort_by='cumulative', top_n=10):
    """Profile code block"""
    profiler = cProfile.Profile()
    profiler.enable()

    yield

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(sort_by)
    stats.print_stats(top_n)

# Usage
with profile_context():
    expensive_computation()
```

### line_profiler

**Installation**:
```bash
pip install line-profiler
```

**Usage**:
```python
# Add @profile decorator (kernprof adds it)
def process_data(items):
    results = []
    for item in items:
        transformed = transform(item)
        validated = validate(transformed)
        results.append(validated)
    return results

# Run with kernprof
# kernprof -l -v script.py
```

**Output**:
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     5                                           def process_data(items):
     6         1          5.0      5.0      0.1      results = []
     7      1000       450.0      0.5      9.0      for item in items:
     8      1000      2500.0      2.5     50.0          transformed = transform(item)
     9      1000      1800.0      1.8     36.0          validated = validate(transformed)
    10      1000       245.0      0.2      4.9          results.append(validated)
```

**Programmatic API**:
```python
from line_profiler import LineProfiler

profiler = LineProfiler()
profiler.add_function(process_data)
profiler.add_function(transform)

profiler.enable()
result = process_data(items)
profiler.disable()

profiler.print_stats()
```

### py-spy (Production Profiling)

**Installation**:
```bash
pip install py-spy
```

**Usage** (no code changes required):
```bash
# Profile running process
py-spy top --pid 12345

# Generate flamegraph
py-spy record -o profile.svg --pid 12345

# Profile subprocess
py-spy record -o profile.svg -- python script.py

# Dump thread stacks
py-spy dump --pid 12345
```

**Flamegraph Analysis**:
- Width: Time spent (wider = more expensive)
- Height: Call stack depth
- Color: Random (for readability)

## Memory Profiling

### memory_profiler

**Installation**:
```bash
pip install memory-profiler
```

**Usage**:
```python
from memory_profiler import profile

@profile
def memory_intensive():
    """Shows line-by-line memory usage"""
    large_list = [i for i in range(1000000)]
    large_dict = {i: i**2 for i in range(100000)}
    return len(large_list), len(large_dict)

# Run: python -m memory_profiler script.py
```

**Output**:
```
Line #    Mem usage    Increment   Line Contents
==================================================
     5     38.0 MiB     38.0 MiB   @profile
     6                             def memory_intensive():
     7     45.8 MiB      7.8 MiB       large_list = [i for i in range(1000000)]
     8     53.2 MiB      7.4 MiB       large_dict = {i: i**2 for i in range(100000)}
     9     53.2 MiB      0.0 MiB       return len(large_list), len(large_dict)
```

**Decorator with Options**:
```python
@profile(precision=4, stream=open('memory.log', 'w+'))
def analyze_memory():
    ...
```

### tracemalloc (Built-in)

**Snapshot Comparison**:
```python
import tracemalloc

# Start tracking
tracemalloc.start()

# Take snapshot before
snapshot1 = tracemalloc.take_snapshot()

# Run code
data = [i for i in range(1000000)]

# Take snapshot after
snapshot2 = tracemalloc.take_snapshot()

# Compare snapshots
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

for stat in top_stats[:10]:
    print(stat)

# Output:
# script.py:10: size=7629 KiB (+7629 KiB), count=1 (+1), average=7629 KiB
```

**Track Memory Leaks**:
```python
import tracemalloc
import time

def track_memory_leak():
    tracemalloc.start()

    # Baseline
    snapshot1 = tracemalloc.take_snapshot()

    # Run several iterations
    for _ in range(10):
        run_application()
        time.sleep(1)

    # Final snapshot
    snapshot2 = tracemalloc.take_snapshot()

    # Show memory growth
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("[ Top 10 memory growth ]")
    for stat in top_stats[:10]:
        print(stat)
```

## Advanced Patterns

### Benchmark Decorator

```python
import time
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec('P')
R = TypeVar('R')

def benchmark(func: Callable[P, R]) -> Callable[P, R]:
    """Measure function execution time"""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@benchmark
def slow_computation():
    return sum(i**2 for i in range(1000000))

# Output: slow_computation took 0.1234 seconds
```

### Statistical Benchmarking

```python
import statistics
import time
from typing import Callable

def run_benchmark(func: Callable, iterations: int = 100) -> dict:
    """Run function multiple times and collect statistics"""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times),
        "min": min(times),
        "max": max(times),
        "p95": statistics.quantiles(times, n=20)[18],  # 95th percentile
    }

# Usage
stats = run_benchmark(lambda: compute_heavy_task())
print(f"Mean: {stats['mean']:.4f}s, P95: {stats['p95']:.4f}s")
```

### Memory Context Manager

```python
import tracemalloc
from contextlib import contextmanager

@contextmanager
def memory_tracker(description: str):
    """Track memory usage of code block"""
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    yield

    snapshot_after = tracemalloc.take_snapshot()
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

    print(f"\n{description} - Top 5 memory allocations:")
    for stat in top_stats[:5]:
        print(stat)

    tracemalloc.stop()

# Usage
with memory_tracker("Data processing"):
    data = process_large_dataset()
```

## Production-Ready Examples

### Flask Request Profiling

```python
from flask import Flask, g
import cProfile
import pstats
import io

app = Flask(__name__)

@app.before_request
def before_request():
    """Start profiling for each request"""
    if app.debug:
        g.profiler = cProfile.Profile()
        g.profiler.enable()

@app.after_request
def after_request(response):
    """Print profiling stats after request"""
    if app.debug and hasattr(g, 'profiler'):
        g.profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(g.profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        print(stream.getvalue())
    return response

@app.route("/api/users")
def get_users():
    users = fetch_users_from_db()
    return jsonify(users)
```

### Async Profiling

```python
import asyncio
import time
from functools import wraps

def async_timer(func):
    """Profile async functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

@async_timer
async def fetch_data():
    await asyncio.sleep(1)
    return {"data": "value"}

# Run
asyncio.run(fetch_data())
# Output: fetch_data took 1.0012s
```

### pytest-benchmark Integration

```python
# pip install pytest-benchmark

def test_list_comprehension(benchmark):
    """Benchmark list comprehension"""
    result = benchmark(lambda: [i**2 for i in range(10000)])
    assert len(result) == 10000

def test_generator(benchmark):
    """Compare with generator"""
    result = benchmark(lambda: list(i**2 for i in range(10000)))
    assert len(result) == 10000

# Run: pytest test_performance.py --benchmark-compare
```

### Database Query Profiling

```python
from sqlalchemy import event
from sqlalchemy.engine import Engine
import logging

logging.basicConfig()
logger = logging.getLogger("sqlalchemy.engine")
logger.setLevel(logging.INFO)

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total_time = time.time() - conn.info['query_start_time'].pop()
    logger.info(f"Query took {total_time:.4f}s: {statement}")
```

## Best Practices

### 1. Profile Before Optimizing
```python
# ❌ Premature optimization
def process(items):
    return [complex_transform(x) for x in items]  # Assume this is slow

# ✅ Measure first
with profile_context():
    result = process(items)
# Then optimize the proven bottleneck
```

### 2. Use Sampling Profilers in Production
```bash
# ✅ py-spy doesn't slow down application
py-spy record -o profile.svg --pid 12345 --duration 60

# ❌ cProfile adds overhead
# Don't use in production
```

### 3. Profile Realistic Workloads
```python
# ❌ Toy data
def test_performance():
    process([1, 2, 3])

# ✅ Production-scale data
def test_performance():
    large_dataset = generate_realistic_data(size=100000)
    process(large_dataset)
```

### 4. Monitor Percentiles, Not Averages
```python
import statistics

def benchmark_percentiles(func, iterations=1000):
    times = [measure_time(func) for _ in range(iterations)]
    return {
        "p50": statistics.median(times),
        "p95": statistics.quantiles(times, n=20)[18],
        "p99": statistics.quantiles(times, n=100)[98],
    }
```

### 5. Check for Memory Leaks in Long-Running Processes
```python
import tracemalloc
import gc

def detect_leaks():
    """Run periodically in long-running processes"""
    gc.collect()
    snapshot = tracemalloc.take_snapshot()
    top = snapshot.statistics('lineno')

    print("[ Top 10 memory consumers ]")
    for stat in top[:10]:
        print(stat)
```

## Common Pitfalls

❌ **Profiling debug builds**
```python
# ❌ Debug mode adds overhead
python -O script.py  # Removes asserts
```
✅ Profile optimized builds
```python
# ✅ Production-like environment
PYTHONOPTIMIZE=1 python script.py
```

❌ **Not accounting for I/O wait**
```python
# ❌ CPU profiling won't show I/O bottlenecks
with profile_context():
    data = requests.get(url).json()  # I/O time not counted correctly
```
✅ Use wall-clock time for I/O-bound code
```python
import time
start = time.perf_counter()
data = requests.get(url).json()
print(f"Total time: {time.perf_counter() - start:.4f}s")
```

❌ **Optimizing non-bottlenecks**
```python
# ❌ Optimize function called once
@cache
def initialize_config():  # Called once at startup
    ...
```
✅ Optimize hot paths
```python
# ✅ Optimize function in tight loop
@cache
def validate_item(item):  # Called millions of times
    ...
```

❌ **Ignoring memory fragmentation**
```python
# ❌ Creating/destroying large objects repeatedly
for _ in range(1000):
    large_array = np.zeros((10000, 10000))  # Memory fragmentation
```
✅ Reuse buffers
```python
# ✅ Pre-allocate and reuse
large_array = np.zeros((10000, 10000))
for _ in range(1000):
    large_array[:] = compute_values()
```

## Optimization Techniques

### 1. Use Built-in Functions (C-optimized)
```python
# ❌ Slow (Python loop)
total = 0
for x in items:
    total += x

# ✅ Fast (C implementation)
total = sum(items)
```

### 2. List Comprehensions > For Loops
```python
# ❌ Slower
result = []
for x in items:
    result.append(x**2)

# ✅ Faster (~30%)
result = [x**2 for x in items]

# ✅✅ Fastest for large data (generator)
result = (x**2 for x in items)
```

### 3. Use NumPy for Numerical Operations
```python
# ❌ Pure Python (slow)
result = [x**2 + 2*x + 1 for x in range(1000000)]

# ✅ NumPy (100x faster)
import numpy as np
x = np.arange(1000000)
result = x**2 + 2*x + 1
```

### 4. Cache Expensive Computations
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n):
    if n <= 1:
        return n
    return expensive_computation(n-1) + expensive_computation(n-2)
```

## Quality Standards

- **Profiling Overhead**: <5% for production profiling (use py-spy)
- **Benchmark Iterations**: ≥100 for statistical significance
- **Memory Tracking**: Check for leaks in processes running >1 hour
- **Performance Targets**: Define SLOs (e.g., P95 < 200ms)
- **Documentation**: Document profiling methodology and bottlenecks found

---

**Skill Type**: Python - Performance
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when optimizing Python application performance
**Performance**: Profiling adds 0-50% overhead depending on tool (py-spy: ~0%, cProfile: ~10%, line_profiler: ~50%)
