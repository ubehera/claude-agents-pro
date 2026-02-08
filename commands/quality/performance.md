---
description: Performance analysis and optimization
args: [target] [--metric latency|throughput|memory|all] [--optimize]
tools: Task, Read, Grep
model: sonnet
---

# Performance Analysis & Optimization

Analyze performance for: **$ARGUMENTS**

## Performance Scope

### Target Identification
- **No arguments**: Analyze entire application
- **File/module**: Specific component analysis
- **Endpoint**: API performance analysis
- **Feature**: End-to-end performance

### Metrics Focus
- **latency** (default): Response time, P50/P95/P99
- **throughput**: Requests per second, concurrent users
- **memory**: Memory usage, leaks, garbage collection
- **all**: Comprehensive performance analysis

## Analysis Process

### Phase 1: Performance Profiling
Delegate to performance-optimization-specialist via Task:
- Identify performance bottlenecks
- Analyze resource utilization
- Measure current baselines
- Detect anomalies

### Phase 2: Detailed Analysis

#### Latency Analysis
- Request/response cycle timing
- Database query performance
- Network round trips
- Render/paint timing (frontend)
- Critical path identification

#### Throughput Analysis
- Concurrent request handling
- Queue processing rates
- Database connection pooling
- Rate limiting impacts
- Scaling bottlenecks

#### Memory Analysis
- Heap usage patterns
- Memory leak detection
- Object allocation rates
- Cache efficiency
- Garbage collection impact

### Phase 3: Optimization Strategy

When `--optimize` flag is present:
1. **Quick Wins** (< 1 hour)
   - Add missing indexes
   - Implement caching
   - Optimize queries
   - Enable compression

2. **Medium Improvements** (< 1 day)
   - Refactor hot paths
   - Implement pagination
   - Add connection pooling
   - Optimize bundle size

3. **Major Optimizations** (> 1 day)
   - Architecture changes
   - Database sharding
   - Service decomposition
   - Infrastructure scaling

## Performance Targets

### Web Application Standards
- **Page Load**: < 3 seconds
- **Time to Interactive**: < 5 seconds
- **API Response**: P95 < 200ms
- **Database Queries**: < 100ms
- **Core Web Vitals**: All green

### Backend Service Standards
- **API Latency**: P50 < 50ms, P95 < 200ms, P99 < 500ms
- **Throughput**: > 1000 RPS per instance
- **Error Rate**: < 0.1%
- **CPU Usage**: < 70% average
- **Memory**: Stable, no leaks

## Optimization Techniques

### Frontend Optimization
- Code splitting and lazy loading
- Image optimization and lazy loading
- Bundle size reduction
- React.memo/useMemo/useCallback
- Virtual scrolling for lists

### Backend Optimization
- Query optimization and indexing
- Caching strategy (Redis, CDN)
- Connection pooling
- Async processing
- Horizontal scaling

### Database Optimization
- Index optimization
- Query plan analysis
- Partitioning strategy
- Read replicas
- Materialized views

## Performance Report

Generate comprehensive report:
1. **Executive Summary**
   - Current performance vs targets
   - Critical issues found
   - Recommended actions

2. **Detailed Metrics**
   - Latency percentiles
   - Throughput measurements
   - Resource utilization
   - Error rates

3. **Bottleneck Analysis**
   - Top 5 slow operations
   - Resource constraints
   - Scaling limitations

4. **Optimization Plan**
   - Prioritized improvements
   - Expected impact
   - Implementation effort

5. **Monitoring Setup**
   - Key metrics to track
   - Alert thresholds
   - Dashboard configuration

## Examples

```
/performance                           # Full app analysis
/performance api/users --metric latency # API endpoint latency
/performance frontend --optimize       # Frontend with optimization
/performance database --metric all     # Comprehensive DB analysis
```

## Follow-up Actions

After analysis:
1. Implement quick wins immediately
2. Plan medium improvements sprint
3. Create backlog for major optimizations
4. Set up performance monitoring
5. Establish performance budget
6. Schedule regular performance reviews