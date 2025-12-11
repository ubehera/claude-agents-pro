---
name: query-optimization
description: Load when user needs SQL query optimization, indexing strategies, execution plans, N+1 queries, database performance tuning, or PostgreSQL/MySQL best practices
trigger_keywords: [query optimization, sql optimization, index, execution plan, explain, n+1 query, database performance, slow query, query tuning, postgresql, mysql, covering index, composite index]
---

# SQL Query Optimization

Production-grade SQL query optimization strategies, indexing patterns, and performance tuning for PostgreSQL and MySQL databases.

## Core Concepts

### Query Performance Fundamentals

**Query Execution Phases**:
1. **Parsing**: Syntax validation
2. **Planning**: Query optimizer creates execution plan
3. **Execution**: Database engine executes plan
4. **Result Return**: Data returned to client

**Performance Metrics**:
- **Execution Time**: Total query duration
- **Rows Scanned**: Total rows examined
- **Rows Returned**: Actual result set size
- **Index Usage**: Whether indexes are utilized
- **Join Strategy**: Nested loop, hash join, merge join

### EXPLAIN Command

```sql
-- PostgreSQL
EXPLAIN ANALYZE
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.status = 'active'
GROUP BY u.id, u.name;

-- Output shows:
-- - Seq Scan vs Index Scan
-- - Join methods
-- - Estimated vs actual rows
-- - Execution time
```

```sql
-- MySQL
EXPLAIN FORMAT=JSON
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.status = 'active'
GROUP BY u.id, u.name;
```

## Indexing Strategies

### 1. Single-Column Indexes

```sql
-- Create index on frequently queried columns
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);

-- ✅ GOOD: Index used
SELECT * FROM users WHERE email = 'alice@example.com';

-- ❌ BAD: Index not used (leading wildcard)
SELECT * FROM users WHERE email LIKE '%example.com';

-- ✅ GOOD: Index used (trailing wildcard)
SELECT * FROM users WHERE email LIKE 'alice%';
```

### 2. Composite (Multi-Column) Indexes

```sql
-- Order matters! Left-to-right prefix rule
CREATE INDEX idx_orders_user_status_created
ON orders(user_id, status, created_at);

-- ✅ Uses index (matches prefix)
SELECT * FROM orders WHERE user_id = 123;
SELECT * FROM orders WHERE user_id = 123 AND status = 'completed';
SELECT * FROM orders WHERE user_id = 123 AND status = 'completed' AND created_at > '2025-01-01';

-- ❌ Doesn't use index (no prefix match)
SELECT * FROM orders WHERE status = 'completed';
SELECT * FROM orders WHERE created_at > '2025-01-01';

-- ❌ Doesn't use index (skips middle column)
SELECT * FROM orders WHERE user_id = 123 AND created_at > '2025-01-01';
```

**Index Column Ordering Rule**:
1. Equality columns first (`=`)
2. Range columns next (`>`, `<`, `BETWEEN`)
3. Sort columns last (`ORDER BY`)

```sql
-- Optimized for: WHERE user_id = ? AND created_at > ? ORDER BY created_at DESC
CREATE INDEX idx_orders_optimal ON orders(user_id, created_at DESC);
```

### 3. Covering Indexes

```sql
-- Query needs: user_id, status, total_amount
SELECT user_id, status, total_amount
FROM orders
WHERE user_id = 123 AND status = 'completed';

-- ✅ Covering index includes all needed columns
CREATE INDEX idx_orders_covering
ON orders(user_id, status)
INCLUDE (total_amount);  -- PostgreSQL 11+

-- MySQL equivalent
CREATE INDEX idx_orders_covering
ON orders(user_id, status, total_amount);

-- Database can satisfy query entirely from index without touching table
```

### 4. Partial Indexes (PostgreSQL)

```sql
-- Index only active users (reduces index size)
CREATE INDEX idx_users_active_email
ON users(email)
WHERE status = 'active';

-- Index only recent orders
CREATE INDEX idx_orders_recent
ON orders(created_at)
WHERE created_at > '2025-01-01';

-- Smaller index = faster queries, less storage
```

### 5. Full-Text Search Indexes

```sql
-- PostgreSQL
CREATE INDEX idx_products_name_gin
ON products USING gin(to_tsvector('english', name));

SELECT * FROM products
WHERE to_tsvector('english', name) @@ to_tsquery('laptop');

-- MySQL
CREATE FULLTEXT INDEX idx_products_name_fulltext
ON products(name, description);

SELECT * FROM products
WHERE MATCH(name, description) AGAINST('laptop' IN NATURAL LANGUAGE MODE);
```

## Query Optimization Patterns

### 1. The N+1 Query Problem

**❌ BAD: N+1 queries**

```typescript
// Fetches users (1 query)
const users = await db.query('SELECT * FROM users LIMIT 10');

// Fetches orders for each user (N queries)
for (const user of users) {
  const orders = await db.query(
    'SELECT * FROM orders WHERE user_id = $1',
    [user.id]
  );
  user.orders = orders;
}
// Total: 11 queries for 10 users
```

**✅ GOOD: Single query with JOIN**

```typescript
const query = `
  SELECT
    u.id, u.name, u.email,
    o.id as order_id, o.total_amount, o.status
  FROM users u
  LEFT JOIN orders o ON o.user_id = u.id
  LIMIT 10
`;

const rows = await db.query(query);

// Transform flat result into nested structure
const users = rows.reduce((acc, row) => {
  let user = acc.find(u => u.id === row.id);
  if (!user) {
    user = { id: row.id, name: row.name, email: row.email, orders: [] };
    acc.push(user);
  }
  if (row.order_id) {
    user.orders.push({
      id: row.order_id,
      total_amount: row.total_amount,
      status: row.status
    });
  }
  return acc;
}, []);
```

**✅ ALTERNATIVE: Batch loading**

```typescript
// Fetch users
const users = await db.query('SELECT * FROM users LIMIT 10');
const userIds = users.map(u => u.id);

// Fetch all orders in single query
const orders = await db.query(
  'SELECT * FROM orders WHERE user_id = ANY($1)',
  [userIds]
);

// Group orders by user_id
const ordersByUser = orders.reduce((acc, order) => {
  if (!acc[order.user_id]) acc[order.user_id] = [];
  acc[order.user_id].push(order);
  return acc;
}, {});

// Attach orders to users
users.forEach(user => {
  user.orders = ordersByUser[user.id] || [];
});
// Total: 2 queries
```

### 2. Pagination Performance

**❌ BAD: OFFSET for large offsets**

```sql
-- Slow for large offsets (database must scan all skipped rows)
SELECT * FROM orders
ORDER BY created_at DESC
OFFSET 100000 LIMIT 20;  -- Scans 100,020 rows!
```

**✅ GOOD: Cursor-based pagination**

```sql
-- First page
SELECT * FROM orders
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- Subsequent pages (use last item's values as cursor)
SELECT * FROM orders
WHERE (created_at, id) < ('2025-12-01 10:30:00', 12345)
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- Index needed
CREATE INDEX idx_orders_cursor ON orders(created_at DESC, id DESC);
```

### 3. Aggregation Optimization

**❌ BAD: Multiple aggregation queries**

```typescript
const totalUsers = await db.query('SELECT COUNT(*) FROM users');
const activeUsers = await db.query('SELECT COUNT(*) FROM users WHERE status = "active"');
const premiumUsers = await db.query('SELECT COUNT(*) FROM users WHERE plan = "premium"');
// 3 queries
```

**✅ GOOD: Single query with conditional aggregation**

```sql
SELECT
  COUNT(*) as total_users,
  COUNT(*) FILTER (WHERE status = 'active') as active_users,
  COUNT(*) FILTER (WHERE plan = 'premium') as premium_users
FROM users;
-- 1 query
```

### 4. Subquery vs JOIN

**❌ SLOW: Correlated subquery**

```sql
-- Executes subquery for EACH user
SELECT u.name, (
  SELECT COUNT(*)
  FROM orders o
  WHERE o.user_id = u.id
) as order_count
FROM users u;
```

**✅ FAST: JOIN with GROUP BY**

```sql
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.name;
```

### 5. EXISTS vs IN

```sql
-- ✅ GOOD: EXISTS (stops at first match)
SELECT * FROM users u
WHERE EXISTS (
  SELECT 1 FROM orders o
  WHERE o.user_id = u.id AND o.status = 'completed'
);

-- ❌ SLOWER: IN with large subquery
SELECT * FROM users u
WHERE u.id IN (
  SELECT user_id FROM orders WHERE status = 'completed'
);

-- ✅ GOOD: IN with small list
SELECT * FROM users
WHERE id IN (1, 2, 3, 4, 5);
```

## Advanced Optimization Techniques

### 1. Index-Only Scans

```sql
-- Query uses only indexed columns
CREATE INDEX idx_orders_user_status ON orders(user_id, status);

-- ✅ Index-only scan (no table access needed)
SELECT user_id, status FROM orders WHERE user_id = 123;

-- ❌ Index scan + table access (fetches total_amount from table)
SELECT user_id, status, total_amount FROM orders WHERE user_id = 123;
```

### 2. Avoiding SELECT *

```sql
-- ❌ BAD: Fetches unnecessary data
SELECT * FROM users WHERE id = 123;

-- ✅ GOOD: Only fetch needed columns
SELECT id, name, email FROM users WHERE id = 123;

-- Benefits:
-- - Less data transfer
-- - Enables covering indexes
-- - Reduces memory usage
-- - Faster serialization
```

### 3. Batch Operations

```sql
-- ❌ BAD: Multiple inserts
INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');
INSERT INTO users (name, email) VALUES ('Bob', 'bob@example.com');
-- ... 1000 times = 1000 round trips

-- ✅ GOOD: Batch insert
INSERT INTO users (name, email) VALUES
  ('Alice', 'alice@example.com'),
  ('Bob', 'bob@example.com'),
  ('Charlie', 'charlie@example.com');
-- 1 round trip
```

### 4. Using CTEs for Readability

```sql
-- Complex query split into logical parts
WITH active_users AS (
  SELECT id, name FROM users WHERE status = 'active'
),
recent_orders AS (
  SELECT user_id, COUNT(*) as order_count, SUM(total_amount) as total_spent
  FROM orders
  WHERE created_at > NOW() - INTERVAL '30 days'
  GROUP BY user_id
)
SELECT
  u.name,
  COALESCE(ro.order_count, 0) as order_count,
  COALESCE(ro.total_spent, 0) as total_spent
FROM active_users u
LEFT JOIN recent_orders ro ON ro.user_id = u.id
ORDER BY total_spent DESC
LIMIT 10;
```

### 5. Materialized Views (PostgreSQL)

```sql
-- Expensive aggregation query
CREATE MATERIALIZED VIEW mv_user_order_stats AS
SELECT
  u.id,
  u.name,
  COUNT(o.id) as total_orders,
  SUM(o.total_amount) as total_spent,
  MAX(o.created_at) as last_order_date
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.name;

-- Create index on materialized view
CREATE INDEX idx_mv_user_order_stats_total_spent
ON mv_user_order_stats(total_spent DESC);

-- Refresh periodically (e.g., daily cron job)
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_order_stats;

-- Query materialized view (fast!)
SELECT * FROM mv_user_order_stats
WHERE total_spent > 1000
ORDER BY total_spent DESC;
```

## PostgreSQL-Specific Optimizations

### 1. Analyze and Vacuum

```sql
-- Update query planner statistics
ANALYZE users;
ANALYZE orders;

-- Reclaim space and update statistics
VACUUM ANALYZE users;

-- Auto-vacuum settings (postgresql.conf)
autovacuum = on
autovacuum_naptime = 1min
autovacuum_vacuum_scale_factor = 0.1
```

### 2. Connection Pooling

```typescript
import { Pool } from 'pg';

// ✅ GOOD: Connection pool
const pool = new Pool({
  host: 'localhost',
  port: 5432,
  database: 'myapp',
  user: 'dbuser',
  password: process.env.DB_PASSWORD,
  max: 20,              // Max connections
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

// Reuses connections
const result = await pool.query('SELECT * FROM users WHERE id = $1', [123]);
```

### 3. Prepared Statements

```typescript
// ✅ GOOD: Prepared statement (plan cached)
const query = {
  name: 'fetch-user',
  text: 'SELECT * FROM users WHERE email = $1',
  values: ['alice@example.com']
};

const result = await pool.query(query);
```

## MySQL-Specific Optimizations

### 1. Query Cache (MySQL 5.7 and earlier)

```sql
-- Check query cache
SHOW VARIABLES LIKE 'query_cache%';

-- Enable query cache (my.cnf)
query_cache_type = 1
query_cache_size = 256M
query_cache_limit = 2M
```

### 2. InnoDB Buffer Pool

```sql
-- Critical for performance (my.cnf)
innodb_buffer_pool_size = 16G  # 70-80% of available RAM
innodb_buffer_pool_instances = 8
```

### 3. Index Hints

```sql
-- Force index usage (when optimizer chooses wrong index)
SELECT * FROM orders
USE INDEX (idx_orders_user_status)
WHERE user_id = 123 AND status = 'completed';
```

## Monitoring and Diagnostics

### PostgreSQL Slow Query Log

```sql
-- postgresql.conf
log_min_duration_statement = 1000  # Log queries > 1 second
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_statement = 'none'
log_duration = off

-- Query slow queries
SELECT
  query,
  calls,
  total_time / 1000 as total_seconds,
  mean_time / 1000 as avg_seconds,
  max_time / 1000 as max_seconds
FROM pg_stat_statements
WHERE mean_time > 1000  -- > 1 second
ORDER BY total_time DESC
LIMIT 20;
```

### MySQL Slow Query Log

```sql
-- Enable slow query log (my.cnf)
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow-query.log
long_query_time = 1
log_queries_not_using_indexes = 1

-- Analyze with mysqldumpslow
-- $ mysqldumpslow -s t -t 10 /var/log/mysql/slow-query.log
```

## Best Practices

### 1. Index Maintenance

```sql
-- Check index usage (PostgreSQL)
SELECT
  schemaname,
  tablename,
  indexname,
  idx_scan,
  idx_tup_read,
  idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0  -- Unused indexes
ORDER BY pg_relation_size(indexrelid) DESC;

-- Drop unused indexes
DROP INDEX idx_unused_index;
```

### 2. Query Pattern Checklist

- ✅ Use indexes on WHERE, JOIN, ORDER BY columns
- ✅ Avoid SELECT * (fetch only needed columns)
- ✅ Use LIMIT for large result sets
- ✅ Batch operations instead of loops
- ✅ Use connection pooling
- ✅ Use prepared statements for repeated queries
- ✅ Monitor slow queries
- ✅ Run ANALYZE regularly

### 3. When NOT to Index

- ❌ Small tables (<1000 rows)
- ❌ Columns with low cardinality (e.g., boolean)
- ❌ Columns rarely used in queries
- ❌ Tables with frequent writes (index overhead)

## Common Anti-Patterns

### ❌ Anti-Pattern 1: Function in WHERE Clause

```sql
-- ❌ BAD: Index not used
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';

-- ✅ GOOD: Store lowercase in database or use functional index
CREATE INDEX idx_users_email_lower ON users(LOWER(email));
```

### ❌ Anti-Pattern 2: OR Conditions

```sql
-- ❌ BAD: Index not used efficiently
SELECT * FROM orders WHERE user_id = 123 OR status = 'pending';

-- ✅ GOOD: Use UNION
SELECT * FROM orders WHERE user_id = 123
UNION
SELECT * FROM orders WHERE status = 'pending';
```

### ❌ Anti-Pattern 3: Implicit Type Conversion

```sql
-- ❌ BAD: user_id is integer, '123' is string (index not used)
SELECT * FROM orders WHERE user_id = '123';

-- ✅ GOOD: Correct type
SELECT * FROM orders WHERE user_id = 123;
```

### ❌ Anti-Pattern 4: SELECT DISTINCT to Fix Duplicates

```sql
-- ❌ BAD: DISTINCT hides join problem
SELECT DISTINCT u.name FROM users u
JOIN orders o ON o.user_id = u.id;

-- ✅ GOOD: Use proper aggregation
SELECT u.name FROM users u
JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.name;
```

## Performance Testing

```typescript
import { performance } from 'perf_hooks';

async function benchmarkQuery(query: string, params: any[]) {
  const iterations = 100;
  const times: number[] = [];

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await pool.query(query, params);
    const end = performance.now();
    times.push(end - start);
  }

  const avg = times.reduce((a, b) => a + b) / times.length;
  const min = Math.min(...times);
  const max = Math.max(...times);

  console.log(`Query Performance (${iterations} iterations):`);
  console.log(`  Average: ${avg.toFixed(2)}ms`);
  console.log(`  Min: ${min.toFixed(2)}ms`);
  console.log(`  Max: ${max.toFixed(2)}ms`);
}
```

## Quality Standards

- **Index Strategy**: Covering indexes for critical queries
- **Query Patterns**: Eliminate N+1, use batch operations
- **Monitoring**: Slow query logging, execution plan analysis
- **Testing**: Benchmark queries with production-sized datasets
- **Documentation**: Document index rationale and query patterns
- **Maintenance**: Regular ANALYZE/VACUUM, index usage audits

---

**Skill Type**: Database - Performance
**Complexity**: Moderate
**Typical Usage**: Activated when database architects optimize queries or design indexing strategies
**Databases**: PostgreSQL 12+, MySQL 8+
