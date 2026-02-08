---
name: warehouse-design-patterns
description: Design and implement modern data warehouse architectures with Snowflake, BigQuery, and Redshift. Load when building dimensional models, implementing dbt transformations, or optimizing warehouse performance.
trigger_keywords: [data warehouse, snowflake, bigquery, redshift, dimensional modeling, star schema, fact table, dimension table, dbt, data modeling, warehouse optimization, medallion architecture]
---

# Data Warehouse Design Patterns

Modern data warehouse design patterns for Snowflake, BigQuery, and Redshift with dimensional modeling and dbt.

## Core Concepts

- **Dimensional Modeling**: Structure data around business processes using fact tables (measures/metrics) and dimension tables (descriptive attributes) - facts answer "how much/many", dimensions answer "who/what/when/where"
- **Grain Definition**: Always define the grain (one row represents what?) before building fact tables - granularity determines what questions can be answered and cannot be changed without rebuilding
- **Surrogate Keys**: Use synthetic integer keys for dimensions instead of natural business keys - they handle SCD changes, improve join performance, and isolate the warehouse from source system key changes
- **Incremental Processing**: Load only changed data using watermarks, change data capture, or merge operations - full table scans are expensive and don't scale
- **Separation of Concerns**: Maintain distinct layers (bronze/silver/gold or staging/intermediate/marts) - each layer has a specific purpose, testing requirements, and access patterns

## Core Architectures

### Medallion Architecture (Bronze-Silver-Gold)

```
Bronze (Raw)  →  Silver (Cleaned)  →  Gold (Business)
   ↓                  ↓                    ↓
Landing Zone     Standardized         Analytics-Ready
Immutable        Conformed            Aggregated
Schema-less      Validated            Star/Snowflake
```

**Bronze Layer**:
- Raw data ingestion
- Immutable, append-only
- Full historical record
- Minimal transformations

**Silver Layer**:
- Cleaned and validated
- Standardized formats
- Deduplicated
- Business logic applied

**Gold Layer**:
- Business-level aggregates
- Dimensional models (star/snowflake)
- Optimized for consumption
- Denormalized for performance

## Dimensional Modeling

### Star Schema Pattern

```sql
-- Fact table (center of star)
CREATE TABLE fact_orders (
    order_id INTEGER PRIMARY KEY,
    date_key INTEGER REFERENCES dim_date(date_key),
    customer_key INTEGER REFERENCES dim_customer(customer_key),
    product_key INTEGER REFERENCES dim_product(product_key),
    location_key INTEGER REFERENCES dim_location(location_key),

    -- Measures
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    discount_amount DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    revenue DECIMAL(10,2),

    -- Metadata
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Dimension tables (points of star)
CREATE TABLE dim_customer (
    customer_key INTEGER PRIMARY KEY,
    customer_id VARCHAR(50),  -- Natural key
    customer_name VARCHAR(255),
    customer_email VARCHAR(255),
    customer_segment VARCHAR(50),
    customer_lifetime_value DECIMAL(10,2),

    -- SCD Type 2 fields
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    is_current BOOLEAN,

    -- Metadata
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE dim_product (
    product_key INTEGER PRIMARY KEY,
    product_id VARCHAR(50),
    product_name VARCHAR(255),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    unit_cost DECIMAL(10,2),
    is_active BOOLEAN
);

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE,
    day_of_week VARCHAR(10),
    day_of_month INTEGER,
    day_of_year INTEGER,
    week_of_year INTEGER,
    month INTEGER,
    month_name VARCHAR(10),
    quarter INTEGER,
    year INTEGER,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER
);

CREATE TABLE dim_location (
    location_key INTEGER PRIMARY KEY,
    location_id VARCHAR(50),
    store_name VARCHAR(255),
    street_address VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50),
    postal_code VARCHAR(20),
    latitude DECIMAL(10,6),
    longitude DECIMAL(10,6),
    region VARCHAR(50),
    territory VARCHAR(50)
);
```

### Slowly Changing Dimensions (SCD)

**SCD Type 1**: Overwrite (no history)
```sql
-- Update existing record
UPDATE dim_customer
SET customer_email = 'new_email@example.com',
    updated_at = CURRENT_TIMESTAMP
WHERE customer_id = '123';
```

**SCD Type 2**: Add new row (full history)
```sql
-- Close current record
UPDATE dim_customer
SET valid_to = CURRENT_TIMESTAMP,
    is_current = FALSE
WHERE customer_id = '123' AND is_current = TRUE;

-- Insert new record
INSERT INTO dim_customer (
    customer_id, customer_name, customer_email,
    valid_from, valid_to, is_current
)
VALUES (
    '123', 'John Doe', 'new_email@example.com',
    CURRENT_TIMESTAMP, '9999-12-31', TRUE
);
```

**SCD Type 3**: Add column (limited history)
```sql
ALTER TABLE dim_customer
ADD COLUMN previous_email VARCHAR(255);

UPDATE dim_customer
SET previous_email = customer_email,
    customer_email = 'new_email@example.com'
WHERE customer_id = '123';
```

## dbt Transformation Patterns

### Project Structure

```
dbt_project/
├── models/
│   ├── staging/           # Bronze → Silver
│   │   ├── _staging.yml
│   │   ├── stg_orders.sql
│   │   ├── stg_customers.sql
│   │   └── stg_products.sql
│   ├── intermediate/      # Silver transformations
│   │   ├── int_orders_enriched.sql
│   │   └── int_customer_metrics.sql
│   ├── marts/            # Gold layer
│   │   ├── core/
│   │   │   ├── dim_customer.sql
│   │   │   ├── dim_product.sql
│   │   │   ├── dim_date.sql
│   │   │   └── fact_orders.sql
│   │   └── marketing/
│   │       └── fct_customer_lifetime_value.sql
│   └── schema.yml
├── macros/
│   ├── generate_surrogate_key.sql
│   └── cents_to_dollars.sql
├── tests/
│   └── generic/
│       └── assert_valid_date_range.sql
└── dbt_project.yml
```

### Staging Models (Bronze → Silver)

```sql
-- models/staging/stg_orders.sql
{{
    config(
        materialized='incremental',
        unique_key='order_id',
        on_schema_change='fail'
    )
}}

WITH source AS (
    SELECT
        order_id,
        customer_id,
        order_date,
        status,
        total_amount,
        discount_amount,
        created_at,
        updated_at
    FROM {{ source('raw', 'orders') }}

    {% if is_incremental() %}
        WHERE updated_at > (SELECT MAX(updated_at) FROM {{ this }})
    {% endif %}
),

cleaned AS (
    SELECT
        order_id::INTEGER AS order_id,
        customer_id::INTEGER AS customer_id,
        order_date::DATE AS order_date,
        UPPER(TRIM(status)) AS status,

        -- Data quality fixes
        CASE
            WHEN total_amount < 0 THEN 0
            ELSE total_amount
        END AS total_amount,

        COALESCE(discount_amount, 0) AS discount_amount,

        -- Derived fields
        total_amount - COALESCE(discount_amount, 0) AS revenue,

        created_at::TIMESTAMP AS created_at,
        updated_at::TIMESTAMP AS updated_at,
        CURRENT_TIMESTAMP AS dbt_loaded_at

    FROM source
    WHERE status IN ('PENDING', 'COMPLETED', 'CANCELLED')
        AND total_amount IS NOT NULL
)

SELECT * FROM cleaned
```

### Dimension Models

```sql
-- models/marts/core/dim_customer.sql
{{
    config(
        materialized='table',
        tags=['dimension']
    )
}}

WITH source AS (
    SELECT * FROM {{ ref('stg_customers') }}
),

customer_metrics AS (
    SELECT
        customer_id,
        COUNT(*) AS total_orders,
        SUM(revenue) AS lifetime_value,
        MAX(order_date) AS last_order_date,
        MIN(order_date) AS first_order_date
    FROM {{ ref('stg_orders') }}
    GROUP BY customer_id
),

final AS (
    SELECT
        -- Surrogate key
        {{ dbt_utils.generate_surrogate_key(['s.customer_id']) }} AS customer_key,

        -- Natural key
        s.customer_id,

        -- Attributes
        s.customer_name,
        s.customer_email,
        s.customer_phone,
        s.customer_segment,

        -- Metrics
        COALESCE(m.total_orders, 0) AS total_orders,
        COALESCE(m.lifetime_value, 0) AS lifetime_value,
        m.last_order_date,
        m.first_order_date,

        -- SCD Type 2
        s.valid_from,
        s.valid_to,
        s.is_current,

        -- Metadata
        CURRENT_TIMESTAMP AS dbt_loaded_at

    FROM source s
    LEFT JOIN customer_metrics m ON s.customer_id = m.customer_id
    WHERE s.is_current = TRUE
)

SELECT * FROM final
```

### Fact Models

```sql
-- models/marts/core/fact_orders.sql
{{
    config(
        materialized='incremental',
        unique_key='order_key',
        partition_by={
            'field': 'order_date',
            'data_type': 'date',
            'granularity': 'day'
        },
        cluster_by=['customer_key', 'product_key']
    )
}}

WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
    {% if is_incremental() %}
        WHERE order_date >= (SELECT MAX(order_date) FROM {{ this }})
    {% endif %}
),

customers AS (
    SELECT * FROM {{ ref('dim_customer') }}
),

products AS (
    SELECT * FROM {{ ref('dim_product') }}
),

dates AS (
    SELECT * FROM {{ ref('dim_date') }}
),

final AS (
    SELECT
        -- Surrogate key
        {{ dbt_utils.generate_surrogate_key(['o.order_id']) }} AS order_key,

        -- Foreign keys to dimensions
        c.customer_key,
        p.product_key,
        d.date_key,

        -- Degenerate dimension
        o.order_id,

        -- Measures
        o.quantity,
        o.unit_price,
        o.discount_amount,
        o.total_amount,
        o.revenue,

        -- Dates
        o.order_date,

        -- Metadata
        CURRENT_TIMESTAMP AS dbt_loaded_at

    FROM orders o
    LEFT JOIN customers c ON o.customer_id = c.customer_id
    LEFT JOIN products p ON o.product_id = p.product_id
    LEFT JOIN dates d ON o.order_date = d.full_date
)

SELECT * FROM final
```

### Custom Macros

```sql
-- macros/cents_to_dollars.sql
{% macro cents_to_dollars(column_name, precision=2) %}
    ROUND({{ column_name }} / 100.0, {{ precision }})
{% endmacro %}

-- Usage in model
SELECT {{ cents_to_dollars('amount_cents') }} AS amount_dollars
```

### Custom Tests

```sql
-- tests/generic/assert_valid_date_range.sql
{% test assert_valid_date_range(model, column_name) %}

WITH validation AS (
    SELECT
        {{ column_name }} AS date_value
    FROM {{ model }}
    WHERE {{ column_name }} < '2020-01-01'
        OR {{ column_name }} > CURRENT_DATE + INTERVAL '1 year'
)

SELECT * FROM validation

{% endtest %}
```

## Snowflake Optimization

### Clustering Keys

```sql
-- Create table with cluster key
CREATE TABLE fact_orders (
    order_id INTEGER,
    order_date DATE,
    customer_id INTEGER,
    total_amount DECIMAL(10,2)
)
CLUSTER BY (order_date, customer_id);

-- Add clustering to existing table
ALTER TABLE fact_orders CLUSTER BY (order_date, customer_id);

-- Check clustering effectiveness
SELECT SYSTEM$CLUSTERING_INFORMATION('fact_orders');
```

### Materialized Views

```sql
-- Create materialized view for aggregations
CREATE MATERIALIZED VIEW mv_daily_sales AS
SELECT
    order_date,
    customer_id,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_sales,
    AVG(total_amount) AS avg_order_value
FROM fact_orders
GROUP BY order_date, customer_id;

-- Refresh materialized view
ALTER MATERIALIZED VIEW mv_daily_sales REFRESH;
```

### Time Travel and Zero-Copy Cloning

```sql
-- Query historical data (within retention period)
SELECT *
FROM fact_orders AT(OFFSET => -3600);  -- 1 hour ago

SELECT *
FROM fact_orders BEFORE(STATEMENT => '<query_id>');

-- Zero-copy clone for testing
CREATE TABLE fact_orders_dev CLONE fact_orders;

-- Restore from Time Travel
CREATE OR REPLACE TABLE fact_orders
CLONE fact_orders AT(OFFSET => -7200);  -- Restore to 2 hours ago
```

## BigQuery Optimization

### Partitioning and Clustering

```sql
-- Create partitioned and clustered table
CREATE TABLE project.dataset.fact_orders
PARTITION BY DATE(order_date)
CLUSTER BY customer_id, product_id
AS
SELECT
    order_id,
    customer_id,
    product_id,
    order_date,
    total_amount
FROM source_table;

-- Query uses partition pruning
SELECT *
FROM project.dataset.fact_orders
WHERE order_date BETWEEN '2025-01-01' AND '2025-01-31'
    AND customer_id = 12345;  -- Uses clustering
```

### Nested and Repeated Fields

```sql
-- Use STRUCT for related fields
CREATE TABLE project.dataset.orders AS
SELECT
    order_id,
    STRUCT(
        customer_id AS id,
        customer_name AS name,
        customer_email AS email
    ) AS customer,
    ARRAY_AGG(STRUCT(
        product_id AS id,
        product_name AS name,
        quantity,
        price
    )) AS line_items
FROM source
GROUP BY order_id, customer_id, customer_name, customer_email;

-- Query nested fields
SELECT
    order_id,
    customer.name,
    line_item.product_name,
    line_item.quantity
FROM project.dataset.orders,
UNNEST(line_items) AS line_item;
```

### Table Snapshots

```sql
-- Create snapshot
CREATE SNAPSHOT TABLE project.dataset.orders_snapshot
CLONE project.dataset.orders
OPTIONS(expiration_timestamp=TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 7 DAY));
```

## Redshift Optimization

### Distribution and Sort Keys

```sql
-- Distribution styles
CREATE TABLE fact_orders (
    order_id INTEGER,
    customer_id INTEGER,
    order_date DATE,
    total_amount DECIMAL(10,2)
)
DISTKEY(customer_id)  -- Distribute by customer for joins
SORTKEY(order_date);  -- Sort by date for range queries

-- Alternative: EVEN distribution
CREATE TABLE dim_product (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(255)
)
DISTSTYLE EVEN;

-- Alternative: ALL distribution (broadcast small tables)
CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE
)
DISTSTYLE ALL;
```

### Column Encoding

```sql
-- Analyze and apply optimal encoding
ANALYZE COMPRESSION fact_orders;

-- Apply encoding
CREATE TABLE fact_orders (
    order_id INTEGER ENCODE az64,
    customer_id INTEGER ENCODE az64,
    order_date DATE ENCODE az64,
    total_amount DECIMAL(10,2) ENCODE az64
);
```

### Vacuum and Analyze

```sql
-- Reclaim space and sort rows
VACUUM fact_orders;

-- Update table statistics
ANALYZE fact_orders;

-- Deep copy (recreate table)
CREATE TABLE fact_orders_new AS SELECT * FROM fact_orders;
DROP TABLE fact_orders;
ALTER TABLE fact_orders_new RENAME TO fact_orders;
```

## Performance Best Practices

### Query Optimization

```sql
-- ❌ BAD: SELECT *
SELECT * FROM fact_orders;

-- ✅ GOOD: Select only needed columns
SELECT order_id, customer_id, total_amount FROM fact_orders;

-- ❌ BAD: Filtering in application
SELECT * FROM fact_orders;
-- Filter in application code

-- ✅ GOOD: Filtering in query
SELECT * FROM fact_orders WHERE order_date >= '2025-01-01';

-- ❌ BAD: Subquery in SELECT
SELECT
    o.order_id,
    (SELECT SUM(total_amount) FROM fact_orders WHERE customer_id = o.customer_id) AS customer_total
FROM fact_orders o;

-- ✅ GOOD: JOIN
SELECT
    o.order_id,
    c.customer_total
FROM fact_orders o
LEFT JOIN (
    SELECT customer_id, SUM(total_amount) AS customer_total
    FROM fact_orders
    GROUP BY customer_id
) c ON o.customer_id = c.customer_id;
```

### Incremental Loading Pattern

```sql
-- Create staging table
CREATE TABLE staging_orders AS
SELECT * FROM source_orders
WHERE order_date = CURRENT_DATE;

-- Merge into target
MERGE INTO fact_orders AS target
USING staging_orders AS source
ON target.order_id = source.order_id
WHEN MATCHED AND source.updated_at > target.updated_at THEN
    UPDATE SET
        total_amount = source.total_amount,
        status = source.status,
        updated_at = source.updated_at
WHEN NOT MATCHED THEN
    INSERT (order_id, customer_id, total_amount, status, created_at, updated_at)
    VALUES (source.order_id, source.customer_id, source.total_amount,
            source.status, source.created_at, source.updated_at);
```

## Quality Standards

- **Model Documentation**: All models documented in schema.yml
- **Testing**: Data quality tests for all dimensions and facts
- **Partitioning**: Large tables partitioned by date
- **Incremental Loads**: Incremental models for facts
- **Performance**: Query response time <5 seconds for dashboards

---

**Skill Type**: Data Engineering - Data Warehousing
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated for warehouse design, dimensional modeling, dbt transformations
**Tools**: Snowflake, BigQuery, Redshift, dbt 1.0+
