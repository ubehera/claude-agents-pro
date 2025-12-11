---
name: pandas-polars-patterns
description: Advanced DataFrame manipulation patterns using pandas and polars for high-performance data processing. Load when performing complex data transformations, aggregations, time series analysis, or optimizing DataFrame operations.
trigger_keywords: [pandas, polars, dataframe, data manipulation, aggregation, groupby, merge, join, time series, window functions, vectorization, performance optimization]
---

# Pandas and Polars Patterns

High-performance data manipulation patterns using pandas and polars for production data pipelines.

## Performance Comparison

### When to Use Pandas
- Interactive data analysis
- Mature ecosystem (scikit-learn, statsmodels)
- Complex time series operations
- In-memory datasets (<10GB)

### When to Use Polars
- Large datasets (>10GB)
- Performance-critical pipelines
- Lazy evaluation for query optimization
- Parallelized operations
- Memory efficiency

## Pandas Advanced Patterns

### Memory Optimization

```python
import pandas as pd
import numpy as np

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory usage"""

    # Downcast numeric columns
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # Convert object columns to category if cardinality is low
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df)

        if num_unique / num_total < 0.5:  # <50% unique values
            df[col] = df[col].astype('category')

    return df

# Before optimization
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# After optimization
df_optimized = optimize_dataframe(df)
print(f"Memory usage: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### Chunked Processing

```python
def process_large_csv_in_chunks(file_path: str, chunk_size: int = 10000):
    """Process large CSV file in chunks"""

    results = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        processed = chunk[chunk['amount'] > 0].copy()
        processed['revenue'] = processed['amount'] - processed['discount']

        results.append(processed)

    # Combine results
    final_df = pd.concat(results, ignore_index=True)

    return final_df
```

### Vectorized Operations

```python
# ❌ BAD: Loop (slow)
result = []
for idx, row in df.iterrows():
    result.append(row['price'] * row['quantity'])
df['total'] = result

# ✅ GOOD: Vectorized (100x faster)
df['total'] = df['price'] * df['quantity']

# ❌ BAD: Apply with lambda
df['total'] = df.apply(lambda row: row['price'] * row['quantity'], axis=1)

# ✅ GOOD: NumPy operations
df['total'] = np.where(
    df['quantity'] > 10,
    df['price'] * df['quantity'] * 0.9,  # 10% discount
    df['price'] * df['quantity']
)
```

### Advanced GroupBy

```python
# Multiple aggregations
customer_metrics = df.groupby('customer_id').agg({
    'order_id': 'count',
    'total_amount': ['sum', 'mean', 'median'],
    'order_date': ['min', 'max'],
    'status': lambda x: (x == 'completed').sum()  # Count completed
})

# Flatten multi-level columns
customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]

# Named aggregations (pandas 0.25+)
customer_metrics = df.groupby('customer_id').agg(
    order_count=('order_id', 'count'),
    total_revenue=('total_amount', 'sum'),
    avg_order_value=('total_amount', 'mean'),
    first_order_date=('order_date', 'min'),
    last_order_date=('order_date', 'max'),
    completed_orders=('status', lambda x: (x == 'completed').sum())
).reset_index()

# Transform (keep original shape)
df['customer_total'] = df.groupby('customer_id')['total_amount'].transform('sum')
df['customer_avg'] = df.groupby('customer_id')['total_amount'].transform('mean')

# Filter groups
# Keep only customers with >5 orders
active_customers = df.groupby('customer_id').filter(lambda x: len(x) > 5)

# Cumulative operations
df['cumulative_revenue'] = df.groupby('customer_id')['total_amount'].cumsum()
df['order_rank'] = df.groupby('customer_id')['order_date'].rank(method='dense')
```

### Window Functions

```python
# Rolling window
df['7_day_avg'] = df.groupby('product_id')['sales'].rolling(window=7).mean().reset_index(0, drop=True)

# Expanding window
df['cumulative_sales'] = df.groupby('product_id')['sales'].expanding().sum().reset_index(0, drop=True)

# Lead and lag
df['previous_sale'] = df.groupby('product_id')['sales'].shift(1)
df['next_sale'] = df.groupby('product_id')['sales'].shift(-1)
df['sales_change'] = df['sales'] - df['previous_sale']

# Percent change
df['pct_change'] = df.groupby('product_id')['sales'].pct_change()
```

### Efficient Joins

```python
# Use merge with indicator
merged = pd.merge(
    df_left, df_right,
    on='customer_id',
    how='left',
    indicator=True
)

# Check unmatched records
unmatched = merged[merged['_merge'] == 'left_only']

# Merge with validation
merged = pd.merge(
    df_left, df_right,
    on='customer_id',
    how='inner',
    validate='one_to_many'  # Validate relationship
)

# Merge on index
merged = df_left.merge(df_right, left_index=True, right_index=True)

# Multiple key merge with suffix
merged = pd.merge(
    df_left, df_right,
    on=['customer_id', 'order_date'],
    how='outer',
    suffixes=('_left', '_right')
)
```

### Time Series Operations

```python
# Set datetime index
df['order_date'] = pd.to_datetime(df['order_date'])
df = df.set_index('order_date')

# Resample to daily aggregates
daily_sales = df.resample('D').agg({
    'order_id': 'count',
    'total_amount': 'sum'
})

# Resample with multiple rules
weekly_sales = df.resample('W').agg({
    'order_id': 'count',
    'total_amount': ['sum', 'mean'],
    'customer_id': 'nunique'
})

# Forward fill missing dates
daily_sales = daily_sales.asfreq('D', fill_value=0)

# Rolling statistics
df['7_day_ma'] = df['sales'].rolling(window=7).mean()
df['7_day_std'] = df['sales'].rolling(window=7).std()

# Expanding statistics
df['cumulative_avg'] = df['sales'].expanding().mean()

# Time-based indexing
jan_2025 = df['2025-01']
q1_2025 = df['2025-Q1']

# Date arithmetic
df['days_since_first_order'] = (df.index - df.index.min()).days
```

### Pivot Tables

```python
# Basic pivot
pivot = df.pivot_table(
    index='product_id',
    columns='month',
    values='sales',
    aggfunc='sum',
    fill_value=0
)

# Multiple values
pivot = df.pivot_table(
    index='product_id',
    columns='month',
    values=['sales', 'quantity'],
    aggfunc={'sales': 'sum', 'quantity': 'mean'}
)

# Multi-level indexing
pivot = df.pivot_table(
    index=['category', 'product_id'],
    columns=['year', 'quarter'],
    values='sales',
    aggfunc='sum',
    margins=True,  # Add totals
    margins_name='Total'
)

# Unpivot (melt)
melted = pivot.reset_index().melt(
    id_vars=['category', 'product_id'],
    var_name='period',
    value_name='sales'
)
```

## Polars Patterns

### Lazy Evaluation

```python
import polars as pl

# Lazy DataFrame (query optimization)
lazy_df = pl.scan_csv('large_file.csv')

# Chain operations without execution
result = (
    lazy_df
    .filter(pl.col('amount') > 100)
    .groupby('customer_id')
    .agg([
        pl.col('order_id').count().alias('order_count'),
        pl.col('amount').sum().alias('total_amount')
    ])
    .sort('total_amount', descending=True)
)

# Show optimized query plan
print(result.explain())

# Execute query
materialized = result.collect()
```

### High-Performance Aggregations

```python
# Eager execution
df = pl.read_csv('data.csv')

# GroupBy aggregations
customer_stats = df.groupby('customer_id').agg([
    pl.col('order_id').count().alias('order_count'),
    pl.col('total_amount').sum().alias('total_revenue'),
    pl.col('total_amount').mean().alias('avg_order_value'),
    pl.col('total_amount').quantile(0.5).alias('median_order_value'),
    pl.col('order_date').min().alias('first_order_date'),
    pl.col('order_date').max().alias('last_order_date')
])

# Window functions
df_with_windows = df.select([
    pl.all(),
    pl.col('sales').sum().over('product_id').alias('product_total_sales'),
    pl.col('sales').mean().over('product_id').alias('product_avg_sales'),
    pl.col('sales').rank().over('product_id').alias('sales_rank')
])
```

### Expressions and Transformations

```python
# Complex expressions
result = df.select([
    pl.col('customer_id'),
    pl.col('order_id'),

    # Conditional logic
    pl.when(pl.col('amount') > 1000)
        .then(pl.col('amount') * 0.9)
        .otherwise(pl.col('amount'))
        .alias('discounted_amount'),

    # String operations
    pl.col('email').str.to_lowercase().alias('email_lower'),
    pl.col('customer_name').str.split(' ').alias('name_parts'),

    # Date operations
    pl.col('order_date').dt.year().alias('order_year'),
    pl.col('order_date').dt.quarter().alias('order_quarter'),
    pl.col('order_date').dt.day().alias('order_day')
])

# List operations
df_with_lists = df.groupby('customer_id').agg([
    pl.col('order_id').list().alias('order_ids'),
    pl.col('product_id').list().unique().alias('unique_products')
])

# Explode lists
exploded = df_with_lists.explode('order_ids')
```

### Joins and Merges

```python
# Inner join
joined = df_left.join(
    df_right,
    on='customer_id',
    how='inner'
)

# Left join with suffix
joined = df_left.join(
    df_right,
    on='customer_id',
    how='left',
    suffix='_right'
)

# Cross join
cross = df_left.join(df_right, how='cross')

# Join on multiple keys
joined = df_left.join(
    df_right,
    left_on=['customer_id', 'order_date'],
    right_on=['cust_id', 'date'],
    how='inner'
)
```

### Parallelization

```python
# Polars automatically parallelizes operations

# Concurrent CSV reading
df = pl.scan_csv('*.csv').collect()

# Parallel groupby
result = df.groupby('customer_id').agg([
    pl.col('amount').sum()
]).collect()  # Automatically parallelized

# Set thread pool size
pl.Config.set_global_string_cache(True)
```

### Memory-Mapped Files

```python
# Memory-map for very large files
df = pl.scan_parquet('large_file.parquet')

# Stream processing
for batch in df.collect_batches(batch_size=10000):
    # Process batch
    processed = batch.filter(pl.col('amount') > 0)
    # Write batch
    processed.write_parquet('output.parquet', mode='append')
```

## Pandas vs Polars Comparison

### Common Operations

```python
# Pandas
df_pandas = pd.read_csv('data.csv')
result_pandas = (
    df_pandas[df_pandas['amount'] > 100]
    .groupby('customer_id')
    .agg({'amount': 'sum'})
    .sort_values('amount', ascending=False)
)

# Polars (equivalent)
df_polars = pl.read_csv('data.csv')
result_polars = (
    df_polars
    .filter(pl.col('amount') > 100)
    .groupby('customer_id')
    .agg(pl.col('amount').sum())
    .sort('amount', descending=True)
)
```

### Performance Benchmark

```python
import time

# Pandas
start = time.time()
pandas_result = df_pandas.groupby('customer_id')['amount'].sum()
pandas_time = time.time() - start

# Polars
start = time.time()
polars_result = df_polars.groupby('customer_id').agg(pl.col('amount').sum())
polars_time = time.time() - start

print(f"Pandas: {pandas_time:.4f}s")
print(f"Polars: {polars_time:.4f}s")
print(f"Speedup: {pandas_time / polars_time:.2f}x")
```

## Production Patterns

### ETL Pipeline with Pandas

```python
def etl_pipeline(source_path: str, target_path: str):
    """Production ETL pipeline with pandas"""

    # Extract
    print("Extracting data...")
    df = pd.read_csv(source_path, parse_dates=['order_date'])

    # Transform
    print("Transforming data...")

    # Data cleaning
    df = df.dropna(subset=['order_id', 'customer_id'])
    df = df.drop_duplicates(subset='order_id')

    # Type conversions
    df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')

    # Feature engineering
    df['revenue'] = df['total_amount'] - df['discount_amount']
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month

    # Aggregations
    customer_summary = df.groupby('customer_id').agg({
        'order_id': 'count',
        'revenue': 'sum',
        'order_date': ['min', 'max']
    })

    # Load
    print("Loading data...")
    df.to_parquet(f"{target_path}/fact_orders.parquet", index=False)
    customer_summary.to_parquet(f"{target_path}/customer_summary.parquet")

    print(f"Processed {len(df):,} records")
```

### ETL Pipeline with Polars

```python
def etl_pipeline_polars(source_path: str, target_path: str):
    """Production ETL pipeline with polars (faster)"""

    # Extract (lazy)
    print("Extracting data...")
    df = pl.scan_csv(source_path, parse_dates=True)

    # Transform (lazy operations)
    print("Transforming data...")
    transformed = (
        df
        .filter(
            pl.col('order_id').is_not_null() &
            pl.col('customer_id').is_not_null()
        )
        .unique(subset='order_id')
        .with_columns([
            pl.col('total_amount').cast(pl.Float64),
            (pl.col('total_amount') - pl.col('discount_amount')).alias('revenue'),
            pl.col('order_date').dt.year().alias('order_year'),
            pl.col('order_date').dt.month().alias('order_month')
        ])
    )

    # Aggregations (lazy)
    customer_summary = (
        transformed
        .groupby('customer_id')
        .agg([
            pl.col('order_id').count().alias('order_count'),
            pl.col('revenue').sum().alias('total_revenue'),
            pl.col('order_date').min().alias('first_order_date'),
            pl.col('order_date').max().alias('last_order_date')
        ])
    )

    # Load (execute lazy operations)
    print("Loading data...")
    transformed.collect().write_parquet(f"{target_path}/fact_orders.parquet")
    customer_summary.collect().write_parquet(f"{target_path}/customer_summary.parquet")

    print("Pipeline complete")
```

## Best Practices

### Pandas Best Practices

1. **Use vectorized operations** instead of loops
2. **Set appropriate dtypes** upfront
3. **Use categorical** for low-cardinality string columns
4. **Chain operations** for readability
5. **Avoid inplace=True** (creates copy anyway)
6. **Use query()** for complex filters

```python
# Good practice example
df = (
    pd.read_csv('data.csv', dtype={'category': 'category'})
    .query('amount > 100 and status == "completed"')
    .assign(
        revenue=lambda x: x['amount'] - x['discount'],
        order_year=lambda x: pd.to_datetime(x['order_date']).dt.year
    )
    .groupby('customer_id')
    .agg({'revenue': 'sum'})
    .reset_index()
)
```

### Polars Best Practices

1. **Use lazy evaluation** for large datasets
2. **Leverage parallel execution**
3. **Use expressions** over apply()
4. **Optimize query with explain()**
5. **Use scan_* for large files**

```python
# Good practice example
result = (
    pl.scan_csv('data.csv')
    .filter(
        (pl.col('amount') > 100) &
        (pl.col('status') == 'completed')
    )
    .with_columns([
        (pl.col('amount') - pl.col('discount')).alias('revenue'),
        pl.col('order_date').dt.year().alias('order_year')
    ])
    .groupby('customer_id')
    .agg(pl.col('revenue').sum())
    .collect()
)
```

## Quality Standards

- **Memory Efficiency**: Optimize DataFrame memory usage
- **Performance**: Benchmark operations for large datasets
- **Type Safety**: Explicit type conversions
- **Documentation**: Document complex transformations
- **Testing**: Unit tests for transformation logic

---

**Skill Type**: Data Engineering - DataFrame Processing
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated for data transformations, aggregations, performance optimization
**Tools**: pandas 1.5+, polars 0.16+, NumPy 1.24+
