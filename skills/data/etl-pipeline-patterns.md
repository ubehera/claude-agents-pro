---
name: etl-pipeline-patterns
description: Design and implement production-grade ETL/ELT pipelines with orchestration, incremental processing, and error handling. Load when building data pipelines, batch processing workflows, data orchestration with Airflow/Prefect, or implementing data quality checks.
trigger_keywords: [etl, elt, data pipeline, airflow, dagster, prefect, batch processing, incremental load, data orchestration, pipeline orchestration, dag, workflow automation]
---

# ETL/ELT Pipeline Patterns

Production-grade patterns for extracting, transforming, and loading data at scale with Apache Airflow, Prefect, and modern orchestration tools.

## Core Concepts

### ETL vs ELT

**ETL (Extract-Transform-Load)**:
```
Sources → Transform (Staging) → Load → Data Warehouse
```
- Transform before loading
- Works well with structured data
- Less warehouse compute cost
- Better data validation before load

**ELT (Extract-Load-Transform)**:
```
Sources → Load → Data Warehouse → Transform (dbt/SQL)
```
- Load raw data first
- Leverage warehouse compute (Snowflake, BigQuery)
- More flexible transformations
- Modern data stack pattern

## Apache Airflow Patterns

### Basic ETL DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateObjectOperator
from datetime import datetime, timedelta
import pandas as pd

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(hours=1)
}

dag = DAG(
    'etl_customer_orders',
    default_args=default_args,
    description='Extract customer orders and load to warehouse',
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False,
    max_active_runs=1,
    tags=['production', 'etl', 'orders']
)

def extract_orders(**context):
    """Extract orders from source database"""
    execution_date = context['execution_date']

    # Extract since last successful run
    query = f"""
        SELECT * FROM orders
        WHERE updated_at >= '{execution_date - timedelta(days=1)}'
        AND updated_at < '{execution_date}'
    """

    df = pd.read_sql(query, source_conn)

    # Save to staging
    staging_path = f"s3://staging/orders/{execution_date.strftime('%Y-%m-%d')}.parquet"
    df.to_parquet(staging_path, index=False)

    # Push metadata to XCom
    context['ti'].xcom_push(key='staging_path', value=staging_path)
    context['ti'].xcom_push(key='record_count', value=len(df))

    return staging_path

def validate_orders(**context):
    """Data quality validation"""
    from great_expectations import DataContext

    staging_path = context['ti'].xcom_pull(key='staging_path')
    df = pd.read_parquet(staging_path)

    # Quality checks
    assert df['order_id'].is_unique, "Duplicate order IDs found"
    assert df['total_amount'].min() >= 0, "Negative amounts found"
    assert df['user_id'].notna().all(), "NULL user_ids found"

    # Great Expectations validation
    ge_context = DataContext()
    suite = ge_context.get_expectation_suite("orders_suite")

    validation_result = ge_context.run_validation_operator(
        "validate_orders",
        assets_to_validate=[staging_path],
        expectation_suite_name="orders_suite"
    )

    if not validation_result.success:
        raise ValueError("Data quality validation failed")

def transform_orders(**context):
    """Transform orders data"""
    staging_path = context['ti'].xcom_pull(key='staging_path')
    df = pd.read_parquet(staging_path)

    # Transformations
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['revenue'] = df['total_amount'] - df['discount_amount']
    df['order_quarter'] = df['order_date'].dt.quarter

    # Enrichment (join with dimension tables)
    customers = pd.read_sql("SELECT * FROM dim_customers", warehouse_conn)
    df = df.merge(customers, on='user_id', how='left')

    # Save transformed data
    output_path = staging_path.replace('/orders/', '/orders_transformed/')
    df.to_parquet(output_path, index=False)

    context['ti'].xcom_push(key='transformed_path', value=output_path)

    return output_path

def load_to_warehouse(**context):
    """Load to data warehouse"""
    transformed_path = context['ti'].xcom_pull(key='transformed_path')

    # Snowflake COPY INTO pattern
    copy_sql = f"""
        COPY INTO fact_orders
        FROM @s3_stage/{transformed_path}
        FILE_FORMAT = (TYPE = PARQUET)
        MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
        ON_ERROR = 'ABORT_STATEMENT';
    """

    # Execute load
    warehouse_conn.execute(copy_sql)

    # Update metadata table
    metadata_sql = f"""
        INSERT INTO etl_metadata (table_name, load_date, record_count, status)
        VALUES ('fact_orders', CURRENT_TIMESTAMP, {context['ti'].xcom_pull(key='record_count')}, 'SUCCESS')
    """
    warehouse_conn.execute(metadata_sql)

# Define tasks
extract = PythonOperator(
    task_id='extract_orders',
    python_callable=extract_orders,
    dag=dag
)

validate = PythonOperator(
    task_id='validate_orders',
    python_callable=validate_orders,
    dag=dag
)

transform = PythonOperator(
    task_id='transform_orders',
    python_callable=transform_orders,
    dag=dag
)

load = PythonOperator(
    task_id='load_to_warehouse',
    python_callable=load_to_warehouse,
    dag=dag
)

# Define dependencies
extract >> validate >> transform >> load
```

### Incremental Loading Pattern

```python
def incremental_extract(**context):
    """Incremental data extraction using watermark"""
    from airflow.models import Variable

    # Get last successful watermark
    last_watermark = Variable.get('orders_last_watermark', default_var='2025-01-01')
    current_watermark = context['execution_date'].strftime('%Y-%m-%d %H:%M:%S')

    query = f"""
        SELECT *
        FROM orders
        WHERE updated_at > '{last_watermark}'
        AND updated_at <= '{current_watermark}'
    """

    df = pd.read_sql(query, source_conn)

    # Save data
    staging_path = f"s3://staging/orders_incremental/{current_watermark}.parquet"
    df.to_parquet(staging_path, index=False)

    # Update watermark on success
    Variable.set('orders_last_watermark', current_watermark)

    context['ti'].xcom_push(key='records_processed', value=len(df))

    return staging_path

# Merge pattern for incremental load
merge_sql = """
    MERGE INTO fact_orders AS target
    USING staging_orders AS source
    ON target.order_id = source.order_id
    WHEN MATCHED AND source.updated_at > target.updated_at THEN
        UPDATE SET
            total_amount = source.total_amount,
            status = source.status,
            updated_at = source.updated_at
    WHEN NOT MATCHED THEN
        INSERT (order_id, user_id, total_amount, status, created_at, updated_at)
        VALUES (source.order_id, source.user_id, source.total_amount,
                source.status, source.created_at, source.updated_at);
"""
```

### Parallel Processing Pattern

```python
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator

def determine_partitions(**context):
    """Determine data partitions for parallel processing"""
    partitions = [
        {'region': 'us-east', 'date': '2025-01-01'},
        {'region': 'us-west', 'date': '2025-01-01'},
        {'region': 'eu', 'date': '2025-01-01'},
    ]

    context['ti'].xcom_push(key='partitions', value=partitions)

    # Return task IDs for dynamic task mapping
    return [f"process_partition_{p['region']}" for p in partitions]

def process_partition(partition, **context):
    """Process a single partition"""
    query = f"""
        SELECT * FROM orders
        WHERE region = '{partition['region']}'
        AND order_date = '{partition['date']}'
    """

    df = pd.read_sql(query, source_conn)

    # Process data
    df_transformed = transform_data(df)

    # Save partition
    output_path = f"s3://warehouse/orders/region={partition['region']}/date={partition['date']}.parquet"
    df_transformed.to_parquet(output_path, index=False)

# Dynamic task generation (Airflow 2.0+)
from airflow.decorators import task

@task
def get_partitions():
    return [
        {'region': 'us-east', 'date': '2025-01-01'},
        {'region': 'us-west', 'date': '2025-01-01'},
        {'region': 'eu', 'date': '2025-01-01'},
    ]

@task
def process_partition(partition):
    # Processing logic
    pass

partitions = get_partitions()
process_partition.expand(partition=partitions)
```

## Data Quality Patterns

### Great Expectations Integration

```python
from great_expectations.data_context import DataContext
from great_expectations.checkpoint import SimpleCheckpoint

def validate_with_great_expectations(**context):
    """Validate data using Great Expectations"""

    # Initialize GE context
    ge_context = DataContext('/path/to/great_expectations')

    # Get data
    staging_path = context['ti'].xcom_pull(key='staging_path')
    df = pd.read_parquet(staging_path)

    # Create batch
    batch = ge_context.get_batch(
        batch_kwargs={
            'datasource': 'pandas_datasource',
            'dataset': df,
            'data_asset_name': 'orders'
        },
        expectation_suite_name='orders_suite'
    )

    # Define expectations
    batch.expect_column_values_to_not_be_null('order_id')
    batch.expect_column_values_to_be_unique('order_id')
    batch.expect_column_values_to_be_between('total_amount', min_value=0, max_value=1000000)
    batch.expect_column_values_to_be_in_set('status', ['pending', 'completed', 'cancelled'])

    # Run checkpoint
    checkpoint = SimpleCheckpoint(
        name='orders_checkpoint',
        data_context=ge_context,
        validations=[
            {
                'batch_request': batch.batch_request,
                'expectation_suite_name': 'orders_suite'
            }
        ]
    )

    result = checkpoint.run()

    # Raise if validation fails
    if not result.success:
        raise ValueError(f"Data quality check failed: {result}")

    # Push validation metrics
    context['ti'].xcom_push(key='validation_success_rate', value=result.statistics['success_percent'])
```

### Custom Data Quality Checks

```python
def data_quality_checks(df, **context):
    """Custom data quality validation"""

    checks_passed = []
    checks_failed = []

    # Check 1: No duplicates
    duplicates = df['order_id'].duplicated().sum()
    if duplicates > 0:
        checks_failed.append(f"Found {duplicates} duplicate order_ids")
    else:
        checks_passed.append("No duplicate order_ids")

    # Check 2: Valid date range
    invalid_dates = df[df['order_date'] > datetime.now()].shape[0]
    if invalid_dates > 0:
        checks_failed.append(f"Found {invalid_dates} future dates")
    else:
        checks_passed.append("All dates are valid")

    # Check 3: Referential integrity
    customer_ids = set(pd.read_sql("SELECT id FROM customers", conn)['id'])
    orphan_orders = df[~df['user_id'].isin(customer_ids)].shape[0]
    if orphan_orders > 0:
        checks_failed.append(f"Found {orphan_orders} orders with invalid user_ids")
    else:
        checks_passed.append("Referential integrity maintained")

    # Check 4: Business rule validation
    negative_amounts = df[df['total_amount'] < 0].shape[0]
    if negative_amounts > 0:
        checks_failed.append(f"Found {negative_amounts} negative amounts")
    else:
        checks_passed.append("All amounts are positive")

    # Check 5: Completeness
    required_columns = ['order_id', 'user_id', 'total_amount', 'order_date']
    for col in required_columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            checks_failed.append(f"Column {col} has {null_count} NULL values")
        else:
            checks_passed.append(f"Column {col} is complete")

    # Log results
    print(f"Checks passed: {len(checks_passed)}")
    print(f"Checks failed: {len(checks_failed)}")

    # Push to XCom
    context['ti'].xcom_push(key='quality_checks_passed', value=checks_passed)
    context['ti'].xcom_push(key='quality_checks_failed', value=checks_failed)

    # Fail task if critical checks fail
    if checks_failed:
        raise ValueError(f"Data quality checks failed:\n" + "\n".join(checks_failed))
```

## Error Handling Patterns

### Retry with Exponential Backoff

```python
from airflow.exceptions import AirflowException
import time

def extract_with_retry(**context):
    """Extract with custom retry logic"""
    max_retries = 5
    base_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            # Attempt extraction
            df = pd.read_sql(query, source_conn)
            return df

        except Exception as e:
            if attempt == max_retries - 1:
                raise AirflowException(f"Failed after {max_retries} attempts: {e}")

            # Exponential backoff
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed. Retrying in {delay}s...")
            time.sleep(delay)
```

### Dead Letter Queue Pattern

```python
def process_with_dlq(**context):
    """Process data with dead letter queue for failures"""
    staging_path = context['ti'].xcom_pull(key='staging_path')
    df = pd.read_parquet(staging_path)

    successful_records = []
    failed_records = []

    for idx, record in df.iterrows():
        try:
            # Process record
            processed = transform_record(record)
            successful_records.append(processed)

        except Exception as e:
            # Send to DLQ
            failed_record = {
                **record.to_dict(),
                'error': str(e),
                'failed_at': datetime.now(),
                'pipeline_run_id': context['run_id']
            }
            failed_records.append(failed_record)

    # Save successful records
    pd.DataFrame(successful_records).to_parquet(
        f"s3://warehouse/orders/{context['ds']}.parquet"
    )

    # Save failed records to DLQ
    if failed_records:
        pd.DataFrame(failed_records).to_parquet(
            f"s3://dlq/orders/{context['ds']}.parquet"
        )

        # Alert if too many failures
        failure_rate = len(failed_records) / len(df)
        if failure_rate > 0.05:  # >5% failure rate
            raise AirflowException(f"High failure rate: {failure_rate:.2%}")
```

## Monitoring and Observability

### Pipeline Metrics

```python
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

def collect_pipeline_metrics(**context):
    """Collect and push pipeline metrics"""
    from prometheus_client import Gauge, push_to_gateway

    # Get metrics from XCom
    record_count = context['ti'].xcom_pull(key='record_count')
    processing_time = context['ti'].xcom_pull(key='processing_time')

    # Push to Prometheus
    gauge = Gauge('etl_records_processed', 'Records processed', ['dag_id', 'task_id'])
    gauge.labels(dag_id=context['dag'].dag_id, task_id=context['task'].task_id).set(record_count)

    push_to_gateway('localhost:9091', job='airflow_etl', registry=CollectorRegistry())

    # Push to DataDog
    from datadog import statsd
    statsd.gauge('etl.records_processed', record_count, tags=[f"dag:{context['dag'].dag_id}"])
    statsd.timing('etl.processing_time', processing_time, tags=[f"dag:{context['dag'].dag_id}"])

def send_success_notification(**context):
    """Send success notification"""
    record_count = context['ti'].xcom_pull(key='record_count')

    message = f"""
    ✅ ETL Pipeline Completed Successfully

    DAG: {context['dag'].dag_id}
    Run Date: {context['execution_date']}
    Records Processed: {record_count:,}
    Duration: {context['dag_run'].duration} seconds
    """

    return message

success_notification = SlackWebhookOperator(
    task_id='send_success_notification',
    http_conn_id='slack_webhook',
    message=send_success_notification(),
    dag=dag
)
```

## Prefect Patterns

### Prefect Flow Example

```python
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

@task(retries=3, retry_delay_seconds=60)
def extract_data(source_table: str, execution_date: str):
    """Extract data from source"""
    query = f"""
        SELECT * FROM {source_table}
        WHERE DATE(updated_at) = '{execution_date}'
    """
    return pd.read_sql(query, source_conn)

@task
def validate_data(df: pd.DataFrame):
    """Validate extracted data"""
    assert df['id'].is_unique, "Duplicate IDs found"
    assert df['amount'].min() >= 0, "Negative amounts found"
    return df

@task
def transform_data(df: pd.DataFrame):
    """Transform data"""
    df['revenue'] = df['amount'] - df['discount']
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

@task
def load_data(df: pd.DataFrame, target_table: str):
    """Load data to warehouse"""
    df.to_sql(target_table, warehouse_conn, if_exists='append', index=False)
    return len(df)

@flow(task_runner=ConcurrentTaskRunner())
def etl_orders_flow(execution_date: str):
    """ETL flow for orders"""

    # Extract
    raw_data = extract_data('orders', execution_date)

    # Validate
    validated_data = validate_data(raw_data)

    # Transform
    transformed_data = transform_data(validated_data)

    # Load
    record_count = load_data(transformed_data, 'fact_orders')

    return record_count

# Create deployment
deployment = Deployment.build_from_flow(
    flow=etl_orders_flow,
    name='etl-orders-daily',
    schedule=CronSchedule(cron='0 2 * * *'),  # 2 AM daily
    work_queue_name='data-pipelines'
)

deployment.apply()
```

## Best Practices

### Pipeline Design Principles

1. **Idempotency**: Pipelines should be rerunnable without side effects
   ```python
   # Use MERGE/UPSERT instead of INSERT
   # Use partition overwrite instead of append
   # Check for existing data before processing
   ```

2. **Atomicity**: Use transactions for consistency
   ```python
   with warehouse_conn.begin():
       # All operations succeed or all fail
       conn.execute("DELETE FROM staging_orders WHERE date = '{ds}'")
       conn.execute("INSERT INTO staging_orders SELECT * FROM ...")
   ```

3. **Observability**: Log everything
   ```python
   import logging

   logging.info(f"Processing {len(df)} records for {execution_date}")
   logging.info(f"Extracted from: {source_table}")
   logging.info(f"Loading to: {target_table}")
   ```

4. **Data Lineage**: Track data provenance
   ```python
   metadata = {
       'source': source_table,
       'target': target_table,
       'execution_date': execution_date,
       'record_count': len(df),
       'pipeline_run_id': run_id
   }
   ```

5. **Schema Evolution**: Handle schema changes gracefully
   ```python
   # Detect schema changes
   current_schema = df.dtypes.to_dict()
   expected_schema = load_expected_schema()

   if current_schema != expected_schema:
       log_schema_change(current_schema, expected_schema)
   ```

## Common Anti-Patterns

### ❌ Anti-Pattern 1: No Incremental Processing

```python
# ❌ BAD: Full table scan every time
df = pd.read_sql("SELECT * FROM orders", conn)

# ✅ GOOD: Incremental load
df = pd.read_sql(f"""
    SELECT * FROM orders
    WHERE updated_at > '{last_watermark}'
""", conn)
```

### ❌ Anti-Pattern 2: No Error Handling

```python
# ❌ BAD: No error handling
df.to_sql('fact_orders', conn)

# ✅ GOOD: Proper error handling
try:
    df.to_sql('fact_orders', conn, if_exists='append')
except Exception as e:
    logging.error(f"Load failed: {e}")
    # Save to DLQ
    df.to_parquet(f"s3://dlq/orders/{execution_date}.parquet")
    raise
```

### ❌ Anti-Pattern 3: Hardcoded Configuration

```python
# ❌ BAD: Hardcoded values
df = pd.read_sql("SELECT * FROM orders WHERE region = 'us-east'", conn)

# ✅ GOOD: Configurable
region = Variable.get('etl_region', default_var='us-east')
df = pd.read_sql(f"SELECT * FROM orders WHERE region = '{region}'", conn)
```

## Quality Standards

- **Idempotency**: All pipeline runs are rerunnable
- **Monitoring**: Metrics tracked for every run
- **Data Quality**: Automated validation before load
- **Error Handling**: Retry logic and DLQ for failures
- **Documentation**: Data lineage and pipeline documentation
- **Performance**: SLA adherence (<1 hour for batch jobs)

---

**Skill Type**: Data Engineering - Pipeline Orchestration
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when data engineers build ETL/ELT pipelines, orchestrate workflows, or implement data quality frameworks
**Tools**: Apache Airflow 2.0+, Prefect 2.0+, Dagster, Great Expectations
