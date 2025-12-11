---
name: data-quality-frameworks
description: Implement comprehensive data quality validation using Great Expectations, custom validators, and automated testing frameworks. Load when ensuring data integrity, implementing quality gates, or building data validation pipelines.
trigger_keywords: [data quality, data validation, great expectations, data testing, schema validation, quality checks, data integrity, data profiling, validation rules, quality gates]
---

# Data Quality Frameworks

Production-grade data quality validation patterns using Great Expectations, custom validators, and automated testing.

## Core Principles

### Data Quality Dimensions

1. **Accuracy**: Data correctly represents real-world values
2. **Completeness**: All required data is present
3. **Consistency**: Data is uniform across systems
4. **Timeliness**: Data is up-to-date
5. **Validity**: Data conforms to business rules
6. **Uniqueness**: No duplicate records exist

### Quality Gate Strategy

```
Data Source → Validation → Pipeline Processing → Validation → Target
     ↓            ↓                                    ↓           ↓
   Schema     Raw Data                            Transformed   Load
   Check      Quality                               Quality    Success
```

## Great Expectations Framework

### Setup and Configuration

```python
import great_expectations as gx
from great_expectations.data_context import DataContext
from great_expectations.checkpoint import SimpleCheckpoint
import pandas as pd

# Initialize Great Expectations
context = gx.get_context()

# Create datasource
datasource_config = {
    "name": "postgres_datasource",
    "class_name": "Datasource",
    "execution_engine": {
        "class_name": "SqlAlchemyExecutionEngine",
        "connection_string": "postgresql://user:pass@localhost:5432/db"
    },
    "data_connectors": {
        "default_inferred_data_connector": {
            "class_name": "InferredAssetSqlDataConnector",
            "include_schema_name": True
        }
    }
}

context.add_datasource(**datasource_config)
```

### Creating Expectation Suites

```python
# Create expectation suite
suite = context.create_expectation_suite(
    expectation_suite_name="orders_validation_suite",
    overwrite_existing=True
)

# Define expectations
validator = context.get_validator(
    batch_request=BatchRequest(
        datasource_name="postgres_datasource",
        data_connector_name="default_inferred_data_connector",
        data_asset_name="orders"
    ),
    expectation_suite_name="orders_validation_suite"
)

# Table-level expectations
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=1000000)
validator.expect_table_column_count_to_equal(value=15)

# Column existence
validator.expect_column_to_exist(column="order_id")
validator.expect_column_to_exist(column="customer_id")
validator.expect_column_to_exist(column="total_amount")

# Null checks
validator.expect_column_values_to_not_be_null(column="order_id")
validator.expect_column_values_to_not_be_null(column="customer_id")

# Uniqueness
validator.expect_column_values_to_be_unique(column="order_id")

# Data types
validator.expect_column_values_to_be_of_type(column="order_id", type_="INTEGER")
validator.expect_column_values_to_be_of_type(column="total_amount", type_="NUMERIC")

# Value ranges
validator.expect_column_values_to_be_between(
    column="total_amount",
    min_value=0,
    max_value=1000000
)

# Set membership
validator.expect_column_values_to_be_in_set(
    column="status",
    value_set=['pending', 'processing', 'completed', 'cancelled']
)

# String patterns
validator.expect_column_values_to_match_regex(
    column="email",
    regex=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

# Statistical expectations
validator.expect_column_mean_to_be_between(
    column="total_amount",
    min_value=10,
    max_value=500
)

validator.expect_column_median_to_be_between(
    column="total_amount",
    min_value=20,
    max_value=200
)

# Date expectations
validator.expect_column_values_to_be_between(
    column="order_date",
    min_value="2020-01-01",
    max_value="2030-12-31"
)

# Custom business logic
validator.expect_column_pair_values_a_to_be_greater_than_b(
    column_a="total_amount",
    column_b="discount_amount"
)

# Save suite
validator.save_expectation_suite(discard_failed_expectations=False)
```

### Running Validations

```python
# Create checkpoint
checkpoint_config = {
    "name": "orders_checkpoint",
    "config_version": 1.0,
    "class_name": "SimpleCheckpoint",
    "validations": [
        {
            "batch_request": {
                "datasource_name": "postgres_datasource",
                "data_connector_name": "default_inferred_data_connector",
                "data_asset_name": "orders"
            },
            "expectation_suite_name": "orders_validation_suite"
        }
    ]
}

context.add_checkpoint(**checkpoint_config)

# Run checkpoint
results = context.run_checkpoint(checkpoint_name="orders_checkpoint")

# Check results
if results["success"]:
    print("✅ Data quality validation passed")
else:
    print("❌ Data quality validation failed")
    for result in results["run_results"].values():
        print(f"Failed expectations: {result['validation_result']['statistics']['unsuccessful_expectations']}")
```

### Airflow Integration

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
import great_expectations as gx

def validate_data_quality(**context):
    """Validate data quality with Great Expectations"""

    ge_context = gx.get_context()

    # Run checkpoint
    results = ge_context.run_checkpoint(
        checkpoint_name="orders_checkpoint",
        run_name=f"validation_{context['execution_date']}"
    )

    # Push results to XCom
    context['ti'].xcom_push(key='validation_success', value=results['success'])
    context['ti'].xcom_push(
        key='validation_stats',
        value=results['run_results'][0]['validation_result']['statistics']
    )

    # Fail task if validation fails
    if not results['success']:
        raise ValueError("Data quality validation failed")

    return results

validate_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag
)
```

## Custom Validation Framework

### Validator Base Class

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd

class DataValidator(ABC):
    """Base class for data validators"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_results = []

    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """Run validation checks"""
        pass

    def add_result(self, check_name: str, passed: bool, message: str, severity: str = 'error'):
        """Add validation result"""
        self.validation_results.append({
            'check': check_name,
            'passed': passed,
            'message': message,
            'severity': severity
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r['passed'])
        failed_checks = total_checks - passed_checks

        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'success_rate': (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            'results': self.validation_results
        }

    def is_valid(self) -> bool:
        """Check if all validations passed"""
        return all(r['passed'] for r in self.validation_results if r['severity'] == 'error')
```

### Schema Validator

```python
class SchemaValidator(DataValidator):
    """Validate DataFrame schema"""

    def __init__(self, df: pd.DataFrame, expected_schema: Dict[str, str]):
        super().__init__(df)
        self.expected_schema = expected_schema

    def validate(self) -> Dict[str, Any]:
        """Validate schema matches expectations"""

        # Check columns exist
        expected_columns = set(self.expected_schema.keys())
        actual_columns = set(self.df.columns)

        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns

        if missing_columns:
            self.add_result(
                'missing_columns',
                False,
                f"Missing columns: {missing_columns}",
                'error'
            )
        else:
            self.add_result('missing_columns', True, "All expected columns present")

        if extra_columns:
            self.add_result(
                'extra_columns',
                False,
                f"Extra columns: {extra_columns}",
                'warning'
            )

        # Check data types
        for column, expected_dtype in self.expected_schema.items():
            if column in self.df.columns:
                actual_dtype = str(self.df[column].dtype)

                if actual_dtype != expected_dtype:
                    self.add_result(
                        f'dtype_{column}',
                        False,
                        f"Column {column}: expected {expected_dtype}, got {actual_dtype}",
                        'error'
                    )
                else:
                    self.add_result(f'dtype_{column}', True, f"Column {column} has correct type")

        return self.get_summary()


# Usage
expected_schema = {
    'order_id': 'int64',
    'customer_id': 'int64',
    'total_amount': 'float64',
    'order_date': 'datetime64[ns]',
    'status': 'object'
}

validator = SchemaValidator(df, expected_schema)
results = validator.validate()

if not validator.is_valid():
    print("Schema validation failed!")
    for result in results['results']:
        if not result['passed']:
            print(f"  ❌ {result['check']}: {result['message']}")
```

### Business Rule Validator

```python
class BusinessRuleValidator(DataValidator):
    """Validate business rules"""

    def validate(self) -> Dict[str, Any]:
        """Run business rule validations"""

        # Rule 1: Order total must be positive
        negative_totals = self.df[self.df['total_amount'] <= 0]
        self.add_result(
            'positive_totals',
            len(negative_totals) == 0,
            f"Found {len(negative_totals)} orders with non-positive totals"
        )

        # Rule 2: Discount cannot exceed total
        invalid_discounts = self.df[self.df['discount_amount'] > self.df['total_amount']]
        self.add_result(
            'valid_discounts',
            len(invalid_discounts) == 0,
            f"Found {len(invalid_discounts)} orders where discount > total"
        )

        # Rule 3: Status transitions are valid
        valid_statuses = ['pending', 'processing', 'completed', 'cancelled']
        invalid_statuses = self.df[~self.df['status'].isin(valid_statuses)]
        self.add_result(
            'valid_statuses',
            len(invalid_statuses) == 0,
            f"Found {len(invalid_statuses)} orders with invalid status"
        )

        # Rule 4: Order date cannot be in future
        future_orders = self.df[self.df['order_date'] > pd.Timestamp.now()]
        self.add_result(
            'valid_dates',
            len(future_orders) == 0,
            f"Found {len(future_orders)} orders with future dates"
        )

        # Rule 5: Customer ID must exist in customers table
        valid_customer_ids = get_valid_customer_ids()
        orphan_orders = self.df[~self.df['customer_id'].isin(valid_customer_ids)]
        self.add_result(
            'referential_integrity',
            len(orphan_orders) == 0,
            f"Found {len(orphan_orders)} orders with invalid customer_id",
            'error'
        )

        return self.get_summary()
```

### Data Quality Metrics

```python
class DataQualityMetrics:
    """Calculate data quality metrics"""

    @staticmethod
    def completeness(df: pd.DataFrame, column: str) -> float:
        """Calculate completeness percentage"""
        total_rows = len(df)
        non_null_rows = df[column].notna().sum()
        return (non_null_rows / total_rows) * 100

    @staticmethod
    def uniqueness(df: pd.DataFrame, column: str) -> float:
        """Calculate uniqueness percentage"""
        total_rows = len(df)
        unique_rows = df[column].nunique()
        return (unique_rows / total_rows) * 100

    @staticmethod
    def validity(df: pd.DataFrame, column: str, valid_values: set) -> float:
        """Calculate validity percentage"""
        total_rows = len(df)
        valid_rows = df[df[column].isin(valid_values)].shape[0]
        return (valid_rows / total_rows) * 100

    @staticmethod
    def accuracy(df: pd.DataFrame, column: str, reference_df: pd.DataFrame, reference_column: str) -> float:
        """Calculate accuracy by comparing to reference data"""
        merged = df.merge(reference_df, left_on=column, right_on=reference_column, how='inner')
        return (len(merged) / len(df)) * 100

    @staticmethod
    def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        profile = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'columns': {}
        }

        for column in df.columns:
            col_profile = {
                'dtype': str(df[column].dtype),
                'null_count': int(df[column].isna().sum()),
                'null_percentage': float((df[column].isna().sum() / len(df)) * 100),
                'unique_count': int(df[column].nunique()),
                'unique_percentage': float((df[column].nunique() / len(df)) * 100)
            }

            # Numeric column stats
            if pd.api.types.is_numeric_dtype(df[column]):
                col_profile.update({
                    'min': float(df[column].min()) if not df[column].isna().all() else None,
                    'max': float(df[column].max()) if not df[column].isna().all() else None,
                    'mean': float(df[column].mean()) if not df[column].isna().all() else None,
                    'median': float(df[column].median()) if not df[column].isna().all() else None,
                    'std': float(df[column].std()) if not df[column].isna().all() else None
                })

            # String column stats
            elif pd.api.types.is_string_dtype(df[column]):
                col_profile.update({
                    'min_length': int(df[column].str.len().min()) if not df[column].isna().all() else None,
                    'max_length': int(df[column].str.len().max()) if not df[column].isna().all() else None,
                    'avg_length': float(df[column].str.len().mean()) if not df[column].isna().all() else None
                })

            profile['columns'][column] = col_profile

        return profile
```

## Quality Monitoring Dashboard

### Quality Metrics Collection

```python
import json
from datetime import datetime

class QualityMetricsCollector:
    """Collect and store quality metrics"""

    def __init__(self, metrics_store_path: str):
        self.metrics_store_path = metrics_store_path

    def collect_metrics(self, df: pd.DataFrame, table_name: str, run_date: str):
        """Collect quality metrics for dataset"""

        metrics = {
            'table_name': table_name,
            'run_date': run_date,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }

        # Overall metrics
        metrics['metrics']['row_count'] = len(df)
        metrics['metrics']['column_count'] = len(df.columns)
        metrics['metrics']['duplicate_rows'] = int(df.duplicated().sum())

        # Per-column metrics
        for column in df.columns:
            metrics['metrics'][f'{column}_completeness'] = DataQualityMetrics.completeness(df, column)
            metrics['metrics'][f'{column}_uniqueness'] = DataQualityMetrics.uniqueness(df, column)

        # Save metrics
        with open(f"{self.metrics_store_path}/{table_name}_{run_date}.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def get_historical_metrics(self, table_name: str, days: int = 30):
        """Get historical metrics for trending"""
        # Implementation to fetch and aggregate historical metrics
        pass
```

## Best Practices

### 1. Layered Validation

```python
def validate_data_pipeline(df: pd.DataFrame) -> bool:
    """Multi-layer validation"""

    # Layer 1: Schema validation
    schema_validator = SchemaValidator(df, expected_schema)
    schema_results = schema_validator.validate()

    if not schema_validator.is_valid():
        print("Schema validation failed")
        return False

    # Layer 2: Business rule validation
    business_validator = BusinessRuleValidator(df)
    business_results = business_validator.validate()

    if not business_validator.is_valid():
        print("Business rule validation failed")
        return False

    # Layer 3: Statistical validation
    # Check for data drift, anomalies, etc.

    return True
```

### 2. Quality Gates

```python
def quality_gate(df: pd.DataFrame, min_quality_score: float = 95.0) -> bool:
    """Enforce quality gate"""

    validators = [
        SchemaValidator(df, expected_schema),
        BusinessRuleValidator(df)
    ]

    total_checks = 0
    passed_checks = 0

    for validator in validators:
        results = validator.validate()
        total_checks += results['total_checks']
        passed_checks += results['passed_checks']

    quality_score = (passed_checks / total_checks) * 100

    print(f"Quality Score: {quality_score:.2f}%")

    if quality_score < min_quality_score:
        raise ValueError(f"Quality gate failed: {quality_score:.2f}% < {min_quality_score}%")

    return True
```

### 3. Incremental Validation

```python
def validate_incremental_data(new_df: pd.DataFrame, baseline_df: pd.DataFrame):
    """Validate incremental data against baseline"""

    # Check schema consistency
    assert set(new_df.columns) == set(baseline_df.columns), "Schema mismatch"

    # Check data drift
    for column in new_df.select_dtypes(include=['number']).columns:
        new_mean = new_df[column].mean()
        baseline_mean = baseline_df[column].mean()

        drift_percentage = abs(new_mean - baseline_mean) / baseline_mean * 100

        if drift_percentage > 20:  # 20% drift threshold
            print(f"Warning: {column} has {drift_percentage:.2f}% drift")
```

## Quality Standards

- **Validation Coverage**: >95% of data validated
- **Quality Score**: >98% validation pass rate
- **Automation**: All quality checks automated
- **Monitoring**: Real-time quality dashboards
- **Documentation**: All validation rules documented

---

**Skill Type**: Data Engineering - Data Quality
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated for data quality validation, pipeline quality gates, data profiling
**Tools**: Great Expectations 0.15+, pandas, custom frameworks
