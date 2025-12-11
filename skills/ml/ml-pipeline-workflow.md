---
name: ml-pipeline-workflow
description: Build end-to-end MLOps pipelines from data preparation through model training, validation, and production deployment. Use when creating ML pipelines, implementing MLOps practices, or automating model training and deployment workflows.
trigger_keywords:
  - ml pipeline
  - mlops
  - model training
  - model deployment
  - feature engineering
  - data pipeline
  - airflow
  - kubeflow
  - model validation
  - continuous training
---

# ML Pipeline Workflow

Complete end-to-end MLOps pipeline orchestration from data preparation through model deployment.

## Overview

This skill provides comprehensive guidance for building production ML pipelines that handle the full lifecycle: data ingestion → preparation → training → validation → deployment → monitoring.

## When to Use This Skill

- Building new ML pipelines from scratch
- Designing workflow orchestration for ML systems
- Implementing data → model → deployment automation
- Setting up reproducible training workflows
- Creating DAG-based ML orchestration
- Integrating ML components into production systems

## Core Capabilities

### 1. Pipeline Architecture
- End-to-end workflow design
- DAG orchestration patterns (Airflow, Dagster, Kubeflow)
- Component dependencies and data flow
- Error handling and retry strategies

### 2. Data Preparation
- Data validation and quality checks
- Feature engineering pipelines
- Data versioning and lineage
- Train/validation/test splitting strategies

### 3. Model Training
- Training job orchestration
- Hyperparameter management
- Experiment tracking integration
- Distributed training patterns

### 4. Model Validation
- Validation frameworks and metrics
- A/B testing infrastructure
- Performance regression detection
- Model comparison workflows

### 5. Deployment Automation
- Model serving patterns
- Canary deployments
- Blue-green deployment strategies
- Rollback mechanisms

## Usage Patterns

### Basic Pipeline Setup

```python
# 1. Define pipeline stages
stages = [
    "data_ingestion",
    "data_validation",
    "feature_engineering",
    "model_training",
    "model_validation",
    "model_deployment"
]

# 2. Configure dependencies
# See pipeline templates for full example
```

### Production Workflow

1. **Data Preparation Phase**
   - Ingest raw data from sources
   - Run data quality checks
   - Apply feature transformations
   - Version processed datasets

2. **Training Phase**
   - Load versioned training data
   - Execute training jobs
   - Track experiments and metrics
   - Save trained models

3. **Validation Phase**
   - Run validation test suite
   - Compare against baseline
   - Generate performance reports
   - Approve for deployment

4. **Deployment Phase**
   - Package model artifacts
   - Deploy to serving infrastructure
   - Configure monitoring
   - Validate production traffic

## Best Practices

### Pipeline Design

- **Modularity**: Each stage should be independently testable
- **Idempotency**: Re-running stages should be safe
- **Observability**: Log metrics at every stage
- **Versioning**: Track data, code, and model versions
- **Failure Handling**: Implement retry logic and alerting

### Data Management

- Use data validation libraries (Great Expectations, TFX)
- Version datasets with DVC or similar tools
- Document feature engineering transformations
- Maintain data lineage tracking

### Model Operations

- Separate training and serving infrastructure
- Use model registries (MLflow, Weights & Biases)
- Implement gradual rollouts for new models
- Monitor model performance drift
- Maintain rollback capabilities

### Deployment Strategies

- Start with shadow deployments
- Use canary releases for validation
- Implement A/B testing infrastructure
- Set up automated rollback triggers
- Monitor latency and throughput

## Integration Points

### Orchestration Tools

- **Apache Airflow**: DAG-based workflow orchestration
- **Dagster**: Asset-based pipeline orchestration
- **Kubeflow Pipelines**: Kubernetes-native ML workflows
- **Prefect**: Modern dataflow automation

### Experiment Tracking

- MLflow for experiment tracking and model registry
- Weights & Biases for visualization and collaboration
- TensorBoard for training metrics

### Deployment Platforms

- AWS SageMaker for managed ML infrastructure
- Google Vertex AI for GCP deployments
- Azure ML for Azure cloud
- Kubernetes + KServe for cloud-agnostic serving

## Progressive Disclosure

Start with the basics and gradually add complexity:

1. **Level 1**: Simple linear pipeline (data → train → deploy)
2. **Level 2**: Add validation and monitoring stages
3. **Level 3**: Implement hyperparameter tuning
4. **Level 4**: Add A/B testing and gradual rollouts
5. **Level 5**: Multi-model pipelines with ensemble strategies

## Common Patterns

### Batch Training Pipeline

```yaml
# Pipeline stages
stages:
  - name: data_preparation
    dependencies: []
  - name: model_training
    dependencies: [data_preparation]
  - name: model_evaluation
    dependencies: [model_training]
  - name: model_deployment
    dependencies: [model_evaluation]
```

### Real-time Feature Pipeline

```python
# Stream processing for real-time features
# Combined with batch training
from kafka import KafkaConsumer
import mlflow

# Consume streaming data
consumer = KafkaConsumer('feature_events')

for message in consumer:
    features = process_message(message)

    # Store features for training
    mlflow.log_metrics(features)

    # Make real-time predictions
    prediction = model.predict(features)
```

### Continuous Training

```python
# Automated retraining on schedule
# Triggered by data drift detection

def should_retrain(model, new_data):
    """Check if model needs retraining."""
    current_metrics = evaluate_model(model, new_data)
    baseline_metrics = load_baseline_metrics()

    # Retrain if performance drops >5%
    return current_metrics['accuracy'] < baseline_metrics['accuracy'] * 0.95

if should_retrain(model, validation_data):
    trigger_training_pipeline()
```

## Airflow DAG Example

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline',
    schedule_interval='@daily',
    catchup=False
)

# Define tasks
ingest_data = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data_task,
    dag=dag
)

validate_data = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data_task,
    dag=dag
)

engineer_features = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features_task,
    dag=dag
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag
)

evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag
)

deploy_model = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model_task,
    dag=dag
)

# Define dependencies
ingest_data >> validate_data >> engineer_features >> train_model >> evaluate_model >> deploy_model
```

## Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Define feature transformations
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['occupation', 'region']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create full pipeline
ml_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train pipeline
ml_pipeline.fit(X_train, y_train)

# Save pipeline for deployment
joblib.dump(ml_pipeline, 'model_pipeline.pkl')
```

## Model Registry Integration

```python
import mlflow
from mlflow.tracking import MlflowClient

# Initialize MLflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("customer_churn_prediction")

# Start training run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_params({
        'learning_rate': 0.01,
        'max_depth': 10,
        'n_estimators': 100
    })

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)
    mlflow.log_metrics(metrics)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="customer_churn_model"
    )

# Promote to production
client = MlflowClient()
client.transition_model_version_stage(
    name="customer_churn_model",
    version=run.info.run_id,
    stage="Production"
)
```

## Data Validation with Great Expectations

```python
import great_expectations as gx

# Create data context
context = gx.get_context()

# Create expectation suite
suite = context.create_expectation_suite("training_data_expectations")

# Define expectations
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="training_data_expectations"
)

# Add expectations
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=1000000)
validator.expect_column_values_to_not_be_null(column="target")
validator.expect_column_values_to_be_between(column="age", min_value=18, max_value=100)
validator.expect_column_mean_to_be_between(column="income", min_value=20000, max_value=200000)

# Run validation
results = validator.validate()

if not results.success:
    raise ValueError("Data validation failed!")
```

## Troubleshooting

### Common Issues

- **Pipeline failures**: Check dependencies and data availability
- **Training instability**: Review hyperparameters and data quality
- **Deployment issues**: Validate model artifacts and serving config
- **Performance degradation**: Monitor data drift and model metrics

### Debugging Steps

1. Check pipeline logs for each stage
2. Validate input/output data at boundaries
3. Test components in isolation
4. Review experiment tracking metrics
5. Inspect model artifacts and metadata

## Next Steps

After setting up your pipeline:

1. Implement comprehensive monitoring and alerting
2. Set up automated model retraining triggers
3. Create dashboards for pipeline observability
4. Document pipeline architecture and dependencies
5. Establish SLAs for model performance
