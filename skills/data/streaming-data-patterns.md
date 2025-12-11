---
name: streaming-data-patterns
description: Build real-time data streaming pipelines with Kafka, Spark Streaming, and Flink. Load when implementing event-driven architectures, real-time analytics, CDC pipelines, or stream processing applications.
trigger_keywords: [kafka, stream processing, spark streaming, flink, real-time data, event streaming, cdc, change data capture, kinesis, event driven, streaming analytics, kafka streams]
---

# Streaming Data Patterns

Production patterns for real-time data streaming with Apache Kafka, Spark Streaming, and Apache Flink.

## Core Concepts

### Streaming vs Batch

**Batch Processing**:
- Processes data in large chunks
- Higher latency (hours/days)
- Easier to implement
- Lower cost per record

**Stream Processing**:
- Processes data record-by-record
- Low latency (milliseconds/seconds)
- More complex
- Higher cost per record

### Stream Processing Patterns

1. **Event Time vs Processing Time**
   - Event time: When event occurred
   - Processing time: When system processes event
   - Watermarks handle late-arriving data

2. **Windows**
   - Tumbling: Fixed-size, non-overlapping
   - Sliding: Fixed-size, overlapping
   - Session: Activity-based gaps

3. **State Management**
   - Stateless: Each event processed independently
   - Stateful: Maintains state across events

## Apache Kafka Patterns

### Producer Pattern

```python
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
from datetime import datetime

class OrderEventProducer:
    def __init__(self, bootstrap_servers):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8'),
            # Reliability settings
            acks='all',  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=1,  # Ensure ordering
            # Performance settings
            compression_type='gzip',
            batch_size=16384,
            linger_ms=10  # Wait 10ms to batch
        )

    def send_order_event(self, order_id, event_type, payload):
        """Send order event with proper key and headers"""

        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'order_id': order_id,
            'payload': payload
        }

        # Use order_id as key for partitioning
        # All events for same order go to same partition (ordering guaranteed)
        future = self.producer.send(
            topic='orders',
            key=str(order_id),
            value=event,
            headers=[
                ('event_type', event_type.encode('utf-8')),
                ('source', 'order-service'.encode('utf-8'))
            ]
        )

        # Synchronous send with callback
        try:
            record_metadata = future.get(timeout=10)
            print(f"Event sent: partition={record_metadata.partition}, offset={record_metadata.offset}")
        except KafkaError as e:
            print(f"Failed to send event: {e}")
            raise

    def send_batch_events(self, events):
        """Send events in batch"""
        for event in events:
            self.producer.send(
                topic='orders',
                key=event['order_id'],
                value=event
            )

        # Flush to ensure all sent
        self.producer.flush()

    def close(self):
        self.producer.close()
```

### Consumer Pattern

```python
from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
import json

class OrderEventConsumer:
    def __init__(self, bootstrap_servers, group_id):
        self.consumer = KafkaConsumer(
            'orders',
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            # Deserialization
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8'),
            # Consumer settings
            auto_offset_reset='earliest',  # Start from beginning if no offset
            enable_auto_commit=False,  # Manual commit for exactly-once
            max_poll_records=500,
            max_poll_interval_ms=300000  # 5 minutes
        )

    def process_events(self):
        """Process events with manual commit"""
        try:
            while True:
                # Poll for messages
                msg_batch = self.consumer.poll(timeout_ms=1000, max_records=100)

                for topic_partition, messages in msg_batch.items():
                    for message in messages:
                        try:
                            # Process message
                            self.handle_event(message.value)

                            # Commit offset after successful processing
                            self.consumer.commit()

                        except Exception as e:
                            print(f"Error processing message: {e}")
                            # Send to DLQ
                            self.send_to_dlq(message)
                            # Continue processing
                            continue

        except KeyboardInterrupt:
            print("Shutting down consumer...")
        finally:
            self.consumer.close()

    def handle_event(self, event):
        """Handle individual event"""
        event_type = event['event_type']

        if event_type == 'order_created':
            self.handle_order_created(event)
        elif event_type == 'order_updated':
            self.handle_order_updated(event)
        elif event_type == 'order_cancelled':
            self.handle_order_cancelled(event)
        else:
            print(f"Unknown event type: {event_type}")

    def handle_order_created(self, event):
        """Process order created event"""
        order_id = event['order_id']
        payload = event['payload']

        # Insert to database
        db.execute("""
            INSERT INTO orders (order_id, customer_id, amount, status, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (order_id, payload['customer_id'], payload['amount'], 'created', event['timestamp']))

    def send_to_dlq(self, message):
        """Send failed message to dead letter queue"""
        dlq_producer = KafkaProducer(bootstrap_servers=self.consumer.config['bootstrap_servers'])
        dlq_producer.send(
            topic='orders-dlq',
            key=message.key,
            value=message.value,
            headers=[('original_topic', b'orders'), ('error_timestamp', str(datetime.utcnow()).encode())]
        )
        dlq_producer.close()
```

### Kafka Streams Pattern

```python
from kafka import KafkaConsumer, KafkaProducer
from collections import defaultdict
import time

class OrderAggregationStream:
    """Aggregate order events in real-time"""

    def __init__(self, bootstrap_servers):
        self.consumer = KafkaConsumer(
            'orders',
            bootstrap_servers=bootstrap_servers,
            group_id='order-aggregation',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # In-memory state (use Redis/RocksDB for production)
        self.customer_totals = defaultdict(lambda: {'total_amount': 0, 'order_count': 0})

    def process_stream(self):
        """Process stream with stateful aggregation"""

        for message in self.consumer:
            event = message.value

            if event['event_type'] == 'order_created':
                customer_id = event['payload']['customer_id']
                amount = event['payload']['amount']

                # Update state
                self.customer_totals[customer_id]['total_amount'] += amount
                self.customer_totals[customer_id]['order_count'] += 1

                # Emit aggregated event
                self.producer.send(
                    topic='customer-order-totals',
                    key=customer_id,
                    value={
                        'customer_id': customer_id,
                        'total_amount': self.customer_totals[customer_id]['total_amount'],
                        'order_count': self.customer_totals[customer_id]['order_count'],
                        'updated_at': datetime.utcnow().isoformat()
                    }
                )
```

## Spark Structured Streaming

### Basic Stream Processing

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("OrderStreamProcessor") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

# Define schema
order_schema = StructType([
    StructField("order_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("amount", DoubleType()),
    StructField("timestamp", TimestampType())
])

# Read from Kafka
orders_stream = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "orders") \
    .option("startingOffsets", "latest") \
    .option("maxOffsetsPerTrigger", 10000) \
    .load()

# Parse JSON
parsed_orders = orders_stream \
    .select(from_json(col("value").cast("string"), order_schema).alias("data")) \
    .select("data.*")

# Windowed aggregation
windowed_aggregates = parsed_orders \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window(col("timestamp"), "5 minutes", "1 minute"),
        col("customer_id")
    ) \
    .agg(
        sum("amount").alias("total_amount"),
        count("*").alias("order_count"),
        avg("amount").alias("avg_amount")
    )

# Write to multiple sinks
# Sink 1: Kafka (for downstream consumers)
kafka_query = windowed_aggregates \
    .select(
        col("customer_id").cast("string").alias("key"),
        to_json(struct("*")).alias("value")
    ) \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "customer-aggregates") \
    .option("checkpointLocation", "/tmp/checkpoint/kafka") \
    .outputMode("update") \
    .trigger(processingTime='30 seconds') \
    .start()

# Sink 2: Delta Lake (for analytics)
delta_query = windowed_aggregates \
    .writeStream \
    .format("delta") \
    .option("checkpointLocation", "/tmp/checkpoint/delta") \
    .option("path", "s3://data-lake/customer-aggregates/") \
    .outputMode("append") \
    .trigger(processingTime='1 minute') \
    .start()

# Sink 3: Console (for debugging)
console_query = windowed_aggregates \
    .writeStream \
    .format("console") \
    .outputMode("update") \
    .start()

# Wait for termination
spark.streams.awaitAnyTermination()
```

### Stateful Stream Processing

```python
from pyspark.sql.functions import *

# Define state update function
def update_customer_state(customer_id, new_events, state):
    """Update customer aggregation state"""

    # Get current state
    if state.exists:
        total_amount = state.get.total_amount
        order_count = state.get.order_count
    else:
        total_amount = 0.0
        order_count = 0

    # Process new events
    for event in new_events:
        total_amount += event.amount
        order_count += 1

    # Update state
    state.update((total_amount, order_count))

    # Return aggregated result
    return (customer_id, total_amount, order_count)

# Apply stateful operation
customer_aggregates = parsed_orders \
    .groupBy("customer_id") \
    .applyInPandasWithState(
        update_customer_state,
        outputStructType=StructType([
            StructField("customer_id", StringType()),
            StructField("total_amount", DoubleType()),
            StructField("order_count", IntegerType())
        ]),
        stateStructType=StructType([
            StructField("total_amount", DoubleType()),
            StructField("order_count", IntegerType())
        ]),
        outputMode="update",
        timeoutConf="NoTimeout"
    )
```

### Join Streams

```python
# Stream-stream join
orders_stream = spark.readStream.format("kafka") \
    .option("subscribe", "orders").load()

payments_stream = spark.readStream.format("kafka") \
    .option("subscribe", "payments").load()

# Join with watermark
joined = orders_stream \
    .withWatermark("order_timestamp", "10 minutes") \
    .join(
        payments_stream.withWatermark("payment_timestamp", "10 minutes"),
        expr("""
            order_id = payment_order_id AND
            payment_timestamp >= order_timestamp AND
            payment_timestamp <= order_timestamp + interval 1 hour
        """),
        "leftOuter"
    )

# Stream-static join
customers_df = spark.read.parquet("s3://data/customers/")

enriched_orders = parsed_orders \
    .join(customers_df, "customer_id", "left")
```

## Change Data Capture (CDC)

### Debezium with Kafka

```python
from kafka import KafkaConsumer
import json

def process_cdc_events():
    """Process CDC events from Debezium"""

    consumer = KafkaConsumer(
        'dbserver1.public.orders',  # Debezium topic
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    for message in consumer:
        cdc_event = message.value

        # Debezium payload structure
        operation = cdc_event['op']  # c=create, u=update, d=delete
        before = cdc_event.get('before')  # State before change
        after = cdc_event.get('after')  # State after change

        if operation == 'c':  # INSERT
            handle_insert(after)
        elif operation == 'u':  # UPDATE
            handle_update(before, after)
        elif operation == 'd':  # DELETE
            handle_delete(before)

def handle_insert(record):
    """Handle insert event"""
    # Replicate to data warehouse
    warehouse_conn.execute("""
        INSERT INTO orders (order_id, customer_id, amount, created_at)
        VALUES (%s, %s, %s, %s)
    """, (record['order_id'], record['customer_id'], record['amount'], record['created_at']))

def handle_update(before, after):
    """Handle update event"""
    # Replicate update to data warehouse
    warehouse_conn.execute("""
        UPDATE orders
        SET amount = %s, status = %s, updated_at = %s
        WHERE order_id = %s
    """, (after['amount'], after['status'], after['updated_at'], after['order_id']))

def handle_delete(record):
    """Handle delete event"""
    # Soft delete in warehouse
    warehouse_conn.execute("""
        UPDATE orders
        SET deleted_at = CURRENT_TIMESTAMP
        WHERE order_id = %s
    """, (record['order_id'],))
```

### Spark CDC Processing

```python
from pyspark.sql.functions import *

# Read CDC stream
cdc_stream = spark \
    .readStream \
    .format("kafka") \
    .option("subscribe", "dbserver1.public.orders") \
    .load()

# Parse CDC event
parsed_cdc = cdc_stream \
    .select(from_json(col("value").cast("string"), cdc_schema).alias("cdc")) \
    .select("cdc.*")

# Apply CDC logic
def apply_cdc_to_delta(batch_df, batch_id):
    """Apply CDC events to Delta table"""

    # Separate by operation
    inserts = batch_df.filter(col("op") == "c").select("after.*")
    updates = batch_df.filter(col("op") == "u")
    deletes = batch_df.filter(col("op") == "d").select("before.order_id")

    # Apply to Delta table
    from delta.tables import DeltaTable

    delta_table = DeltaTable.forPath(spark, "s3://data-lake/orders/")

    # Handle updates
    if updates.count() > 0:
        delta_table.alias("target").merge(
            updates.select("after.*").alias("source"),
            "target.order_id = source.order_id"
        ).whenMatchedUpdateAll().execute()

    # Handle inserts
    if inserts.count() > 0:
        inserts.write.format("delta").mode("append").save("s3://data-lake/orders/")

    # Handle deletes (soft delete)
    if deletes.count() > 0:
        delta_table.alias("target").merge(
            deletes.alias("source"),
            "target.order_id = source.order_id"
        ).whenMatchedUpdate(set={"deleted_at": current_timestamp()}).execute()

# Apply CDC transformation
cdc_query = parsed_cdc \
    .writeStream \
    .foreachBatch(apply_cdc_to_delta) \
    .option("checkpointLocation", "/tmp/checkpoint/cdc") \
    .start()
```

## Best Practices

### Exactly-Once Semantics

```python
# Kafka producer with idempotence
producer = KafkaProducer(
    enable_idempotence=True,  # Exactly-once producer
    transactional_id='order-producer-1'  # Required for transactions
)

# Start transaction
producer.begin_transaction()

try:
    # Send messages
    producer.send('orders', key='order-1', value=order_event)
    producer.send('order-items', key='order-1', value=item_events)

    # Commit transaction
    producer.commit_transaction()
except Exception as e:
    # Rollback on error
    producer.abort_transaction()
    raise
```

### Backpressure Handling

```python
# Spark Streaming with backpressure
spark = SparkSession.builder \
    .config("spark.streaming.backpressure.enabled", "true") \
    .config("spark.streaming.kafka.maxRatePerPartition", "1000") \
    .getOrCreate()
```

### Late Data Handling

```python
# Define watermark for late data
windowed_stream = parsed_orders \
    .withWatermark("timestamp", "10 minutes") \  # Allow 10 min late data
    .groupBy(window(col("timestamp"), "5 minutes")) \
    .count()
```

## Monitoring

### Kafka Lag Monitoring

```python
from kafka import KafkaConsumer

def get_consumer_lag(group_id, topic):
    """Calculate consumer lag"""
    consumer = KafkaConsumer(
        bootstrap_servers='localhost:9092',
        group_id=group_id
    )

    partitions = consumer.partitions_for_topic(topic)

    for partition in partitions:
        tp = TopicPartition(topic, partition)

        # Get committed offset
        committed = consumer.committed(tp)

        # Get latest offset
        consumer.assign([tp])
        consumer.seek_to_end(tp)
        latest = consumer.position(tp)

        # Calculate lag
        lag = latest - (committed or 0)

        print(f"Partition {partition}: Lag = {lag}")

    consumer.close()
```

## Quality Standards

- **Exactly-Once Processing**: Idempotent producers and transactional consumers
- **Low Latency**: P99 latency <100ms for stream processing
- **Fault Tolerance**: Auto-recovery from failures
- **Monitoring**: Real-time lag and throughput metrics
- **Scalability**: Horizontal scaling with partitions

---

**Skill Type**: Data Engineering - Stream Processing
**Complexity**: Advanced
**Typical Usage**: Activated for real-time data pipelines, event-driven architectures, CDC implementations
**Tools**: Apache Kafka 3.0+, Spark Streaming 3.0+, Apache Flink, Debezium
