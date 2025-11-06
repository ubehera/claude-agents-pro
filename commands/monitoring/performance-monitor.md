---
description: Monitor agent performance and system resource utilization
args: [--duration 5m|1h|24h] [--metrics cpu|memory|disk|network] [--export-format json|csv]
tools: Bash(ps:*), Bash(top:*), Bash(df:*), Read, Write
model: claude-sonnet-4-5
---

## Objective
Comprehensive performance monitoring for agent execution, script performance, and system resource utilization with automated alerting and optimization recommendations.

## Before You Run
- Ensure monitoring tools are available (`htop`, `iotop`, `nethogs` if available)
- Verify adequate disk space for performance logs
- Check system permissions for resource monitoring
- Configure baseline metrics for comparison

## Execution
Start performance monitoring:

```bash
# Basic performance monitoring for 5 minutes
!/performance-monitor --duration 5m

# Focused CPU and memory monitoring
!/performance-monitor --duration 1h --metrics cpu,memory

# Export detailed metrics to JSON
!/performance-monitor --duration 30m --export-format json > perf-report.json
```

## Performance Metrics

### Agent Execution Performance
- **Response Time**: Agent invocation to completion latency
- **Processing Speed**: Operations per second
- **Memory Usage**: Peak and average memory consumption
- **CPU Utilization**: Processor usage patterns
- **I/O Operations**: File read/write performance
- **Network Activity**: API calls and data transfer

### Script Performance
- **Execution Time**: Script runtime analysis
- **Resource Consumption**: CPU, memory, disk usage
- **Error Rates**: Failure frequency and patterns
- **Throughput**: Operations processed per unit time
- **Concurrency**: Parallel execution efficiency

### System Resource Monitoring
```bash
# CPU monitoring with detailed breakdown
top -b -n 1 | head -20
mpstat 1 5  # 5 samples at 1-second intervals

# Memory analysis
free -h
ps aux --sort=-%mem | head -10

# Disk I/O monitoring
iostat -x 1 5
df -h

# Network monitoring (if available)
netstat -tuln
ss -tuln
```

## Performance Baselines

### Agent Response Times (Target < 2s)
- **Simple Queries**: < 500ms
- **Complex Analysis**: < 2000ms
- **Batch Operations**: < 30s
- **File Operations**: < 1000ms

### Resource Utilization (Target < 70%)
- **CPU Usage**: < 70% sustained
- **Memory Usage**: < 70% available RAM
- **Disk I/O**: < 80% capacity
- **Network**: < 80% bandwidth

### Quality Metrics
- **Success Rate**: > 95%
- **Error Recovery**: < 5s
- **Cache Hit Ratio**: > 80%
- **Concurrent Users**: Support 10+ simultaneous

## Automated Analysis

### Performance Regression Detection
```bash
# Compare current performance to baseline
awk '{if($2 > baseline * 1.2) print "REGRESSION: " $0}' current_metrics.txt
```

### Resource Optimization Recommendations
- **Memory Leaks**: Detect gradually increasing memory usage
- **CPU Bottlenecks**: Identify high-CPU processes
- **I/O Optimization**: Suggest disk/network improvements
- **Caching Opportunities**: Recommend cacheable operations

### Alert Conditions
- **Critical**: Resource usage > 90%
- **Warning**: Performance degradation > 50%
- **Info**: Optimization opportunities detected
- **Success**: Performance improvements achieved

## Real-time Monitoring Dashboard

### System Overview
```bash
# Create simple dashboard
watch -n 1 'echo "=== Claude Agents Pro Performance Dashboard ==="; \
echo "CPU: $(top -b -n1 | grep "Cpu(s)" | awk "{print \$2}" | cut -d% -f1)%"; \
echo "Memory: $(free | grep Mem | awk "{printf(\"%.1f%%\", \$3/\$2 * 100.0)}")"; \
echo "Disk: $(df / | tail -1 | awk "{print \$5}")"; \
echo "Load: $(uptime | awk -F"load average:" "{print \$2}")"; \
echo "Processes: $(ps aux | wc -l)"'
```

### Agent-Specific Metrics
- **Active Agents**: Currently executing agents
- **Queue Depth**: Pending operations
- **Error Rate**: Recent failure percentage
- **Response Time**: Average and P95 latency

## Performance Optimization

### Automatic Optimizations
- **Process Priority**: Adjust nice values for performance
- **Cache Management**: Clear stale caches automatically
- **Resource Allocation**: Dynamic resource assignment
- **Garbage Collection**: Trigger cleanup operations

### Manual Optimization Suggestions
```bash
# Memory optimization
echo 3 > /proc/sys/vm/drop_caches  # Clear system caches

# CPU optimization
renice -n -5 -p $AGENT_PID  # Higher priority for agents

# I/O optimization
echo mq-deadline > /sys/block/sda/queue/scheduler
```

### Configuration Tuning
- **Concurrency Limits**: Optimal parallel execution
- **Buffer Sizes**: I/O buffer optimization
- **Timeout Values**: Balance responsiveness and reliability
- **Retry Policies**: Efficient error recovery

## Reporting & Analytics

### Performance Reports
- **Executive Summary**: High-level performance overview
- **Trend Analysis**: Performance over time
- **Bottleneck Identification**: Primary performance constraints
- **Optimization Recommendations**: Actionable improvements

### Export Formats
```bash
# JSON export for programmatic analysis
{
  "timestamp": "2024-01-15T10:30:00Z",
  "metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 62.8,
    "response_time": {
      "avg": 850,
      "p95": 1200,
      "p99": 1800
    }
  }
}

# CSV export for spreadsheet analysis
echo "timestamp,cpu_usage,memory_usage,response_time_avg" > perf.csv
```

### Integration with Monitoring Systems
- **Prometheus**: Metrics export for monitoring
- **Grafana**: Visualization dashboards
- **Alertmanager**: Automated alerting
- **ELK Stack**: Log aggregation and analysis

## Follow Up
- Review performance trends and patterns
- Implement optimization recommendations
- Set up automated monitoring and alerting
- Schedule regular performance reviews
- Update performance baselines
- Document performance improvements

## Troubleshooting

### Common Performance Issues
- **High CPU**: Identify resource-intensive operations
- **Memory Leaks**: Monitor memory growth patterns
- **Slow I/O**: Analyze disk and network bottlenecks
- **Context Switching**: Optimize process scheduling

### Diagnostic Commands
```bash
# Detailed process analysis
strace -c -p $PID  # System call analysis
lsof -p $PID       # Open file descriptors
pmap $PID          # Memory mapping
```
