---
name: kubernetes-advanced-patterns
description: Advanced Kubernetes patterns including StatefulSets, Operators, CRDs, advanced networking, and production-grade cluster management. Use when implementing complex K8s workloads, custom controllers, or enterprise cluster architectures.
trigger_keywords: [kubernetes, k8s, statefulset, operator, custom resource definition, crd, daemonset, service mesh, istio, ingress, network policy, pod disruption budget, kustomize, helm]
---

# Kubernetes Advanced Patterns

Production-grade Kubernetes patterns for complex workloads, custom resources, operators, and enterprise cluster management.

## Core Concepts

### StatefulSets vs Deployments

**StatefulSets** - For stateful applications requiring:
- Stable, unique network identifiers
- Stable, persistent storage
- Ordered, graceful deployment and scaling
- Ordered, automated rolling updates

**Use cases**: Databases, message queues, distributed systems (Kafka, Cassandra, Elasticsearch)

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
        ports:
        - containerPort: 3306
          name: mysql
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: password
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
---
# Headless service for stable network identity
apiVersion: v1
kind: Service
metadata:
  name: mysql
spec:
  clusterIP: None  # Headless service
  selector:
    app: mysql
  ports:
  - port: 3306
    name: mysql
```

### Custom Resource Definitions (CRDs)

**Extend Kubernetes API with custom resources:**

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com
spec:
  group: example.com
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                engine:
                  type: string
                  enum: [postgres, mysql, mongodb]
                version:
                  type: string
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 10
                storage:
                  type: string
                  pattern: '^[0-9]+Gi$'
              required: [engine, version]
            status:
              type: object
              properties:
                phase:
                  type: string
                  enum: [Pending, Creating, Running, Failed]
                endpoint:
                  type: string
  scope: Namespaced
  names:
    plural: databases
    singular: database
    kind: Database
    shortNames:
    - db
---
# Use custom resource
apiVersion: example.com/v1
kind: Database
metadata:
  name: production-db
spec:
  engine: postgres
  version: "15.3"
  replicas: 3
  storage: 100Gi
```

### Kubernetes Operators

**Operator Pattern** - Custom controller that extends Kubernetes:

```go
// Simplified operator controller example
package controller

import (
    "context"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/reconcile"
)

type DatabaseReconciler struct {
    client.Client
}

func (r *DatabaseReconciler) Reconcile(ctx context.Context, req reconcile.Request) (reconcile.Result, error) {
    // Fetch the Database instance
    database := &examplev1.Database{}
    err := r.Get(ctx, req.NamespacedName, database)
    if err != nil {
        return reconcile.Result{}, client.IgnoreNotFound(err)
    }

    // Reconciliation logic
    switch database.Spec.Engine {
    case "postgres":
        return r.reconcilePostgres(ctx, database)
    case "mysql":
        return r.reconcileMySQL(ctx, database)
    default:
        return reconcile.Result{}, nil
    }
}

func (r *DatabaseReconciler) reconcilePostgres(ctx context.Context, db *examplev1.Database) (reconcile.Result, error) {
    // Create StatefulSet
    statefulSet := &appsv1.StatefulSet{
        ObjectMeta: metav1.ObjectMeta{
            Name:      db.Name,
            Namespace: db.Namespace,
        },
        Spec: appsv1.StatefulSetSpec{
            Replicas: &db.Spec.Replicas,
            Selector: &metav1.LabelSelector{
                MatchLabels: map[string]string{"app": db.Name},
            },
            // ... rest of StatefulSet spec
        },
    }

    // Create or update StatefulSet
    err := r.Client.Create(ctx, statefulSet)
    if err != nil {
        // Handle error or update existing
    }

    // Update status
    db.Status.Phase = "Running"
    db.Status.Endpoint = fmt.Sprintf("%s.%s.svc.cluster.local", db.Name, db.Namespace)
    return reconcile.Result{}, r.Status().Update(ctx, db)
}
```

## Advanced Networking

### Network Policies

**Restrict pod communication with network policies:**

```yaml
# Default deny all ingress traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
---
# Allow specific ingress from frontend to backend
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-ingress
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: backend
  policyTypes:
  - Ingress
  ingress:
  # Allow from frontend pods
  - from:
    - podSelector:
        matchLabels:
          tier: frontend
    ports:
    - protocol: TCP
      port: 8080
  # Allow from monitoring
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
---
# Egress control - allow only DNS and external APIs
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-egress
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: backend
  policyTypes:
  - Egress
  egress:
  # Allow DNS
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53
  # Allow to database
  - to:
    - podSelector:
        matchLabels:
          tier: database
    ports:
    - protocol: TCP
      port: 5432
  # Allow external API
  - to:
    - podSelector: {}
    ports:
    - protocol: TCP
      port: 443
```

### Ingress with TLS

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - app.example.com
    - api.example.com
    secretName: app-tls
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
  - host: api.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8080
```

### Service Mesh (Istio)

```yaml
# VirtualService for traffic management
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
  - reviews
  http:
  # Canary deployment: 90% to v1, 10% to v2
  - match:
    - headers:
        end-user:
          exact: test-user
    route:
    - destination:
        host: reviews
        subset: v2
  - route:
    - destination:
        host: reviews
        subset: v1
      weight: 90
    - destination:
        host: reviews
        subset: v2
      weight: 10
    timeout: 10s
    retries:
      attempts: 3
      perTryTimeout: 2s
---
# DestinationRule for circuit breaking
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews
spec:
  host: reviews
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 5
    outlierDetection:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

## Advanced Scheduling

### Pod Affinity and Anti-Affinity

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 3
  template:
    spec:
      # Anti-affinity: spread pods across nodes
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - backend
            topologyKey: kubernetes.io/hostname
        # Affinity: prefer nodes with cache
        podAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - cache
              topologyKey: kubernetes.io/hostname
        # Node affinity: prefer SSD nodes
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 50
            preference:
              matchExpressions:
              - key: disk-type
                operator: In
                values:
                - ssd
```

### Taints and Tolerations

```yaml
# Taint node for dedicated workloads
# kubectl taint nodes node1 workload=database:NoSchedule

apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  template:
    spec:
      # Tolerate database taint
      tolerations:
      - key: workload
        operator: Equal
        value: database
        effect: NoSchedule
      # Also tolerate spot instances
      - key: node.kubernetes.io/instance-type
        operator: Equal
        value: spot
        effect: NoSchedule
      nodeSelector:
        workload: database
```

### Pod Disruption Budgets

```yaml
# Ensure minimum availability during disruptions
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: backend-pdb
spec:
  minAvailable: 2  # At least 2 pods must remain
  selector:
    matchLabels:
      app: backend
---
# Alternative: max disruption
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: frontend-pdb
spec:
  maxUnavailable: 1  # At most 1 pod can be unavailable
  selector:
    matchLabels:
      app: frontend
```

## DaemonSets

**Run pod on every node (logging, monitoring, networking):**

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: logging
spec:
  selector:
    matchLabels:
      name: fluentd
  template:
    metadata:
      labels:
        name: fluentd
    spec:
      tolerations:
      # Run on all nodes including masters
      - key: node-role.kubernetes.io/control-plane
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        resources:
          limits:
            memory: 200Mi
          requests:
            cpu: 100m
            memory: 200Mi
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
```

## Jobs and CronJobs

### One-time Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: database-migration
spec:
  ttlSecondsAfterFinished: 3600  # Clean up after 1 hour
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: migrate
        image: myapp:latest
        command: ["npm", "run", "migrate"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

### Scheduled CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  concurrencyPolicy: Forbid  # Don't allow concurrent runs
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/sh
            - -c
            - |
              pg_dump $DATABASE_URL | gzip > /backup/$(date +%Y%m%d).sql.gz
              aws s3 cp /backup/$(date +%Y%m%d).sql.gz s3://backups/
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: postgres-secret
                  key: url
            volumeMounts:
            - name: backup
              mountPath: /backup
          volumes:
          - name: backup
            emptyDir: {}
```

## Kustomize Advanced Patterns

**Directory structure:**
```
base/
├── kustomization.yaml
├── deployment.yaml
├── service.yaml
└── configmap.yaml
overlays/
├── dev/
│   ├── kustomization.yaml
│   └── patches/
│       └── replica-count.yaml
├── staging/
│   └── kustomization.yaml
└── production/
    ├── kustomization.yaml
    └── patches/
        ├── replica-count.yaml
        └── resource-limits.yaml
```

**base/kustomization.yaml:**
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- deployment.yaml
- service.yaml
- configmap.yaml

commonLabels:
  app: myapp
  managed-by: kustomize

commonAnnotations:
  deployed-by: ci-cd
```

**overlays/production/kustomization.yaml:**
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

bases:
- ../../base

namespace: production

replicas:
- name: myapp
  count: 10

images:
- name: myapp
  newTag: v2.1.0

patchesStrategicMerge:
- patches/replica-count.yaml
- patches/resource-limits.yaml

configMapGenerator:
- name: myapp-config
  behavior: merge
  literals:
  - ENV=production
  - LOG_LEVEL=warn
```

## Advanced ConfigMap and Secret Management

### External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: database-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: postgres-secret
    creationPolicy: Owner
  data:
  - secretKey: password
    remoteRef:
      key: production/database
      property: password
  - secretKey: url
    remoteRef:
      key: production/database
      property: connection_url
```

### Sealed Secrets

```bash
# Install sealed-secrets controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Create sealed secret
kubectl create secret generic mysecret --dry-run=client --from-literal=password=secret -o yaml | \
  kubeseal -o yaml > sealed-secret.yaml

# Commit sealed-secret.yaml to git (safe!)
```

```yaml
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: mysecret
spec:
  encryptedData:
    password: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEq...
```

## Best Practices

### Resource Quotas and Limit Ranges

```yaml
# Namespace resource quota
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: development
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
    pods: "50"
---
# Default limits for pods
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: development
spec:
  limits:
  - default:
      cpu: 500m
      memory: 512Mi
    defaultRequest:
      cpu: 100m
      memory: 128Mi
    type: Container
```

### Priority Classes

```yaml
# High priority for critical workloads
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
value: 1000000
globalDefault: false
description: "High priority for critical production workloads"
---
# Use in deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: critical-app
spec:
  template:
    spec:
      priorityClassName: high-priority
```

## Quality Standards

- **High Availability**: Use PodDisruptionBudgets for critical services
- **Security**: Implement NetworkPolicies, Pod Security Standards
- **Observability**: Add Prometheus annotations, structured logging
- **Resource Management**: Set requests/limits, use ResourceQuotas
- **Disaster Recovery**: Regular backups, documented restore procedures
- **GitOps**: All manifests in version control, automated deployment

## Related Skills

- `ci-cd-patterns` - For deployment automation
- `terraform-module-library` - For infrastructure provisioning
- `prometheus-configuration` - For monitoring

---

**Skill Type**: DevOps - Container Orchestration
**Complexity**: Advanced
**Typical Usage**: Enterprise Kubernetes deployments, stateful workloads, custom controllers
**Prerequisites**: Basic Kubernetes knowledge, YAML proficiency
