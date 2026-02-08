---
name: multi-cloud-patterns
description: Multi-cloud architecture patterns for AWS, GCP, and Azure. Includes cloud-agnostic infrastructure, service mapping, disaster recovery, and cost optimization across providers. Use when designing multi-cloud systems or migrating between cloud providers.
trigger_keywords: [aws, azure, gcp, google cloud, multi-cloud, cloud migration, cloud agnostic, cloud comparison, s3, blob storage, gcs, rds, cloud sql, ec2, compute engine, lambda, cloud functions]
---

# Multi-Cloud Architecture Patterns

Production-grade patterns for multi-cloud infrastructure across AWS, GCP, and Azure with service mapping, cost optimization, and disaster recovery.

## Core Concepts

- **Cloud Agnosticism**: Abstract cloud-specific APIs behind interfaces (storage, compute, database) to enable portability - use Terraform, Pulumi, or Crossplane for infrastructure abstraction
- **Data Gravity**: Data transfer costs and latency often dictate architecture - keep compute close to data, minimize cross-cloud egress, replicate strategically
- **Vendor Lock-in Spectrum**: Balance portability vs. managed service benefits - use cloud-native for commodity services (storage, compute), abstract for differentiated features
- **Active-Active vs. Active-Passive**: Active-active provides better availability but requires data synchronization complexity; active-passive simplifies consistency but increases RTO
- **Unified Observability**: Centralize monitoring, logging, and tracing across clouds using vendor-neutral tools (Prometheus, OpenTelemetry, Grafana) for consistent operational visibility

## Cloud Service Comparison

### Compute Services

| Service Type | AWS | GCP | Azure |
|-------------|-----|-----|-------|
| VMs | EC2 | Compute Engine | Virtual Machines |
| Serverless | Lambda | Cloud Functions | Functions |
| Containers | ECS, EKS | GKE, Cloud Run | AKS, Container Instances |
| Batch | Batch | Batch | Batch |
| App Platform | Elastic Beanstalk | App Engine | App Service |

### Storage Services

| Service Type | AWS | GCP | Azure |
|-------------|-----|-----|-------|
| Object Storage | S3 | Cloud Storage (GCS) | Blob Storage |
| File Storage | EFS | Filestore | Files |
| Block Storage | EBS | Persistent Disk | Managed Disks |
| Archive | Glacier | Coldline/Archive | Archive Storage |

### Database Services

| Service Type | AWS | GCP | Azure |
|-------------|-----|-----|-------|
| Relational | RDS | Cloud SQL | SQL Database |
| NoSQL Document | DocumentDB | Firestore | Cosmos DB |
| NoSQL Key-Value | DynamoDB | Bigtable | Cosmos DB |
| Data Warehouse | Redshift | BigQuery | Synapse Analytics |
| Cache | ElastiCache | Memorystore | Cache for Redis |

### Networking

| Service Type | AWS | GCP | Azure |
|-------------|-----|-----|-------|
| VPC | VPC | VPC | Virtual Network |
| Load Balancer | ELB/ALB/NLB | Cloud Load Balancing | Load Balancer |
| CDN | CloudFront | Cloud CDN | Front Door |
| DNS | Route 53 | Cloud DNS | DNS |
| VPN | VPN Gateway | Cloud VPN | VPN Gateway |

## Cloud-Agnostic Infrastructure

### Terraform Multi-Cloud

**Provider configuration:**

```hcl
# providers.tf
terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

provider "azurerm" {
  features {}
}
```

**Multi-cloud object storage:**

```hcl
# storage.tf

# AWS S3 bucket
resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-aws"

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id

  versioning_configuration {
    status = "Enabled"
  }
}

# GCP Cloud Storage bucket
resource "google_storage_bucket" "data" {
  name          = "${var.project_name}-data-gcp"
  location      = var.gcp_region
  storage_class = "STANDARD"

  versioning {
    enabled = true
  }

  labels = local.common_tags
}

# Azure Blob Storage
resource "azurerm_storage_account" "data" {
  name                     = "${var.project_name}datagcp"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = var.azure_region
  account_tier             = "Standard"
  account_replication_type = "GRS"

  blob_properties {
    versioning_enabled = true
  }

  tags = local.common_tags
}

resource "azurerm_storage_container" "data" {
  name                  = "data"
  storage_account_name  = azurerm_storage_account.data.name
  container_access_type = "private"
}

# Locals for common configuration
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
```

**Multi-cloud Kubernetes:**

```hcl
# kubernetes.tf

# AWS EKS
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project_name}-eks"
  cluster_version = "1.28"

  vpc_id     = module.vpc_aws.vpc_id
  subnet_ids = module.vpc_aws.private_subnets

  eks_managed_node_groups = {
    main = {
      min_size     = 2
      max_size     = 10
      desired_size = 3

      instance_types = ["t3.large"]
    }
  }
}

# GCP GKE
resource "google_container_cluster" "primary" {
  name     = "${var.project_name}-gke"
  location = var.gcp_region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "main-pool"
  cluster    = google_container_cluster.primary.name
  location   = var.gcp_region
  node_count = 3

  node_config {
    machine_type = "e2-standard-4"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }
}

# Azure AKS
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.project_name}-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "${var.project_name}-aks"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D4s_v3"

    enable_auto_scaling = true
    min_count          = 2
    max_count          = 10
  }

  identity {
    type = "SystemAssigned"
  }
}

# Output kubeconfig for all clusters
output "eks_kubeconfig" {
  value     = module.eks.kubeconfig
  sensitive = true
}

output "gke_kubeconfig" {
  value = templatefile("${path.module}/kubeconfig.tpl", {
    cluster_name = google_container_cluster.primary.name
    endpoint     = google_container_cluster.primary.endpoint
    ca_cert      = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  })
  sensitive = true
}

output "aks_kubeconfig" {
  value     = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive = true
}
```

## Application-Level Abstraction

### Storage Abstraction Layer

```typescript
// storage-adapter.ts
export interface StorageAdapter {
  upload(key: string, data: Buffer, metadata?: Record<string, string>): Promise<void>;
  download(key: string): Promise<Buffer>;
  delete(key: string): Promise<void>;
  list(prefix: string): Promise<string[]>;
  getSignedUrl(key: string, expiresIn: number): Promise<string>;
}

// AWS S3 implementation
import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

export class S3StorageAdapter implements StorageAdapter {
  private client: S3Client;
  private bucket: string;

  constructor(region: string, bucket: string) {
    this.client = new S3Client({ region });
    this.bucket = bucket;
  }

  async upload(key: string, data: Buffer, metadata?: Record<string, string>): Promise<void> {
    await this.client.send(new PutObjectCommand({
      Bucket: this.bucket,
      Key: key,
      Body: data,
      Metadata: metadata,
    }));
  }

  async download(key: string): Promise<Buffer> {
    const response = await this.client.send(new GetObjectCommand({
      Bucket: this.bucket,
      Key: key,
    }));
    return Buffer.from(await response.Body.transformToByteArray());
  }

  async getSignedUrl(key: string, expiresIn: number): Promise<string> {
    const command = new GetObjectCommand({
      Bucket: this.bucket,
      Key: key,
    });
    return getSignedUrl(this.client, command, { expiresIn });
  }

  // ... other methods
}

// GCP Cloud Storage implementation
import { Storage } from '@google-cloud/storage';

export class GCSStorageAdapter implements StorageAdapter {
  private storage: Storage;
  private bucket: string;

  constructor(projectId: string, bucket: string) {
    this.storage = new Storage({ projectId });
    this.bucket = bucket;
  }

  async upload(key: string, data: Buffer, metadata?: Record<string, string>): Promise<void> {
    const file = this.storage.bucket(this.bucket).file(key);
    await file.save(data, {
      metadata: metadata,
    });
  }

  async download(key: string): Promise<Buffer> {
    const file = this.storage.bucket(this.bucket).file(key);
    const [contents] = await file.download();
    return contents;
  }

  async getSignedUrl(key: string, expiresIn: number): Promise<string> {
    const file = this.storage.bucket(this.bucket).file(key);
    const [url] = await file.getSignedUrl({
      action: 'read',
      expires: Date.now() + expiresIn * 1000,
    });
    return url;
  }

  // ... other methods
}

// Azure Blob Storage implementation
import { BlobServiceClient } from '@azure/storage-blob';

export class AzureBlobStorageAdapter implements StorageAdapter {
  private client: BlobServiceClient;
  private containerName: string;

  constructor(connectionString: string, containerName: string) {
    this.client = BlobServiceClient.fromConnectionString(connectionString);
    this.containerName = containerName;
  }

  async upload(key: string, data: Buffer, metadata?: Record<string, string>): Promise<void> {
    const containerClient = this.client.getContainerClient(this.containerName);
    const blockBlobClient = containerClient.getBlockBlobClient(key);
    await blockBlobClient.upload(data, data.length, {
      metadata: metadata,
    });
  }

  async download(key: string): Promise<Buffer> {
    const containerClient = this.client.getContainerClient(this.containerName);
    const blobClient = containerClient.getBlobClient(key);
    const downloadResponse = await blobClient.download();
    return Buffer.from(await downloadResponse.blobBody);
  }

  // ... other methods
}

// Factory pattern
export class StorageFactory {
  static create(provider: 'aws' | 'gcp' | 'azure', config: any): StorageAdapter {
    switch (provider) {
      case 'aws':
        return new S3StorageAdapter(config.region, config.bucket);
      case 'gcp':
        return new GCSStorageAdapter(config.projectId, config.bucket);
      case 'azure':
        return new AzureBlobStorageAdapter(config.connectionString, config.container);
      default:
        throw new Error(`Unsupported provider: ${provider}`);
    }
  }
}

// Usage
const storage = StorageFactory.create(process.env.CLOUD_PROVIDER, {
  region: process.env.AWS_REGION,
  bucket: process.env.STORAGE_BUCKET,
  // ... other config
});

await storage.upload('file.pdf', fileBuffer, { contentType: 'application/pdf' });
```

### Database Abstraction

```typescript
// database-adapter.ts
export interface DatabaseAdapter {
  query<T>(sql: string, params: any[]): Promise<T[]>;
  execute(sql: string, params: any[]): Promise<void>;
  transaction<T>(callback: (tx: Transaction) => Promise<T>): Promise<T>;
}

// PostgreSQL (RDS/Cloud SQL/Azure SQL)
import { Pool } from 'pg';

export class PostgreSQLAdapter implements DatabaseAdapter {
  private pool: Pool;

  constructor(config: { host: string; database: string; user: string; password: string }) {
    this.pool = new Pool(config);
  }

  async query<T>(sql: string, params: any[]): Promise<T[]> {
    const result = await this.pool.query(sql, params);
    return result.rows as T[];
  }

  async execute(sql: string, params: any[]): Promise<void> {
    await this.pool.query(sql, params);
  }

  async transaction<T>(callback: (tx: any) => Promise<T>): Promise<T> {
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      const result = await callback(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }
}
```

## Multi-Cloud Disaster Recovery

### Cross-Cloud Backup Strategy

```yaml
# backup-strategy.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cross-cloud-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup-tool:latest
            env:
            # AWS credentials
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret-key
            # GCP credentials
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: /secrets/gcp/credentials.json
            # Azure credentials
            - name: AZURE_STORAGE_CONNECTION_STRING
              valueFrom:
                secretKeyRef:
                  name: azure-credentials
                  key: connection-string
            command:
            - /bin/sh
            - -c
            - |
              # Backup from AWS S3 to GCP GCS
              aws s3 sync s3://primary-bucket gs://backup-bucket --delete

              # Backup database
              pg_dump $DATABASE_URL | gzip | \
                aws s3 cp - s3://backups/$(date +%Y%m%d).sql.gz

              # Replicate to Azure
              aws s3 sync s3://backups/ \
                https://${AZURE_ACCOUNT}.blob.core.windows.net/backups/
          restartPolicy: OnFailure
```

### Multi-Region Failover

**DNS-based failover with health checks:**

```hcl
# Route 53 (AWS) for global DNS
resource "aws_route53_health_check" "primary" {
  fqdn              = "api-aws.example.com"
  port              = 443
  type              = "HTTPS"
  resource_path     = "/health"
  failure_threshold = 3
  request_interval  = 30

  tags = {
    Name = "Primary API Health Check"
  }
}

resource "aws_route53_health_check" "secondary_gcp" {
  fqdn              = "api-gcp.example.com"
  port              = 443
  type              = "HTTPS"
  resource_path     = "/health"
  failure_threshold = 3
  request_interval  = 30
}

# Failover routing
resource "aws_route53_record" "api" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.example.com"
  type    = "A"

  # Primary endpoint (AWS)
  failover_routing_policy {
    type = "PRIMARY"
  }

  set_identifier  = "Primary"
  health_check_id = aws_route53_health_check.primary.id

  alias {
    name                   = aws_lb.primary.dns_name
    zone_id                = aws_lb.primary.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "api_failover" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.example.com"
  type    = "A"

  # Secondary endpoint (GCP)
  failover_routing_policy {
    type = "SECONDARY"
  }

  set_identifier  = "Secondary"
  health_check_id = aws_route53_health_check.secondary_gcp.id

  alias {
    name                   = google_compute_global_address.api.address
    zone_id                = "Z3AQBSTGFYJSTF"  # GCP zone
    evaluate_target_health = true
  }
}
```

## Cost Optimization

### Cost Comparison Calculator

```python
# cost_calculator.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class ComputeCost:
    provider: str
    instance_type: str
    hourly_rate: float
    monthly_rate: float

def compare_compute_costs(vcpus: int, memory_gb: int, hours_per_month: int = 730):
    costs = {
        'aws': ComputeCost(
            provider='AWS',
            instance_type='t3.xlarge',
            hourly_rate=0.1664,
            monthly_rate=0.1664 * hours_per_month
        ),
        'gcp': ComputeCost(
            provider='GCP',
            instance_type='e2-standard-4',
            hourly_rate=0.13,
            monthly_rate=0.13 * hours_per_month
        ),
        'azure': ComputeCost(
            provider='Azure',
            instance_type='D4s_v3',
            hourly_rate=0.192,
            monthly_rate=0.192 * hours_per_month
        )
    }

    return sorted(costs.values(), key=lambda x: x.monthly_rate)

# Storage costs
@dataclass
class StorageCost:
    provider: str
    cost_per_gb_month: float
    egress_cost_per_gb: float

storage_costs = {
    'aws_s3_standard': StorageCost('AWS S3 Standard', 0.023, 0.09),
    'gcp_gcs_standard': StorageCost('GCP GCS Standard', 0.020, 0.12),
    'azure_blob_hot': StorageCost('Azure Blob Hot', 0.018, 0.087),
}
```

## Best Practices

1. **Cloud Selection Criteria**
   - Use each cloud's strengths (AWS: breadth, GCP: data/ML, Azure: enterprise)
   - Consider data residency requirements
   - Evaluate existing tooling and expertise

2. **Abstraction Layers**
   - Implement cloud-agnostic interfaces
   - Use Kubernetes for compute portability
   - Terraform for infrastructure consistency

3. **Cost Management**
   - Use reserved instances/committed use
   - Implement auto-scaling across clouds
   - Monitor and optimize regularly

4. **Security**
   - Centralized identity (OIDC federation)
   - Consistent security policies
   - Encrypted data at rest and in transit

5. **Operations**
   - Unified monitoring (Prometheus, Grafana)
   - Centralized logging
   - Common CI/CD pipelines

## Quality Standards

- **Portability**: 90% of infrastructure can migrate clouds in <1 week
- **Disaster Recovery**: RPO <1 hour, RTO <4 hours
- **Cost Efficiency**: Multi-cloud deployment within 110% of single-cloud cost
- **Reliability**: 99.95% uptime across all clouds

## Related Skills

- `terraform-state-management` - For infrastructure management
- `kubernetes-advanced-patterns` - For container orchestration
- `ci-cd-patterns` - For deployment automation

---

**Skill Type**: Cloud - Multi-Cloud Architecture
**Complexity**: Advanced
**Typical Usage**: Enterprise cloud strategy, disaster recovery, vendor diversity
**Prerequisites**: Cloud fundamentals (AWS/GCP/Azure), infrastructure as code
