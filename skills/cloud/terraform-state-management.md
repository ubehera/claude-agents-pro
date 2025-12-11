---
name: terraform-state-management
description: Advanced Terraform state management, remote backends, workspaces, state locking, and collaboration patterns for production infrastructure. Use when implementing Terraform at scale, managing multi-environment infrastructure, or troubleshooting state issues.
trigger_keywords: [terraform, terraform state, remote backend, s3 backend, terraform workspace, state locking, terraform import, state migration, terraform cloud]
---

# Terraform State Management & Best Practices

Production-grade Terraform state management, remote backends, locking, workspaces, and collaboration patterns for infrastructure at scale.

## Core Concepts

### State Management Fundamentals

**Terraform State** tracks:
- Resource metadata and dependencies
- Output values for cross-stack references
- Provider configurations
- Remote resource attributes

**Critical rules:**
- Never edit state files manually
- Always use remote state for teams
- Enable state locking to prevent corruption
- Regularly back up state files

## Remote State Backends

### AWS S3 + DynamoDB Backend

**Best practice for production AWS infrastructure:**

```hcl
# backend.tf
terraform {
  required_version = ">= 1.6.0"

  backend "s3" {
    bucket         = "myorg-terraform-state"
    key            = "production/vpc/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-west-2:111122223333:key/abcd1234"
    dynamodb_table = "terraform-state-lock"

    # Enable versioning for rollback capability
    versioning     = true
  }
}
```

**Create S3 bucket for state (bootstrap):**

```hcl
# bootstrap/main.tf
provider "aws" {
  region = "us-west-2"
}

resource "aws_s3_bucket" "terraform_state" {
  bucket = "myorg-terraform-state"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.terraform_state.arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# DynamoDB for state locking
resource "aws_dynamodb_table" "terraform_locks" {
  name         = "terraform-state-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# KMS key for encryption
resource "aws_kms_key" "terraform_state" {
  description             = "Terraform state encryption key"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_alias" "terraform_state" {
  name          = "alias/terraform-state"
  target_key_id = aws_kms_key.terraform_state.key_id
}
```

### Terraform Cloud Backend

```hcl
terraform {
  cloud {
    organization = "myorg"

    workspaces {
      name = "production-infrastructure"
    }
  }
}
```

### GCS Backend (Google Cloud)

```hcl
terraform {
  backend "gcs" {
    bucket  = "myorg-terraform-state"
    prefix  = "production/network"

    # Enable state locking
    encryption_key = "projects/myproject/locations/global/keyRings/terraform/cryptoKeys/state"
  }
}
```

### Azure Backend

```hcl
terraform {
  backend "azurerm" {
    resource_group_name  = "terraform-state-rg"
    storage_account_name = "myorgterraformstate"
    container_name       = "tfstate"
    key                  = "production.terraform.tfstate"

    # Enable state locking
    use_azuread_auth     = true
  }
}
```

## Workspaces for Multi-Environment

**Workspaces** - Manage multiple environments with same configuration:

```bash
# List workspaces
terraform workspace list

# Create workspace
terraform workspace new staging
terraform workspace new production

# Switch workspace
terraform workspace select production

# Show current workspace
terraform workspace show
```

**Use workspace in configuration:**

```hcl
# main.tf
locals {
  environment = terraform.workspace

  # Environment-specific settings
  instance_counts = {
    dev        = 1
    staging    = 2
    production = 5
  }

  instance_types = {
    dev        = "t3.micro"
    staging    = "t3.small"
    production = "t3.large"
  }
}

resource "aws_instance" "app" {
  count         = local.instance_counts[local.environment]
  instance_type = local.instance_types[local.environment]

  tags = {
    Name        = "app-${local.environment}-${count.index}"
    Environment = local.environment
  }
}

# Workspace-specific backend keys
terraform {
  backend "s3" {
    bucket = "myorg-terraform-state"
    key    = "env:/${terraform.workspace}/infrastructure.tfstate"
    region = "us-west-2"
  }
}
```

## State Operations

### Importing Existing Resources

```bash
# Import existing AWS VPC
terraform import aws_vpc.main vpc-0abc123def456789

# Import with module
terraform import module.database.aws_db_instance.main mydb-instance

# Import multiple resources
terraform import 'aws_subnet.private[0]' subnet-0abc123
terraform import 'aws_subnet.private[1]' subnet-0def456
```

**Write configuration for imported resources:**

```hcl
# After import, write matching configuration
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "main-vpc"
  }
}

# Verify import worked
# terraform plan  # Should show no changes
```

### State Inspection

```bash
# List all resources in state
terraform state list

# Show specific resource details
terraform state show aws_vpc.main

# Show outputs
terraform output

# Show specific output
terraform output vpc_id
```

### State Manipulation

**Move resources:**

```bash
# Rename resource in state
terraform state mv aws_instance.old aws_instance.new

# Move resource to module
terraform state mv aws_instance.app module.compute.aws_instance.app

# Move between workspaces
terraform state mv -state-out=../other/terraform.tfstate aws_instance.app aws_instance.app
```

**Remove resources from state (keep in cloud):**

```bash
# Remove single resource
terraform state rm aws_instance.old

# Remove entire module
terraform state rm module.old_database

# Remove resource with count
terraform state rm 'aws_subnet.private[2]'
```

### State Recovery

**Pull state for backup:**

```bash
# Download current state
terraform state pull > terraform.tfstate.backup

# Restore state from backup
terraform state push terraform.tfstate.backup
```

**Rollback to previous version (S3):**

```bash
# List S3 versions
aws s3api list-object-versions \
  --bucket myorg-terraform-state \
  --prefix production/vpc/terraform.tfstate

# Download specific version
aws s3api get-object \
  --bucket myorg-terraform-state \
  --key production/vpc/terraform.tfstate \
  --version-id VERSION_ID \
  terraform.tfstate.old

# Push old version
terraform state push terraform.tfstate.old
```

## State Locking

**Handle stuck locks:**

```bash
# Force unlock (use with caution!)
terraform force-unlock LOCK_ID

# Example: DynamoDB lock stuck
terraform force-unlock a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6
```

**Prevent concurrent operations:**

```hcl
# backend.tf - Locking enabled by default with DynamoDB
terraform {
  backend "s3" {
    bucket         = "myorg-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-west-2"
    dynamodb_table = "terraform-state-lock"  # Enables locking
  }
}
```

## State Splitting and Organization

### Monolithic vs. Split State

**❌ Anti-pattern: Single large state file**
```
terraform/
└── main.tf  # Everything in one file (BAD)
```

**✅ Best practice: Split by layer/service**
```
terraform/
├── network/
│   ├── main.tf
│   ├── backend.tf
│   └── outputs.tf
├── compute/
│   ├── main.tf
│   ├── backend.tf
│   └── outputs.tf
└── database/
    ├── main.tf
    ├── backend.tf
    └── outputs.tf
```

**Share data between states with remote state:**

```hcl
# compute/main.tf
data "terraform_remote_state" "network" {
  backend = "s3"

  config = {
    bucket = "myorg-terraform-state"
    key    = "production/network/terraform.tfstate"
    region = "us-west-2"
  }
}

resource "aws_instance" "app" {
  subnet_id = data.terraform_remote_state.network.outputs.private_subnet_ids[0]
  vpc_security_group_ids = [
    data.terraform_remote_state.network.outputs.app_security_group_id
  ]
}
```

## Advanced Patterns

### State Encryption

```hcl
# Encrypt state with KMS
terraform {
  backend "s3" {
    bucket     = "myorg-terraform-state"
    key        = "prod/terraform.tfstate"
    region     = "us-west-2"
    encrypt    = true
    kms_key_id = "arn:aws:kms:us-west-2:111122223333:key/abcd1234"
  }
}

# Encrypt sensitive outputs
output "database_password" {
  value     = random_password.db_password.result
  sensitive = true  # Won't show in plan/apply output
}
```

### State Migration

**Migrate from local to remote:**

```bash
# 1. Add backend configuration
# backend.tf
terraform {
  backend "s3" {
    bucket = "myorg-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-west-2"
  }
}

# 2. Initialize with migration
terraform init -migrate-state

# 3. Verify migration
terraform plan  # Should show no changes

# 4. Delete local state
rm terraform.tfstate terraform.tfstate.backup
```

**Migrate between backends:**

```bash
# Change backend configuration in backend.tf
# Then run:
terraform init -migrate-state -backend-config=new-backend.tfvars
```

### Partial Configuration

**Separate backend config from code:**

```hcl
# backend.tf (partial config)
terraform {
  backend "s3" {
    # Configuration provided via CLI or file
  }
}
```

**backend-prod.tfvars:**
```hcl
bucket         = "myorg-terraform-state"
key            = "production/terraform.tfstate"
region         = "us-west-2"
dynamodb_table = "terraform-state-lock"
encrypt        = true
```

```bash
# Initialize with config file
terraform init -backend-config=backend-prod.tfvars
```

## State Drift Detection

**Detect drift with automation:**

```bash
#!/bin/bash
# check-drift.sh

terraform plan -detailed-exitcode -out=tfplan

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ No drift detected"
elif [ $EXIT_CODE -eq 2 ]; then
  echo "⚠️  Drift detected!"
  terraform show tfplan
  # Send alert (Slack, email, etc.)
  curl -X POST $SLACK_WEBHOOK -d '{"text":"Terraform drift detected!"}'
  exit 1
else
  echo "❌ Terraform plan failed"
  exit 1
fi
```

**Scheduled drift detection (GitHub Actions):**

```yaml
name: Drift Detection

on:
  schedule:
    - cron: '0 8 * * *'  # Daily at 8 AM

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0

      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: us-west-2

      - name: Terraform Init
        run: terraform init

      - name: Detect Drift
        run: |
          terraform plan -detailed-exitcode
        continue-on-error: true

      - name: Alert on Drift
        if: failure()
        run: |
          echo "Drift detected in production infrastructure!"
```

## Collaboration Best Practices

### 1. State File Access Control

```hcl
# IAM policy for Terraform state access
data "aws_iam_policy_document" "terraform_state" {
  statement {
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::111122223333:role/TerraformRole"]
    }

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject"
    ]

    resources = ["${aws_s3_bucket.terraform_state.arn}/*"]
  }

  statement {
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::111122223333:role/TerraformRole"]
    }

    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.terraform_state.arn]
  }
}
```

### 2. Code Review Workflow

```bash
# Developer workflow
git checkout -b feature/new-vpc
# Make changes to .tf files
terraform fmt
terraform validate
terraform plan -out=tfplan

# Create PR with plan output
gh pr create --title "Add VPC" --body "$(terraform show tfplan)"

# After approval, apply
terraform apply tfplan
```

### 3. Module Versioning

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.1.2"  # Pin specific version

  # Don't use:
  # version = ">= 5.0.0"  # Too permissive
  # source = "git::https://..."  # No version control
}
```

## Troubleshooting

### State Corruption

```bash
# Backup current state
terraform state pull > corrupted.tfstate.backup

# Download previous version from S3
aws s3api get-object \
  --bucket myorg-terraform-state \
  --key prod/terraform.tfstate \
  --version-id PREVIOUS_VERSION_ID \
  good.tfstate

# Restore
terraform state push good.tfstate
```

### Lock Timeout

```bash
# Increase timeout
terraform apply -lock-timeout=10m

# Or force unlock if operation failed
terraform force-unlock LOCK_ID
```

### State Inconsistency

```bash
# Refresh state from real infrastructure
terraform refresh

# Or use -refresh-only flag
terraform plan -refresh-only
terraform apply -refresh-only
```

## Best Practices Summary

1. **Always use remote state** for team collaboration
2. **Enable state locking** to prevent corruption
3. **Encrypt state files** - they contain sensitive data
4. **Version state buckets** for rollback capability
5. **Split large states** into logical boundaries
6. **Use workspaces** for environment separation
7. **Document state structure** in README
8. **Automate drift detection** with CI/CD
9. **Restrict state access** with IAM policies
10. **Never commit state files** to Git
11. **Regularly backup state** (automated)
12. **Use partial configuration** for secrets

## Quality Standards

- **Security**: State encrypted at rest and in transit
- **Reliability**: Automated backups, versioning enabled
- **Collaboration**: State locking, code review process
- **Observability**: Drift detection, audit logging
- **Documentation**: Clear state organization, README

## Related Skills

- `terraform-module-library` - For module development
- `ci-cd-patterns` - For automation
- `aws-cloud-patterns` - For AWS infrastructure

---

**Skill Type**: Cloud - Infrastructure as Code
**Complexity**: Advanced
**Typical Usage**: Enterprise Terraform deployments, multi-team collaboration
**Prerequisites**: Terraform fundamentals, cloud provider basics
