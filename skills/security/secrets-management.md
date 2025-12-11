---
name: secrets-management
description: Load when user needs Vault, AWS Secrets Manager, secret rotation, environment variables, or secure secrets management patterns
trigger_keywords: [secrets management, vault, aws secrets manager, secret rotation, environment variables, api key, credentials, secrets, hashicorp vault, key management, secret store]
---

# Secrets Management Skill

Production-grade secrets management with HashiCorp Vault, AWS Secrets Manager, secret rotation strategies, and secure credential handling.

## Overview

Secrets (API keys, database passwords, encryption keys) must never be hardcoded or stored in version control. Use dedicated secret stores with encryption, access control, and audit logging.

**When to Use**:
- Managing database credentials
- Storing API keys and tokens
- Handling encryption keys
- Multi-environment configuration (dev, staging, prod)

## Core Concepts

### Secret Types

```yaml
Credentials:
  - Database passwords
  - API keys
  - OAuth client secrets
  - Service account tokens

Keys:
  - Encryption keys
  - Signing keys (JWT)
  - TLS/SSL certificates

Configuration:
  - Third-party service URLs
  - Feature flags (sensitive)
  - Internal service endpoints
```

### Secret Storage Options

**Environment Variables** (Development Only):
- ✅ Simple, no external dependencies
- ❌ No rotation, no audit trail
- ❌ Visible in process list

**Vault/Secrets Manager** (Production):
- ✅ Centralized, encrypted storage
- ✅ Access control and audit logs
- ✅ Automatic rotation
- ✅ Dynamic credentials

**Cloud Provider Secrets**:
- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager

## Environment Variables (Development)

### Secure .env Files

```python
# .env (NEVER commit to version control)
DATABASE_URL=postgresql://user:password@localhost/db
API_KEY=sk_live_abc123xyz
SECRET_KEY=super-secret-key-change-in-production

# Add to .gitignore
echo ".env" >> .gitignore
```

```python
from pydantic import BaseSettings, PostgresDsn, SecretStr
from functools import lru_cache

class Settings(BaseSettings):
    """Type-safe settings with validation"""

    # Database
    database_url: PostgresDsn

    # API Keys
    api_key: SecretStr  # SecretStr hides value in logs

    # Encryption
    secret_key: SecretStr

    # Environment
    environment: str = "development"

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton"""
    return Settings()

# Usage
settings = get_settings()
db_url = settings.database_url
api_key = settings.api_key.get_secret_value()  # Explicit unwrap
```

### Docker Secrets

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    image: myapp:latest
    secrets:
      - db_password
      - api_key
    environment:
      - DATABASE_URL_FILE=/run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    file: ./secrets/api_key.txt
```

```python
from pathlib import Path

def load_secret(secret_name: str) -> str:
    """Load Docker secret from file"""
    secret_path = Path(f"/run/secrets/{secret_name}")

    if secret_path.exists():
        return secret_path.read_text().strip()

    # Fallback to environment variable (development)
    return os.getenv(secret_name.upper(), "")

# Usage
db_password = load_secret("db_password")
```

## HashiCorp Vault

### Vault Setup

```bash
# Start Vault in dev mode (NOT for production)
vault server -dev

# Export Vault address
export VAULT_ADDR='http://127.0.0.1:8200'
export VAULT_TOKEN='dev-token'

# Enable KV secrets engine (v2)
vault secrets enable -version=2 kv

# Write secret
vault kv put kv/myapp/database password=supersecret

# Read secret
vault kv get kv/myapp/database
```

### Python Vault Client

```python
import hvac
from typing import Dict, Any
from functools import lru_cache

class VaultClient:
    """HashiCorp Vault client"""

    def __init__(self, url: str, token: str):
        self.client = hvac.Client(url=url, token=token)

        if not self.client.is_authenticated():
            raise ValueError("Vault authentication failed")

    def get_secret(self, path: str) -> Dict[str, Any]:
        """Read secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point="kv"
            )
            return response["data"]["data"]
        except hvac.exceptions.InvalidPath:
            raise ValueError(f"Secret not found: {path}")

    def set_secret(self, path: str, secret_data: Dict[str, Any]):
        """Write secret to Vault"""
        self.client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=secret_data,
            mount_point="kv"
        )

    def delete_secret(self, path: str):
        """Delete secret from Vault"""
        self.client.secrets.kv.v2.delete_metadata_and_all_versions(
            path=path,
            mount_point="kv"
        )

@lru_cache()
def get_vault_client() -> VaultClient:
    """Singleton Vault client"""
    vault_url = os.getenv("VAULT_ADDR", "http://localhost:8200")
    vault_token = os.getenv("VAULT_TOKEN")

    return VaultClient(url=vault_url, token=vault_token)

# Usage
vault = get_vault_client()

# Read secrets
db_credentials = vault.get_secret("myapp/database")
db_password = db_credentials["password"]

# Write secrets
vault.set_secret("myapp/api-keys", {
    "stripe": "sk_live_...",
    "sendgrid": "SG...."
})
```

### Dynamic Database Credentials

```python
class VaultDatabaseClient:
    """Generate short-lived database credentials"""

    def __init__(self, client: hvac.Client):
        self.client = client

    def get_db_credentials(self, role: str = "readonly") -> Dict[str, str]:
        """
        Get temporary database credentials (TTL: 1 hour)
        Vault automatically rotates these
        """
        response = self.client.read(f"database/creds/{role}")

        return {
            "username": response["data"]["username"],
            "password": response["data"]["password"],
            "ttl": response["lease_duration"]
        }

# Configure Vault database secrets engine
"""
vault secrets enable database

vault write database/config/postgresql \
    plugin_name=postgresql-database-plugin \
    allowed_roles="readonly,readwrite" \
    connection_url="postgresql://{{username}}:{{password}}@localhost:5432/mydb" \
    username="vault" \
    password="vault-password"

vault write database/roles/readonly \
    db_name=postgresql \
    creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; \
        GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
    default_ttl="1h" \
    max_ttl="24h"
"""

# Usage
vault_db = VaultDatabaseClient(vault.client)
creds = vault_db.get_db_credentials(role="readonly")

# Connect with temporary credentials
engine = create_engine(
    f"postgresql://{creds['username']}:{creds['password']}@localhost/mydb"
)
```

## AWS Secrets Manager

### boto3 Client

```python
import boto3
import json
from botocore.exceptions import ClientError
from functools import lru_cache

class AWSSecretsManager:
    """AWS Secrets Manager client"""

    def __init__(self, region_name: str = "us-east-1"):
        self.client = boto3.client("secretsmanager", region_name=region_name)

    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Retrieve secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)

            # Parse secret string (JSON)
            if "SecretString" in response:
                return json.loads(response["SecretString"])
            else:
                # Binary secret
                return {"secret": response["SecretBinary"]}

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                raise ValueError(f"Secret not found: {secret_name}")
            raise

    def create_secret(self, secret_name: str, secret_value: Dict[str, Any]):
        """Create new secret"""
        self.client.create_secret(
            Name=secret_name,
            SecretString=json.dumps(secret_value)
        )

    def update_secret(self, secret_name: str, secret_value: Dict[str, Any]):
        """Update existing secret"""
        self.client.update_secret(
            SecretId=secret_name,
            SecretString=json.dumps(secret_value)
        )

    def rotate_secret(self, secret_name: str, rotation_lambda_arn: str):
        """Enable automatic rotation"""
        self.client.rotate_secret(
            SecretId=secret_name,
            RotationLambdaARN=rotation_lambda_arn,
            RotationRules={
                "AutomaticallyAfterDays": 30
            }
        )

@lru_cache()
def get_secrets_manager() -> AWSSecretsManager:
    return AWSSecretsManager(region_name=os.getenv("AWS_REGION", "us-east-1"))

# Usage
sm = get_secrets_manager()

# Get database credentials
db_secret = sm.get_secret("prod/myapp/database")
db_password = db_secret["password"]

# Get API keys
api_keys = sm.get_secret("prod/myapp/api-keys")
stripe_key = api_keys["stripe"]
```

### FastAPI Dependency Injection

```python
from fastapi import FastAPI, Depends

app = FastAPI()

async def get_database_password() -> str:
    """Dependency: Fetch DB password from Secrets Manager"""
    sm = get_secrets_manager()
    secret = sm.get_secret("prod/myapp/database")
    return secret["password"]

@app.get("/data")
async def get_data(db_password: str = Depends(get_database_password)):
    """Endpoint with injected secret"""
    # Use db_password to connect to database
    engine = create_engine(f"postgresql://user:{db_password}@host/db")
    # ...
    return {"data": "..."}
```

## Secret Rotation

### Automatic Rotation Pattern

```python
from datetime import datetime, timedelta
import asyncio

class SecretRotator:
    """Automatic secret rotation"""

    def __init__(self, secrets_manager: AWSSecretsManager):
        self.sm = secrets_manager
        self.rotation_interval = timedelta(days=30)

    async def rotate_api_key(self, service_name: str):
        """Rotate API key for external service"""
        secret_name = f"prod/{service_name}/api-key"

        # 1. Generate new API key from service
        new_key = await self.generate_new_key(service_name)

        # 2. Store new key in Secrets Manager
        self.sm.update_secret(secret_name, {"api_key": new_key})

        # 3. Update application configuration (rolling restart)
        await self.trigger_app_reload()

        # 4. Revoke old key (after grace period)
        await asyncio.sleep(3600)  # 1 hour grace period
        await self.revoke_old_key(service_name)

    async def rotate_database_password(self):
        """Rotate database password"""
        secret_name = "prod/myapp/database"

        # 1. Get current credentials
        current = self.sm.get_secret(secret_name)

        # 2. Generate new password
        new_password = secrets.token_urlsafe(32)

        # 3. Create new database user
        await db.execute(
            f"CREATE USER new_user WITH PASSWORD '{new_password}'"
        )

        # 4. Grant same permissions
        await db.execute("GRANT SELECT, INSERT, UPDATE ON ALL TABLES TO new_user")

        # 5. Update secret
        self.sm.update_secret(secret_name, {
            "username": "new_user",
            "password": new_password
        })

        # 6. Restart application (picks up new credentials)
        await self.trigger_app_reload()

        # 7. Drop old user (after grace period)
        await asyncio.sleep(3600)
        await db.execute(f"DROP USER {current['username']}")

    async def background_rotation_task(self):
        """Run periodic rotation"""
        while True:
            await asyncio.sleep(self.rotation_interval.total_seconds())
            await self.rotate_api_key("stripe")
            await self.rotate_database_password()
```

### Lambda Rotation Function (AWS)

```python
import boto3
import json

def lambda_handler(event, context):
    """AWS Lambda function for secret rotation"""
    service_client = boto3.client('secretsmanager')
    secret_id = event['SecretId']
    token = event['ClientRequestToken']
    step = event['Step']

    # Four-step rotation process
    if step == "createSecret":
        create_secret(service_client, secret_id, token)
    elif step == "setSecret":
        set_secret(service_client, secret_id, token)
    elif step == "testSecret":
        test_secret(service_client, secret_id, token)
    elif step == "finishSecret":
        finish_secret(service_client, secret_id, token)

def create_secret(client, secret_id, token):
    """Generate new secret version"""
    new_password = generate_random_password()

    client.put_secret_value(
        SecretId=secret_id,
        ClientRequestToken=token,
        SecretString=json.dumps({"password": new_password}),
        VersionStages=['AWSPENDING']
    )

def set_secret(client, secret_id, token):
    """Apply new secret to database"""
    pending_secret = client.get_secret_value(
        SecretId=secret_id,
        VersionId=token,
        VersionStage='AWSPENDING'
    )

    # Update database password
    new_password = json.loads(pending_secret['SecretString'])['password']
    update_database_password(new_password)

def test_secret(client, secret_id, token):
    """Verify new secret works"""
    pending_secret = client.get_secret_value(
        SecretId=secret_id,
        VersionId=token,
        VersionStage='AWSPENDING'
    )

    # Test database connection with new credentials
    password = json.loads(pending_secret['SecretString'])['password']
    test_database_connection(password)

def finish_secret(client, secret_id, token):
    """Finalize rotation"""
    client.update_secret_version_stage(
        SecretId=secret_id,
        VersionStage='AWSCURRENT',
        MoveToVersionId=token,
        RemoveFromVersionId=get_current_version_id(client, secret_id)
    )
```

## Best Practices

### 1. Never Hardcode Secrets
```python
# ❌ Hardcoded secret
API_KEY = "sk_live_abc123"

# ✅ Load from secret store
API_KEY = vault.get_secret("myapp/api-keys")["stripe"]
```

### 2. Use Secret-Specific Types
```python
from pydantic import SecretStr

class Config(BaseModel):
    api_key: SecretStr  # Never logged or serialized

# Access value explicitly
api_key_value = config.api_key.get_secret_value()
```

### 3. Rotate Secrets Regularly
```python
# ✅ Automatic rotation every 30 days
rotation_policy = {
    "api_keys": timedelta(days=30),
    "database_passwords": timedelta(days=90),
    "encryption_keys": timedelta(days=365)
}
```

### 4. Least Privilege Access
```yaml
# Vault policy (HCL)
path "kv/data/myapp/*" {
  capabilities = ["read"]
}

path "kv/data/admin/*" {
  capabilities = ["deny"]
}
```

### 5. Audit Secret Access
```python
# ✅ Log secret access attempts
import logging

logger = logging.getLogger(__name__)

def get_secret(secret_name: str) -> str:
    logger.info(f"Accessing secret: {secret_name}", extra={
        "user": current_user.id,
        "timestamp": datetime.utcnow()
    })
    return vault.get_secret(secret_name)
```

### 6. Encrypt Secrets at Rest
```python
# ✅ Vault automatically encrypts with master key
# ✅ AWS Secrets Manager uses KMS encryption

# For custom encryption:
from cryptography.fernet import Fernet

def encrypt_secret(plaintext: str, key: bytes) -> bytes:
    f = Fernet(key)
    return f.encrypt(plaintext.encode())

def decrypt_secret(ciphertext: bytes, key: bytes) -> str:
    f = Fernet(key)
    return f.decrypt(ciphertext).decode()
```

## Common Pitfalls

❌ **Committing .env to version control**
```bash
# ✅ Add to .gitignore
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
```

❌ **Logging secrets**
```python
# ❌ Secret appears in logs
logger.info(f"API key: {api_key}")

# ✅ Mask secrets
logger.info(f"API key: {api_key[:8]}***")
```

❌ **Using environment variables in production**
```python
# ❌ Environment variables visible in process list
# ✅ Use Vault or Secrets Manager
```

❌ **Not rotating secrets**
```python
# ❌ Static secrets never rotated
# ✅ Implement automatic rotation
```

❌ **Storing secrets in code**
```python
# ❌ Secret in source code
PASSWORD = "hardcoded-password"

# ✅ Load from secret store
PASSWORD = vault.get_secret("database")["password"]
```

## Quality Standards

- **Storage**: Use Vault/Secrets Manager (never environment variables in production)
- **Rotation**: Automatic rotation every 30-90 days
- **Access Control**: Principle of least privilege
- **Encryption**: Secrets encrypted at rest and in transit
- **Audit Logging**: Log all secret access attempts

---

**Skill Type**: Security - Secrets Management
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when securing credentials and API keys
**Performance**: Secret fetching adds <50ms overhead (cached locally)
