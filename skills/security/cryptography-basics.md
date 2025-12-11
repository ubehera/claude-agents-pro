---
name: cryptography-basics
description: Load when user needs hashing, encryption, signing, key management, or cryptographic primitives for security
trigger_keywords: [cryptography, encryption, hashing, signing, aes, rsa, hmac, sha256, bcrypt, argon2, digital signature, key management, crypto, symmetric encryption, asymmetric encryption]
---

# Cryptography Basics Skill

Production-grade cryptographic patterns including hashing, symmetric/asymmetric encryption, digital signatures, and secure key management.

## Overview

Cryptography provides confidentiality (encryption), integrity (hashing), and authenticity (signatures). Use vetted libraries (cryptography.io, NaCl) - never roll your own crypto.

**When to Use**:
- Password storage (hashing with bcrypt/argon2)
- Data encryption (files, database fields)
- Message integrity (HMAC)
- Digital signatures (JWT, API authentication)
- Secure communication (TLS, encrypted messaging)

## Core Concepts

### Cryptographic Primitives

```yaml
Hashing (One-Way):
  Purpose: Password storage, integrity checks
  Algorithms: SHA-256, SHA-512, BLAKE2
  Password-Specific: bcrypt, argon2, scrypt
  Properties: Irreversible, deterministic

Symmetric Encryption (Same Key):
  Purpose: Data encryption (files, database)
  Algorithms: AES-256-GCM, ChaCha20-Poly1305
  Properties: Fast, requires secure key distribution

Asymmetric Encryption (Public/Private Key):
  Purpose: Secure communication, digital signatures
  Algorithms: RSA, ECC (Elliptic Curve), Ed25519
  Properties: Slow, enables public key infrastructure

Message Authentication:
  Purpose: Verify message integrity and authenticity
  Algorithms: HMAC-SHA256, Poly1305
  Properties: Shared secret, tamper detection
```

## Password Hashing

### bcrypt (Recommended)

```python
from passlib.context import CryptContext

# Configure password context
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Cost factor (higher = slower, more secure)
)

def hash_password(password: str) -> str:
    """Hash password with bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

# Usage
hashed = hash_password("my-secret-password")
# Output: "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5jtRFZCN9Z5Ja"

# Verify
is_valid = verify_password("my-secret-password", hashed)  # True
is_valid = verify_password("wrong-password", hashed)      # False
```

### argon2 (Modern Alternative)

```python
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

ph = PasswordHasher(
    time_cost=2,        # Number of iterations
    memory_cost=102400, # Memory in KiB (100 MB)
    parallelism=8,      # Number of threads
    hash_len=32,        # Hash length in bytes
    salt_len=16         # Salt length in bytes
)

def hash_password_argon2(password: str) -> str:
    """Hash password with Argon2"""
    return ph.hash(password)

def verify_password_argon2(password: str, hashed: str) -> bool:
    """Verify password against Argon2 hash"""
    try:
        ph.verify(hashed, password)
        return True
    except VerifyMismatchError:
        return False

# Usage
hashed = hash_password_argon2("my-secret-password")
# Output: "$argon2id$v=19$m=102400,t=2,p=8$..."
```

### Password Storage Best Practices

```python
from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)  # Never store plaintext

    def set_password(self, password: str):
        """Hash and store password"""
        self.password_hash = pwd_context.hash(password)

    def check_password(self, password: str) -> bool:
        """Verify password"""
        return pwd_context.verify(password, self.password_hash)

# Usage
user = User(email="alice@example.com")
user.set_password("secure-password")

# Later: verify login
if user.check_password("entered-password"):
    # Login successful
    pass
```

## Symmetric Encryption (AES)

### AES-GCM (Authenticated Encryption)

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import os
import base64

class AESEncryption:
    """AES-256-GCM encryption/decryption"""

    @staticmethod
    def generate_key() -> bytes:
        """Generate random 256-bit key"""
        return AESGCM.generate_key(bit_length=256)

    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """Derive key from password using PBKDF2"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())

    @staticmethod
    def encrypt(plaintext: bytes, key: bytes) -> dict:
        """
        Encrypt data with AES-256-GCM
        Returns: {ciphertext, nonce} (both base64-encoded)
        """
        aesgcm = AESGCM(key)

        # Generate random nonce (96 bits for GCM)
        nonce = os.urandom(12)

        # Encrypt (includes authentication tag)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        return {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "nonce": base64.b64encode(nonce).decode()
        }

    @staticmethod
    def decrypt(ciphertext: str, nonce: str, key: bytes) -> bytes:
        """
        Decrypt AES-256-GCM ciphertext
        Raises: InvalidTag if tampered
        """
        aesgcm = AESGCM(key)

        ciphertext_bytes = base64.b64decode(ciphertext)
        nonce_bytes = base64.b64decode(nonce)

        # Decrypt and verify authentication tag
        plaintext = aesgcm.decrypt(nonce_bytes, ciphertext_bytes, None)

        return plaintext

# Usage
aes = AESEncryption()

# Generate or derive key
key = aes.generate_key()
# Or derive from password:
# salt = os.urandom(16)
# key = aes.derive_key("my-password", salt)

# Encrypt
plaintext = b"Sensitive data that needs encryption"
encrypted = aes.encrypt(plaintext, key)
print(f"Ciphertext: {encrypted['ciphertext']}")
print(f"Nonce: {encrypted['nonce']}")

# Decrypt
decrypted = aes.decrypt(encrypted['ciphertext'], encrypted['nonce'], key)
assert decrypted == plaintext
```

### Fernet (High-Level API)

```python
from cryptography.fernet import Fernet

class FernetEncryption:
    """Symmetric encryption with Fernet (AES-128-CBC + HMAC)"""

    @staticmethod
    def generate_key() -> bytes:
        """Generate Fernet key"""
        return Fernet.generate_key()

    @staticmethod
    def encrypt(plaintext: bytes, key: bytes) -> bytes:
        """Encrypt with Fernet"""
        f = Fernet(key)
        return f.encrypt(plaintext)

    @staticmethod
    def decrypt(ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt with Fernet"""
        f = Fernet(key)
        return f.decrypt(ciphertext)

# Usage (simpler than raw AES)
key = Fernet.generate_key()
f = Fernet(key)

plaintext = b"Secret message"
ciphertext = f.encrypt(plaintext)
decrypted = f.decrypt(ciphertext)

assert decrypted == plaintext
```

## Asymmetric Encryption (RSA)

### RSA Key Generation and Encryption

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

class RSAEncryption:
    """RSA asymmetric encryption"""

    @staticmethod
    def generate_keypair():
        """Generate RSA public/private key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048  # 2048-bit (minimum), 4096-bit (recommended)
        )
        public_key = private_key.public_key()

        return private_key, public_key

    @staticmethod
    def encrypt(plaintext: bytes, public_key) -> bytes:
        """Encrypt with RSA public key"""
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return ciphertext

    @staticmethod
    def decrypt(ciphertext: bytes, private_key) -> bytes:
        """Decrypt with RSA private key"""
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext

# Usage
rsa_enc = RSAEncryption()

# Generate keys
private_key, public_key = rsa_enc.generate_keypair()

# Encrypt with public key
plaintext = b"Secret message"
ciphertext = rsa_enc.encrypt(plaintext, public_key)

# Decrypt with private key
decrypted = rsa_enc.decrypt(ciphertext, private_key)
assert decrypted == plaintext

# Save keys to files
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.BestAvailableEncryption(b"password")
)

public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

Path("private_key.pem").write_bytes(private_pem)
Path("public_key.pem").write_bytes(public_pem)
```

## Digital Signatures

### RSA Signatures

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

class RSASignature:
    """RSA digital signatures"""

    @staticmethod
    def sign(message: bytes, private_key) -> bytes:
        """Sign message with private key"""
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    @staticmethod
    def verify(message: bytes, signature: bytes, public_key) -> bool:
        """Verify signature with public key"""
        try:
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

# Usage
rsa_sig = RSASignature()

# Generate keys
private_key, public_key = RSAEncryption.generate_keypair()

# Sign message
message = b"Important document"
signature = rsa_sig.sign(message, private_key)

# Verify signature
is_valid = rsa_sig.verify(message, signature, public_key)  # True

# Tampered message
is_valid = rsa_sig.verify(b"Tampered document", signature, public_key)  # False
```

### HMAC (Symmetric Authentication)

```python
import hmac
import hashlib

def generate_hmac(message: bytes, secret_key: bytes) -> bytes:
    """Generate HMAC-SHA256"""
    return hmac.new(secret_key, message, hashlib.sha256).digest()

def verify_hmac(message: bytes, signature: bytes, secret_key: bytes) -> bool:
    """Verify HMAC signature"""
    expected = generate_hmac(message, secret_key)
    return hmac.compare_digest(expected, signature)  # Constant-time comparison

# Usage
secret_key = os.urandom(32)
message = b"Authenticated message"

# Generate HMAC
signature = generate_hmac(message, secret_key)

# Verify
is_valid = verify_hmac(message, signature, secret_key)  # True
is_valid = verify_hmac(b"Tampered", signature, secret_key)  # False
```

## Hashing (Integrity)

### SHA-256 Hashing

```python
import hashlib

def hash_data(data: bytes) -> str:
    """Hash data with SHA-256"""
    return hashlib.sha256(data).hexdigest()

def hash_file(file_path: str) -> str:
    """Hash file contents with SHA-256"""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

# Usage
data = b"Some data to hash"
hash_value = hash_data(data)
# Output: "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"

# Verify integrity
received_data = b"Some data to hash"
if hash_data(received_data) == hash_value:
    print("Data integrity verified")
```

## Key Management

### Secure Key Storage

```python
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives import hashes
import os
import json

class KeyManager:
    """Secure key derivation and storage"""

    def __init__(self, master_password: str):
        self.master_password = master_password

    def derive_key(self, salt: bytes, purpose: str) -> bytes:
        """Derive key from master password"""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive((self.master_password + purpose).encode())

    def encrypt_key(self, key_to_encrypt: bytes) -> dict:
        """Encrypt key with master password"""
        salt = os.urandom(16)
        encryption_key = self.derive_key(salt, "key_encryption")

        f = Fernet(base64.urlsafe_b64encode(encryption_key))
        encrypted_key = f.encrypt(key_to_encrypt)

        return {
            "encrypted_key": base64.b64encode(encrypted_key).decode(),
            "salt": base64.b64encode(salt).decode()
        }

    def decrypt_key(self, encrypted_data: dict) -> bytes:
        """Decrypt key with master password"""
        salt = base64.b64decode(encrypted_data["salt"])
        encryption_key = self.derive_key(salt, "key_encryption")

        f = Fernet(base64.urlsafe_b64encode(encryption_key))
        encrypted_key = base64.b64decode(encrypted_data["encrypted_key"])

        return f.decrypt(encrypted_key)

# Usage
km = KeyManager(master_password="super-secret-master-password")

# Encrypt sensitive key
sensitive_key = os.urandom(32)
encrypted = km.encrypt_key(sensitive_key)

# Store encrypted key in database/file
Path("encrypted_key.json").write_text(json.dumps(encrypted))

# Later: decrypt key
encrypted_data = json.loads(Path("encrypted_key.json").read_text())
decrypted_key = km.decrypt_key(encrypted_data)

assert decrypted_key == sensitive_key
```

### Key Rotation

```python
from datetime import datetime, timedelta

class RotatingKeyManager:
    """Automatic key rotation"""

    def __init__(self):
        self.current_key = None
        self.previous_key = None
        self.rotation_interval = timedelta(days=90)
        self.last_rotation = None

    def rotate_key(self):
        """Rotate to new key"""
        self.previous_key = self.current_key
        self.current_key = Fernet.generate_key()
        self.last_rotation = datetime.utcnow()

    def should_rotate(self) -> bool:
        """Check if rotation is due"""
        if not self.last_rotation:
            return True
        return datetime.utcnow() - self.last_rotation > self.rotation_interval

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt with current key"""
        if self.should_rotate():
            self.rotate_key()

        f = Fernet(self.current_key)
        return f.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt with current or previous key"""
        # Try current key
        try:
            f = Fernet(self.current_key)
            return f.decrypt(ciphertext)
        except Exception:
            pass

        # Try previous key (during rotation period)
        if self.previous_key:
            try:
                f = Fernet(self.previous_key)
                return f.decrypt(ciphertext)
            except Exception:
                pass

        raise ValueError("Decryption failed with all keys")
```

## Best Practices

### 1. Use Password Hashing for Passwords (Not Encryption)
```python
# ❌ Encrypt password (reversible)
encrypted_password = encrypt(password)

# ✅ Hash password (irreversible)
hashed_password = bcrypt.hash(password)
```

### 2. Always Use Authenticated Encryption
```python
# ❌ AES-CBC without authentication (malleable)
# ✅ AES-GCM (encryption + authentication)
aesgcm = AESGCM(key)
```

### 3. Generate Random Keys Securely
```python
# ❌ Weak randomness
key = str(random.randint(0, 1000000))

# ✅ Cryptographically secure random
key = os.urandom(32)  # 256 bits
```

### 4. Constant-Time Comparisons
```python
# ❌ Timing attack vulnerable
if signature == expected_signature:

# ✅ Constant-time comparison
if hmac.compare_digest(signature, expected_signature):
```

### 5. Rotate Keys Regularly
```python
# ✅ Implement key rotation (90 days for encryption keys)
if key_age > timedelta(days=90):
    rotate_key()
```

### 6. Store Keys Securely
```python
# ❌ Hardcode key in code
KEY = b"my-secret-key"

# ✅ Load from secret store
KEY = vault.get_secret("encryption_key")
```

## Common Pitfalls

❌ **Rolling your own crypto**
```python
# ❌ Custom encryption algorithm
# ✅ Use vetted libraries (cryptography.io)
```

❌ **Using ECB mode**
```python
# ❌ AES-ECB (insecure, reveals patterns)
# ✅ AES-GCM or AES-CBC with authentication
```

❌ **Reusing nonces/IVs**
```python
# ❌ Static nonce
nonce = b"12345678"

# ✅ Random nonce for each encryption
nonce = os.urandom(12)
```

❌ **Not authenticating ciphertext**
```python
# ❌ Encryption without authentication (malleable)
# ✅ Use GCM mode or add HMAC
```

❌ **Weak password hashing**
```python
# ❌ Plain SHA-256 (too fast, no salt)
hash = hashlib.sha256(password.encode()).hexdigest()

# ✅ bcrypt/argon2 (slow, salted, key stretching)
hash = pwd_context.hash(password)
```

## Quality Standards

- **Password Hashing**: bcrypt (cost ≥12) or argon2
- **Symmetric Encryption**: AES-256-GCM
- **Asymmetric Encryption**: RSA-2048 minimum (RSA-4096 recommended)
- **Hashing**: SHA-256 or SHA-512
- **Key Size**: 256 bits minimum for symmetric keys
- **Key Rotation**: Every 90 days for data encryption keys

---

**Skill Type**: Security - Cryptography
**Complexity**: Advanced
**Typical Usage**: Activated when implementing secure data protection
**Performance**: AES-GCM encrypts at ~1 GB/s, RSA at ~1 MB/s
