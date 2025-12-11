---
name: auth-patterns
description: Load when user needs OAuth2, OIDC, JWT, session management, MFA, or authentication and authorization patterns
trigger_keywords: [oauth2, oidc, openid connect, jwt, json web token, authentication, authorization, session, mfa, multi-factor, oauth, bearer token, refresh token, access token]
---

# Authentication Patterns Skill

Production-grade authentication and authorization with OAuth2, OpenID Connect (OIDC), JWT, session management, and multi-factor authentication (MFA).

## Overview

Authentication (who you are) and authorization (what you can do) are foundational security patterns. Modern systems use OAuth2/OIDC for delegated access and JWT for stateless authentication.

**When to Use**:
- APIs requiring secure user authentication
- Third-party integrations (OAuth2 delegation)
- Microservices requiring stateless auth
- Mobile/SPA applications

## Core Concepts

### Authentication vs Authorization

```yaml
Authentication:
  Question: "Who are you?"
  Methods: Password, OAuth2, JWT, biometrics
  Validates: Identity
  Result: Authenticated user context

Authorization:
  Question: "What can you do?"
  Methods: RBAC, ABAC, claims-based
  Validates: Permissions
  Result: Allow/deny access to resources
```

### OAuth2 Flow Types

**Authorization Code (Most Secure)**:
```
1. Client redirects user to auth server
2. User authenticates, consents
3. Auth server returns authorization code
4. Client exchanges code for access token (backend)
5. Client uses access token for API calls
```

**PKCE (Public Clients - SPAs, Mobile)**:
```
1. Client generates code_verifier (random string)
2. Client creates code_challenge = hash(code_verifier)
3. Client requests auth with code_challenge
4. Auth server returns code
5. Client exchanges code + code_verifier for token
```

**Client Credentials (Service-to-Service)**:
```
1. Service requests token with client_id + client_secret
2. Auth server validates credentials
3. Returns access token
```

## JWT (JSON Web Tokens)

### JWT Structure

```
Header.Payload.Signature

Header (Base64):
{
  "alg": "HS256",
  "typ": "JWT"
}

Payload (Base64):
{
  "sub": "user123",
  "name": "Alice",
  "iat": 1640000000,
  "exp": 1640003600,
  "roles": ["user", "admin"]
}

Signature:
HMACSHA256(
  base64UrlEncode(header) + "." + base64UrlEncode(payload),
  secret
)
```

### FastAPI JWT Authentication

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
security = HTTPBearer()

# Configuration
SECRET_KEY = "your-secret-key-keep-this-safe"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class TokenData(BaseModel):
    user_id: str
    email: str
    roles: list[str]

class User(BaseModel):
    id: str
    email: str
    roles: list[str]

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire, "iat": datetime.utcnow()})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Validate JWT and extract user"""
    token = credentials.credentials

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        roles: list[str] = payload.get("roles", [])

        if user_id is None:
            raise credentials_exception

        return User(id=user_id, email=email, roles=roles)

    except JWTError:
        raise credentials_exception

# Protected endpoint
@app.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": f"Hello {current_user.email}", "roles": current_user.roles}

# Login endpoint
@app.post("/login")
async def login(email: str, password: str):
    # Validate credentials (check database)
    user = await authenticate_user(email, password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create access token
    access_token = create_access_token(
        data={
            "sub": user.id,
            "email": user.email,
            "roles": user.roles
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }
```

### Refresh Tokens

```python
from datetime import datetime, timedelta
import secrets

REFRESH_TOKEN_EXPIRE_DAYS = 30

class RefreshToken(BaseModel):
    token: str
    user_id: str
    expires_at: datetime
    revoked: bool = False

async def create_refresh_token(user_id: str) -> str:
    """Create long-lived refresh token"""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    # Store in database
    await db.save_refresh_token(
        token=token,
        user_id=user_id,
        expires_at=expires_at
    )

    return token

@app.post("/token/refresh")
async def refresh_access_token(refresh_token: str):
    """Exchange refresh token for new access token"""
    # Validate refresh token
    stored_token = await db.get_refresh_token(refresh_token)

    if not stored_token or stored_token.revoked:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if stored_token.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Refresh token expired")

    # Get user
    user = await db.get_user(stored_token.user_id)

    # Create new access token
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email, "roles": user.roles}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.post("/logout")
async def logout(
    refresh_token: str,
    current_user: User = Depends(get_current_user)
):
    """Revoke refresh token"""
    await db.revoke_refresh_token(refresh_token)
    return {"message": "Logged out successfully"}
```

## OAuth2 Implementation

### Authorization Code Flow with PKCE

```python
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import secrets
import hashlib
import base64
from urllib.parse import urlencode

app = FastAPI()

# OAuth2 Configuration
OAUTH_CLIENT_ID = "your-client-id"
OAUTH_CLIENT_SECRET = "your-client-secret"
OAUTH_REDIRECT_URI = "http://localhost:8000/callback"
OAUTH_AUTHORIZATION_URL = "https://provider.com/oauth/authorize"
OAUTH_TOKEN_URL = "https://provider.com/oauth/token"

def generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge"""
    # Generate random code_verifier
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')
    code_verifier = code_verifier.rstrip('=')

    # Generate code_challenge = BASE64URL(SHA256(code_verifier))
    code_challenge = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(code_challenge).decode('utf-8')
    code_challenge = code_challenge.rstrip('=')

    return code_verifier, code_challenge

@app.get("/login")
async def login(request: Request):
    """Initiate OAuth2 flow"""
    # Generate PKCE parameters
    code_verifier, code_challenge = generate_pkce_pair()

    # Store code_verifier in session (in production, use Redis/database)
    request.session["code_verifier"] = code_verifier

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state

    # Build authorization URL
    params = {
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid profile email",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256"
    }

    auth_url = f"{OAUTH_AUTHORIZATION_URL}?{urlencode(params)}"

    return RedirectResponse(url=auth_url)

@app.get("/callback")
async def oauth_callback(request: Request, code: str, state: str):
    """Handle OAuth2 callback"""
    # Validate state (CSRF protection)
    stored_state = request.session.get("oauth_state")
    if state != stored_state:
        raise HTTPException(status_code=400, detail="Invalid state")

    # Get code_verifier from session
    code_verifier = request.session.get("code_verifier")

    # Exchange authorization code for access token
    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "client_id": OAUTH_CLIENT_ID,
        "client_secret": OAUTH_CLIENT_SECRET,
        "code_verifier": code_verifier
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OAUTH_TOKEN_URL, data=token_data)

    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Token exchange failed")

    tokens = response.json()

    # Decode ID token (OpenID Connect)
    id_token = tokens["id_token"]
    user_info = jwt.decode(id_token, options={"verify_signature": False})

    # Create session or JWT for your application
    access_token = create_access_token(
        data={
            "sub": user_info["sub"],
            "email": user_info["email"],
            "name": user_info.get("name")
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_info
    }
```

## Role-Based Access Control (RBAC)

### Permission Checking

```python
from typing import List
from fastapi import Depends, HTTPException

class RoleChecker:
    """Dependency for role-based access control"""

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: User = Depends(get_current_user)):
        if not any(role in current_user.roles for role in self.allowed_roles):
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions"
            )
        return current_user

# Usage
admin_only = RoleChecker(allowed_roles=["admin"])
user_or_admin = RoleChecker(allowed_roles=["user", "admin"])

@app.delete("/users/{user_id}", dependencies=[Depends(admin_only)])
async def delete_user(user_id: str):
    """Admin-only endpoint"""
    await db.delete_user(user_id)
    return {"message": "User deleted"}

@app.get("/dashboard", dependencies=[Depends(user_or_admin)])
async def dashboard():
    """Requires user or admin role"""
    return {"data": "dashboard"}
```

### Resource-Based Authorization

```python
async def check_resource_owner(
    resource_id: str,
    current_user: User = Depends(get_current_user)
):
    """Check if user owns the resource"""
    resource = await db.get_resource(resource_id)

    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")

    if resource.owner_id != current_user.id and "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Not authorized")

    return resource

@app.put("/posts/{post_id}")
async def update_post(
    post_id: str,
    data: PostUpdate,
    post = Depends(check_resource_owner)
):
    """Only post owner or admin can update"""
    updated_post = await db.update_post(post_id, data)
    return updated_post
```

## Multi-Factor Authentication (MFA)

### TOTP (Time-Based One-Time Password)

```python
import pyotp
import qrcode
from io import BytesIO

@app.post("/mfa/enable")
async def enable_mfa(current_user: User = Depends(get_current_user)):
    """Generate MFA secret and QR code"""
    # Generate secret
    secret = pyotp.random_base32()

    # Create TOTP URI
    totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=current_user.email,
        issuer_name="YourApp"
    )

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(totp_uri)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    # Store secret (encrypted) in database
    await db.save_mfa_secret(current_user.id, secret)

    return {
        "secret": secret,
        "qr_code": base64.b64encode(buffer.getvalue()).decode()
    }

@app.post("/mfa/verify")
async def verify_mfa(
    code: str,
    current_user: User = Depends(get_current_user)
):
    """Verify TOTP code"""
    # Get stored secret
    mfa_secret = await db.get_mfa_secret(current_user.id)

    if not mfa_secret:
        raise HTTPException(status_code=400, detail="MFA not enabled")

    # Verify code
    totp = pyotp.TOTP(mfa_secret)
    if not totp.verify(code, valid_window=1):  # Allow 1 step tolerance
        raise HTTPException(status_code=401, detail="Invalid code")

    # Mark user as MFA verified
    await db.mark_mfa_verified(current_user.id)

    return {"message": "MFA verified successfully"}
```

### Email/SMS MFA

```python
import secrets
from datetime import datetime, timedelta

async def send_mfa_code(user_id: str, method: str = "email"):
    """Send 6-digit MFA code"""
    # Generate 6-digit code
    code = "".join([str(secrets.randbelow(10)) for _ in range(6)])

    # Store code with expiration
    expires_at = datetime.utcnow() + timedelta(minutes=5)
    await db.save_mfa_code(user_id, code, expires_at)

    # Send via email or SMS
    if method == "email":
        await email_service.send(user_id, f"Your code: {code}")
    else:
        await sms_service.send(user_id, f"Your code: {code}")

    return {"message": f"Code sent via {method}"}

@app.post("/mfa/send-code")
async def send_code(current_user: User = Depends(get_current_user)):
    return await send_mfa_code(current_user.id, method="email")

@app.post("/mfa/verify-code")
async def verify_code(code: str, current_user: User = Depends(get_current_user)):
    stored = await db.get_mfa_code(current_user.id)

    if not stored or stored.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Code expired")

    if stored.code != code:
        raise HTTPException(status_code=401, detail="Invalid code")

    await db.delete_mfa_code(current_user.id)
    return {"message": "Verified successfully"}
```

## Best Practices

### 1. Use HTTPS Only
```python
# ✅ Enforce HTTPS in production
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
app.add_middleware(HTTPSRedirectMiddleware)
```

### 2. Secure Token Storage
```python
# ❌ Store JWT in localStorage (vulnerable to XSS)
# ✅ Store in httpOnly cookie
response.set_cookie(
    key="access_token",
    value=token,
    httponly=True,
    secure=True,  # HTTPS only
    samesite="lax",
    max_age=1800
)
```

### 3. Short-Lived Access Tokens
```python
# ✅ Access token: 15-30 minutes
ACCESS_TOKEN_EXPIRE_MINUTES = 15

# ✅ Refresh token: 7-30 days
REFRESH_TOKEN_EXPIRE_DAYS = 7
```

### 4. Validate JWT Claims
```python
# ✅ Check expiration, issuer, audience
payload = jwt.decode(
    token,
    SECRET_KEY,
    algorithms=[ALGORITHM],
    options={
        "verify_exp": True,
        "verify_iss": True,
        "verify_aud": True
    },
    issuer="your-app",
    audience="api"
)
```

### 5. Rate Limit Login Attempts
```python
# ✅ Prevent brute force
@app.post("/login")
@rate_limit(max_attempts=5, window=300)  # 5 attempts per 5 minutes
async def login(email: str, password: str):
    ...
```

## Common Pitfalls

❌ **Storing passwords in plaintext**
```python
# ✅ Hash passwords with bcrypt/argon2
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
hashed = pwd_context.hash(password)
```

❌ **Not validating redirect URIs**
```python
# ❌ Open redirect vulnerability
# ✅ Whitelist allowed redirect URIs
ALLOWED_REDIRECTS = ["http://localhost:3000", "https://app.example.com"]

if redirect_uri not in ALLOWED_REDIRECTS:
    raise HTTPException(400, "Invalid redirect URI")
```

❌ **Using HS256 with public keys**
```python
# ❌ Symmetric key (HS256) for public APIs
# ✅ Use RS256 (asymmetric) for distributed systems
ALGORITHM = "RS256"  # Private key signs, public key verifies
```

❌ **Not revoking refresh tokens on logout**
```python
# ✅ Revoke tokens on logout
@app.post("/logout")
async def logout(refresh_token: str):
    await db.revoke_refresh_token(refresh_token)
```

## Quality Standards

- **Token Expiry**: Access tokens ≤30 min, refresh tokens ≤30 days
- **Password Hashing**: bcrypt/argon2 with proper cost factor
- **HTTPS**: Enforce TLS 1.2+ in production
- **MFA**: Require for admin/sensitive operations
- **Logging**: Log authentication attempts (success/failure)

---

**Skill Type**: Security - Authentication
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when implementing secure authentication flows
**Performance**: JWT validation adds <1ms overhead per request
