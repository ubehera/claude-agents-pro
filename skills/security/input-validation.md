---
name: input-validation
description: Load when user needs sanitization, allowlists, encoding, injection prevention, or input validation and sanitization patterns
trigger_keywords: [input validation, sanitization, sql injection, xss, cross site scripting, injection, allowlist, whitelist, encoding, escaping, validation, pydantic validation]
---

# Input Validation Skill

Production-grade input validation and sanitization patterns to prevent injection attacks (SQL, XSS, command injection) using allowlists, encoding, and type-safe validation.

## Overview

All user input is untrusted and must be validated, sanitized, and encoded before processing. Input validation prevents injection attacks, data corruption, and business logic bypasses.

**When to Use**:
- Accepting user-submitted data (forms, APIs, file uploads)
- Constructing database queries
- Rendering user content in HTML
- Executing system commands with user input

## Core Concepts

### Validation Hierarchy

```yaml
1. Type Validation:
   - Ensure correct data type (int, str, email)
   - Use Pydantic, JSON Schema, TypeScript

2. Format Validation:
   - Regex patterns (email, phone, UUID)
   - Length constraints (min/max)

3. Range Validation:
   - Numeric ranges (age: 0-150)
   - Date ranges (future/past)

4. Business Logic Validation:
   - Unique constraints (email uniqueness)
   - Foreign key existence
   - State transitions

5. Sanitization:
   - Remove/escape dangerous characters
   - Normalize input (trim whitespace, lowercase)

6. Encoding:
   - HTML encoding (prevent XSS)
   - URL encoding
   - SQL parameterization (prevent SQL injection)
```

## Type-Safe Validation (Pydantic)

### Basic Validation

```python
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import date

class UserCreate(BaseModel):
    """Type-safe user creation input"""

    email: EmailStr  # Validates email format
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)  # 0 <= age <= 150
    phone: Optional[str] = Field(None, regex=r"^\+?1?\d{10,15}$")
    birth_date: Optional[date] = None

    @validator("name")
    def name_must_not_contain_special_chars(cls, v):
        """Custom validator: no special characters in name"""
        if not v.replace(" ", "").isalnum():
            raise ValueError("Name must contain only letters and spaces")
        return v.strip()

    @validator("birth_date")
    def birth_date_must_be_past(cls, v):
        """Birth date must be in the past"""
        if v and v > date.today():
            raise ValueError("Birth date must be in the past")
        return v

# Usage
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.post("/users")
async def create_user(user: UserCreate):
    """Pydantic automatically validates input"""
    # If validation fails, FastAPI returns 422 Unprocessable Entity
    # with detailed error messages

    # user.email is guaranteed to be valid email
    # user.age is guaranteed to be 0-150
    return {"user": user}

# Invalid input example:
# POST /users
# {
#   "email": "invalid-email",  # ❌ Not an email
#   "name": "A",
#   "age": 200  # ❌ Out of range
# }
# Response: 422 with validation errors
```

### Advanced Validation Patterns

```python
from pydantic import BaseModel, root_validator, constr
from typing import List

class PasswordReset(BaseModel):
    """Password reset with confirmation"""

    password: constr(min_length=8, max_length=100)
    confirm_password: str

    @root_validator
    def passwords_match(cls, values):
        """Ensure password and confirmation match"""
        password = values.get("password")
        confirm = values.get("confirm_password")

        if password != confirm:
            raise ValueError("Passwords do not match")

        return values

    @validator("password")
    def password_strength(cls, v):
        """Enforce password complexity"""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        if not any(c in "!@#$%^&*" for c in v):
            raise ValueError("Password must contain special character")
        return v

class SearchQuery(BaseModel):
    """Search with validation"""

    query: constr(min_length=1, max_length=500)
    filters: List[str] = Field(default=[], max_items=10)
    page: int = Field(default=1, ge=1, le=1000)
    page_size: int = Field(default=20, ge=1, le=100)

    @validator("query")
    def sanitize_query(cls, v):
        """Remove potentially dangerous characters"""
        # Allow only alphanumeric, spaces, and basic punctuation
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-")
        sanitized = "".join(c for c in v if c in allowed_chars)
        return sanitized.strip()
```

## SQL Injection Prevention

### Parameterized Queries (Recommended)

```python
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# ❌ NEVER: String concatenation (SQL injection vulnerable)
async def get_user_unsafe(email: str):
    query = f"SELECT * FROM users WHERE email = '{email}'"
    # Attacker input: "' OR '1'='1" → Bypasses authentication
    result = await db.execute(query)

# ✅ ALWAYS: Parameterized queries
async def get_user_safe(session: AsyncSession, email: str):
    """SQL injection safe: parameters are escaped"""
    query = text("SELECT * FROM users WHERE email = :email")
    result = await session.execute(query, {"email": email})
    return result.first()

# ✅ ORM (SQLAlchemy) - automatically parameterized
async def get_user_orm(session: AsyncSession, email: str):
    """ORM prevents SQL injection"""
    from models import User
    stmt = select(User).where(User.email == email)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
```

### Query Builder Pattern

```python
from typing import Optional, List

class QueryBuilder:
    """Safe query builder with parameterization"""

    def __init__(self):
        self.query_parts = []
        self.params = {}

    def select(self, table: str, columns: List[str]):
        # Allowlist table and column names
        allowed_tables = {"users", "posts", "comments"}
        allowed_columns = {"id", "email", "name", "created_at"}

        if table not in allowed_tables:
            raise ValueError(f"Invalid table: {table}")

        for col in columns:
            if col not in allowed_columns:
                raise ValueError(f"Invalid column: {col}")

        self.query_parts.append(f"SELECT {', '.join(columns)} FROM {table}")
        return self

    def where(self, field: str, value: Any):
        """Add WHERE clause with parameter"""
        param_name = f"param_{len(self.params)}"
        self.query_parts.append(f"WHERE {field} = :{param_name}")
        self.params[param_name] = value
        return self

    def build(self):
        """Return safe query and parameters"""
        return " ".join(self.query_parts), self.params

# Usage
builder = QueryBuilder()
query, params = builder.select("users", ["id", "email"]).where("email", "user@example.com").build()

# query: "SELECT id, email FROM users WHERE email = :param_0"
# params: {"param_0": "user@example.com"}

result = await session.execute(text(query), params)
```

## XSS (Cross-Site Scripting) Prevention

### HTML Encoding

```python
import html
from markupsafe import escape

# ❌ Directly rendering user input (XSS vulnerable)
def render_unsafe(user_input: str) -> str:
    return f"<div>{user_input}</div>"
    # Attacker input: "<script>alert('XSS')</script>"

# ✅ HTML escape user input
def render_safe(user_input: str) -> str:
    escaped = html.escape(user_input)
    return f"<div>{escaped}</div>"
    # Output: "<div>&lt;script&gt;alert('XSS')&lt;/script&gt;</div>"

# ✅ MarkupSafe (Jinja2)
def render_template(user_input: str) -> str:
    """Automatic escaping in templates"""
    from jinja2 import Template

    template = Template("<div>{{ user_input }}</div>")
    return template.render(user_input=user_input)  # Auto-escaped

# ✅ FastAPI (automatic escaping in responses)
from fastapi.responses import HTMLResponse

@app.get("/profile")
async def get_profile(name: str):
    # FastAPI automatically escapes HTML
    return HTMLResponse(f"<h1>Welcome {name}</h1>")
```

### Content Security Policy (CSP)

```python
from fastapi import Response

@app.get("/page")
async def get_page(response: Response):
    """Add CSP header to prevent inline scripts"""
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://trusted-cdn.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self' https://api.example.com; "
        "frame-ancestors 'none';"
    )

    return {"content": "..."}
```

### Sanitize Rich Text (Markdown/HTML)

```python
import bleach
from markdown import markdown

ALLOWED_TAGS = [
    'p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'code', 'pre'
]

ALLOWED_ATTRIBUTES = {
    'a': ['href', 'title'],
    'img': ['src', 'alt']
}

def sanitize_html(user_html: str) -> str:
    """Remove dangerous HTML tags and attributes"""
    cleaned = bleach.clean(
        user_html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        strip=True  # Remove disallowed tags entirely
    )
    return cleaned

def safe_markdown(user_markdown: str) -> str:
    """Convert Markdown to HTML safely"""
    # Convert Markdown to HTML
    html_output = markdown(user_markdown)

    # Sanitize HTML
    return sanitize_html(html_output)

# Usage
user_input = "<script>alert('XSS')</script><p>Safe content</p>"
safe_output = sanitize_html(user_input)
# Output: "<p>Safe content</p>"
```

## Command Injection Prevention

### Never Use `os.system` or `subprocess.shell=True`

```python
import subprocess
from pathlib import Path

# ❌ DANGEROUS: Shell injection vulnerable
def run_command_unsafe(filename: str):
    os.system(f"cat {filename}")  # ❌ Never do this
    # Attacker input: "file.txt; rm -rf /"

# ✅ SAFE: Use subprocess with list arguments
def run_command_safe(filename: str):
    """Execute command safely without shell"""
    # Validate filename first
    if not Path(filename).is_file():
        raise ValueError("Invalid file")

    # Use list arguments (no shell interpretation)
    result = subprocess.run(
        ["cat", filename],  # ✅ List arguments
        capture_output=True,
        text=True,
        shell=False,  # ✅ Disable shell
        timeout=10
    )

    return result.stdout

# ✅ Allowlist approach
ALLOWED_COMMANDS = {"ls", "cat", "grep"}

def run_allowed_command(command: str, args: List[str]):
    """Only allow specific commands"""
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"Command not allowed: {command}")

    # Validate arguments (no special characters)
    for arg in args:
        if not arg.replace(".", "").replace("/", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid argument: {arg}")

    result = subprocess.run(
        [command] + args,
        capture_output=True,
        shell=False,
        timeout=10
    )

    return result.stdout
```

## File Upload Validation

### Secure File Uploads

```python
from fastapi import UploadFile, HTTPException
from pathlib import Path
import magic  # python-magic library

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".pdf"}
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "application/pdf"
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

async def validate_upload(file: UploadFile):
    """Comprehensive file upload validation"""

    # 1. Check file extension
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type not allowed: {extension}")

    # 2. Check MIME type (from Content-Type header)
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, f"MIME type not allowed: {file.content_type}")

    # 3. Read file content
    content = await file.read()
    await file.seek(0)  # Reset file pointer

    # 4. Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large: {len(content)} bytes")

    # 5. Validate actual file type (magic numbers)
    mime_type = magic.from_buffer(content, mime=True)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(400, f"Invalid file format: {mime_type}")

    # 6. Sanitize filename
    safe_filename = sanitize_filename(file.filename)

    return safe_filename, content

def sanitize_filename(filename: str) -> str:
    """Remove dangerous characters from filename"""
    # Remove path traversal attempts
    filename = Path(filename).name

    # Allow only alphanumeric, dash, underscore, dot
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    safe_name = "".join(c for c in filename if c in allowed_chars)

    # Ensure extension is preserved
    if not safe_name:
        safe_name = "upload.bin"

    return safe_name

@app.post("/upload")
async def upload_file(file: UploadFile):
    """Secure file upload endpoint"""
    safe_filename, content = await validate_upload(file)

    # Generate unique filename to prevent overwrites
    unique_filename = f"{uuid.uuid4()}_{safe_filename}"

    # Save to safe directory (outside web root)
    upload_path = Path("/var/uploads") / unique_filename
    upload_path.write_bytes(content)

    return {"filename": unique_filename}
```

## Advanced Patterns

### Allowlist Validation

```python
from enum import Enum

class UserRole(str, Enum):
    """Allowlist of valid roles"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class UpdateUser(BaseModel):
    role: UserRole  # Only accepts values from enum

# ❌ String validation (accepts any string)
class UpdateUserUnsafe(BaseModel):
    role: str  # Can be anything

# ✅ Enum validation (allowlist)
@app.put("/users/{user_id}")
async def update_user(user_id: int, data: UpdateUser):
    # data.role is guaranteed to be "admin", "user", or "guest"
    return {"role": data.role.value}
```

### Rate Limiting (Validation)

```python
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/data")
@limiter.limit("5/minute")  # Max 5 requests per minute
async def rate_limited_endpoint(request: Request):
    """Prevent abuse through rate limiting"""
    return {"data": "..."}
```

## Best Practices

### 1. Validate at API Boundary
```python
# ✅ Validate input immediately
@app.post("/users")
async def create_user(user: UserCreate):  # Pydantic validates here
    # user is already validated
    return await db.save(user)
```

### 2. Use Type-Safe Models
```python
# ✅ Pydantic models ensure type safety
class UserUpdate(BaseModel):
    email: EmailStr
    age: int = Field(ge=0, le=150)
```

### 3. Allowlist, Don't Blocklist
```python
# ❌ Blocklist (easy to bypass)
if "<script>" in user_input:
    raise ValueError("XSS detected")

# ✅ Allowlist (only allow known-safe)
ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789 ")
sanitized = "".join(c for c in user_input if c.lower() in ALLOWED_CHARS)
```

### 4. Encode Output, Not Input
```python
# ❌ Store escaped data (causes double-escaping issues)
user.name = html.escape(user_input)

# ✅ Store raw, escape on output
user.name = user_input  # Store as-is
display_name = html.escape(user.name)  # Escape when rendering
```

### 5. Use Prepared Statements Always
```python
# ✅ ALWAYS use parameterized queries
stmt = select(User).where(User.email == email)  # Safe
result = await session.execute(stmt)
```

## Common Pitfalls

❌ **Trusting client-side validation**
```javascript
// ❌ Client-side only (easily bypassed)
if (email.includes("@")) { submit(); }

// ✅ Always validate server-side
```

❌ **Regex injection**
```python
# ❌ User-controlled regex
pattern = user_input  # Attacker: ".*" (DoS)
re.search(pattern, text)

# ✅ Predefined patterns only
```

❌ **Path traversal**
```python
# ❌ Allows ../../etc/passwd
file_path = f"/uploads/{user_filename}"

# ✅ Sanitize filename
safe_name = Path(user_filename).name  # Removes ../
file_path = f"/uploads/{safe_name}"
```

❌ **Not validating MIME types**
```python
# ❌ Check extension only
if filename.endswith(".jpg"):  # Attacker: "shell.php.jpg"

# ✅ Check magic numbers
mime = magic.from_buffer(content, mime=True)
```

## Quality Standards

- **Type Safety**: Use Pydantic/TypeScript for all inputs
- **SQL**: Parameterized queries 100% of the time
- **XSS**: HTML-escape all user content in templates
- **Files**: Validate extension, MIME type, and magic numbers
- **Commands**: Never use shell=True, allowlist commands

---

**Skill Type**: Security - Input Validation
**Complexity**: Moderate
**Typical Usage**: Activated when validating user input and preventing injection
**Performance**: Pydantic validation adds <1ms overhead per request
