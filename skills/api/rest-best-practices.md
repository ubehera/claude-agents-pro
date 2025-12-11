---
name: rest-best-practices
description: Load when user needs REST API design patterns, HTTP methods, status codes, versioning, pagination, rate limiting, authentication, or error handling
trigger_keywords: [rest, rest api, restful, http methods, get, post, put, patch, delete, status code, api design, api versioning, pagination, rate limit, jwt, oauth]
---

# REST API Best Practices

Production-grade REST API design patterns following industry standards, focusing on developer experience, scalability, and maintainability.

## Core Concepts

### REST Principles

**Resource-Oriented Design**:
- URLs represent resources (nouns), not actions (verbs)
- Use HTTP methods to define operations
- Stateless communication
- Uniform interface

**HTTP Methods** (CRUD Mapping):
- `GET`: Retrieve resource(s) - Safe, Idempotent, Cacheable
- `POST`: Create new resource - Not idempotent
- `PUT`: Replace entire resource - Idempotent
- `PATCH`: Partial update - Idempotent (usually)
- `DELETE`: Remove resource - Idempotent

### URL Design Best Practices

```
✅ GOOD: Resource-oriented URLs
GET    /api/v1/users              # List users
GET    /api/v1/users/123          # Get user 123
POST   /api/v1/users              # Create user
PUT    /api/v1/users/123          # Replace user 123
PATCH  /api/v1/users/123          # Update user 123
DELETE /api/v1/users/123          # Delete user 123

GET    /api/v1/users/123/orders   # Get orders for user 123
POST   /api/v1/users/123/orders   # Create order for user 123

❌ BAD: Action-oriented URLs
GET    /api/getUsers
POST   /api/createUser
POST   /api/updateUser/123
POST   /api/deleteUser/123
```

## HTTP Status Codes

### Success (2xx)

```
200 OK                  - GET, PUT, PATCH succeeded
201 Created             - POST created new resource (include Location header)
202 Accepted            - Request accepted for async processing
204 No Content          - DELETE succeeded, no response body
```

### Client Errors (4xx)

```
400 Bad Request         - Invalid request body/parameters
401 Unauthorized        - Authentication required
403 Forbidden           - Authenticated but not authorized
404 Not Found           - Resource doesn't exist
405 Method Not Allowed  - HTTP method not supported
409 Conflict            - Request conflicts with current state
422 Unprocessable Entity - Validation errors
429 Too Many Requests   - Rate limit exceeded
```

### Server Errors (5xx)

```
500 Internal Server Error - Unexpected server error
502 Bad Gateway          - Invalid upstream response
503 Service Unavailable  - Temporary overload/maintenance
504 Gateway Timeout      - Upstream timeout
```

## Implementation Patterns

### 1. Consistent Error Response Format

```typescript
// Error response schema
interface ErrorResponse {
  error: {
    code: string;           // Machine-readable error code
    message: string;        // Human-readable message
    details?: object;       // Additional context
    timestamp: string;      // ISO 8601 timestamp
    path: string;          // Request path
    request_id: string;    // Trace ID for debugging
  };
}

// Example error responses
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "email": ["Email is required", "Email format is invalid"],
      "age": ["Age must be at least 18"]
    },
    "timestamp": "2025-12-10T10:30:00Z",
    "path": "/api/v1/users",
    "request_id": "req_abc123xyz"
  }
}

{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "User with ID 999 not found",
    "timestamp": "2025-12-10T10:30:00Z",
    "path": "/api/v1/users/999",
    "request_id": "req_def456uvw"
  }
}
```

### 2. Pagination Patterns

**Cursor-Based Pagination** (Recommended for large datasets):

```typescript
// Request
GET /api/v1/users?limit=20&cursor=eyJpZCI6MTAwfQ==

// Response
{
  "data": [
    { "id": 101, "name": "Alice" },
    { "id": 102, "name": "Bob" }
    // ... 20 items
  ],
  "pagination": {
    "next_cursor": "eyJpZCI6MTIwfQ==",
    "prev_cursor": "eyJpZCI6MTAwfQ==",
    "has_more": true,
    "limit": 20
  }
}

// Implementation
interface PaginationParams {
  limit: number;
  cursor?: string;  // Base64-encoded JSON with last item's sort key
}

function decodeCursor(cursor: string): { id: number } {
  return JSON.parse(Buffer.from(cursor, 'base64').toString());
}

function encodeCursor(id: number): string {
  return Buffer.from(JSON.stringify({ id })).toString('base64');
}
```

**Offset-Based Pagination** (Simple, but slow for large offsets):

```typescript
// Request
GET /api/v1/users?page=2&per_page=20

// Response
{
  "data": [...],
  "pagination": {
    "page": 2,
    "per_page": 20,
    "total_pages": 50,
    "total_count": 1000
  },
  "links": {
    "self": "/api/v1/users?page=2&per_page=20",
    "first": "/api/v1/users?page=1&per_page=20",
    "prev": "/api/v1/users?page=1&per_page=20",
    "next": "/api/v1/users?page=3&per_page=20",
    "last": "/api/v1/users?page=50&per_page=20"
  }
}
```

### 3. Filtering, Sorting, and Field Selection

```typescript
// Filtering (query parameters)
GET /api/v1/users?status=active&role=admin&created_after=2025-01-01

// Sorting
GET /api/v1/users?sort=-created_at,name  // Descending created_at, ascending name

// Field selection (sparse fieldsets)
GET /api/v1/users?fields=id,name,email  // Only return specified fields

// Combined
GET /api/v1/users?status=active&sort=-created_at&fields=id,name&limit=20

// Implementation
interface QueryParams {
  status?: string;
  role?: string;
  created_after?: string;
  sort?: string;     // Format: "-field1,field2" (- prefix for descending)
  fields?: string;   // Format: "field1,field2,field3"
  limit?: number;
  cursor?: string;
}

function parseSort(sortParam: string): Array<{ field: string; order: 'ASC' | 'DESC' }> {
  return sortParam.split(',').map(s => ({
    field: s.startsWith('-') ? s.substring(1) : s,
    order: s.startsWith('-') ? 'DESC' : 'ASC'
  }));
}
```

### 4. API Versioning

**URL Versioning** (Most common, explicit):

```typescript
// v1 API
GET /api/v1/users/123

// v2 API with breaking changes
GET /api/v2/users/123

// Express.js implementation
app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);
```

**Header Versioning** (Cleaner URLs):

```http
GET /api/users/123
Accept: application/vnd.myapp.v2+json

HTTP/1.1 200 OK
Content-Type: application/vnd.myapp.v2+json
```

**Query Parameter Versioning** (Avoid - cache issues):

```
❌ Not recommended
GET /api/users/123?version=2
```

### 5. Rate Limiting

**Response Headers**:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000        # Total requests allowed per window
X-RateLimit-Remaining: 987     # Requests remaining in current window
X-RateLimit-Reset: 1702209600  # Unix timestamp when limit resets

HTTP/1.1 429 Too Many Requests
Retry-After: 60                # Seconds until retry allowed
```

**Implementation** (Express middleware):

```typescript
import rateLimit from 'express-rate-limit';

const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 100,                   // Max 100 requests per windowMs
  standardHeaders: true,      // Return rate limit info in headers
  legacyHeaders: false,
  message: {
    error: {
      code: 'RATE_LIMIT_EXCEEDED',
      message: 'Too many requests, please try again later'
    }
  }
});

app.use('/api/', apiLimiter);
```

### 6. Authentication Patterns

**JWT Bearer Tokens**:

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securePassword123"
}

HTTP/1.1 200 OK
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600  # seconds
}

# Subsequent requests
GET /api/v1/users/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

**API Keys** (For service-to-service):

```http
GET /api/v1/data
X-API-Key: sk_live_abc123xyz456
```

**OAuth 2.0** (For third-party integrations):

```http
# Authorization Code Flow
GET /oauth/authorize?client_id=CLIENT_ID&redirect_uri=CALLBACK&response_type=code&scope=read:users

# Token exchange
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code=AUTH_CODE&client_id=CLIENT_ID&client_secret=SECRET
```

### 7. Request/Response Examples

**POST - Create Resource**:

```http
POST /api/v1/users
Content-Type: application/json
Authorization: Bearer token123

{
  "name": "Alice Smith",
  "email": "alice@example.com",
  "role": "admin"
}

HTTP/1.1 201 Created
Location: /api/v1/users/123
Content-Type: application/json

{
  "id": 123,
  "name": "Alice Smith",
  "email": "alice@example.com",
  "role": "admin",
  "created_at": "2025-12-10T10:30:00Z",
  "updated_at": "2025-12-10T10:30:00Z"
}
```

**PUT - Replace Resource**:

```http
PUT /api/v1/users/123
Content-Type: application/json

{
  "name": "Alice Johnson",  # Full replacement
  "email": "alice.j@example.com",
  "role": "admin"
}

HTTP/1.1 200 OK
{
  "id": 123,
  "name": "Alice Johnson",
  "email": "alice.j@example.com",
  "role": "admin",
  "updated_at": "2025-12-10T11:00:00Z"
}
```

**PATCH - Partial Update**:

```http
PATCH /api/v1/users/123
Content-Type: application/json

{
  "name": "Alice Cooper"  # Only update name
}

HTTP/1.1 200 OK
{
  "id": 123,
  "name": "Alice Cooper",
  "email": "alice.j@example.com",  # Unchanged
  "role": "admin",                 # Unchanged
  "updated_at": "2025-12-10T11:30:00Z"
}
```

## Production-Ready API Framework

### Express.js REST API Template

```typescript
import express, { Request, Response, NextFunction } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';

const app = express();

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
  credentials: true
}));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100
});
app.use('/api/', limiter);

// Request ID middleware
app.use((req: Request, res: Response, next: NextFunction) => {
  req.id = req.headers['x-request-id'] || generateRequestId();
  res.setHeader('X-Request-ID', req.id);
  next();
});

// API routes
app.use('/api/v1/users', usersRouter);
app.use('/api/v1/orders', ordersRouter);

// 404 handler
app.use((req: Request, res: Response) => {
  res.status(404).json({
    error: {
      code: 'NOT_FOUND',
      message: `Route ${req.method} ${req.path} not found`,
      timestamp: new Date().toISOString(),
      path: req.path,
      request_id: req.id
    }
  });
});

// Global error handler
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error('Error:', err);

  const statusCode = err.statusCode || 500;
  const errorCode = err.code || 'INTERNAL_SERVER_ERROR';

  res.status(statusCode).json({
    error: {
      code: errorCode,
      message: err.message || 'Internal server error',
      timestamp: new Date().toISOString(),
      path: req.path,
      request_id: req.id
    }
  });
});

export default app;
```

### Controller Pattern

```typescript
import { Request, Response, NextFunction } from 'express';

class UserController {
  async list(req: Request, res: Response, next: NextFunction) {
    try {
      const { page = 1, per_page = 20, status, sort } = req.query;

      const users = await userService.findAll({
        page: Number(page),
        perPage: Number(per_page),
        status: status as string,
        sort: sort as string
      });

      res.status(200).json({
        data: users.items,
        pagination: {
          page: users.page,
          per_page: users.perPage,
          total_pages: users.totalPages,
          total_count: users.totalCount
        }
      });
    } catch (error) {
      next(error);
    }
  }

  async get(req: Request, res: Response, next: NextFunction) {
    try {
      const { id } = req.params;
      const user = await userService.findById(id);

      if (!user) {
        return res.status(404).json({
          error: {
            code: 'USER_NOT_FOUND',
            message: `User with ID ${id} not found`
          }
        });
      }

      res.status(200).json(user);
    } catch (error) {
      next(error);
    }
  }

  async create(req: Request, res: Response, next: NextFunction) {
    try {
      const user = await userService.create(req.body);

      res.status(201)
         .location(`/api/v1/users/${user.id}`)
         .json(user);
    } catch (error) {
      next(error);
    }
  }

  async update(req: Request, res: Response, next: NextFunction) {
    try {
      const { id } = req.params;
      const user = await userService.update(id, req.body);

      res.status(200).json(user);
    } catch (error) {
      next(error);
    }
  }

  async delete(req: Request, res: Response, next: NextFunction) {
    try {
      const { id } = req.params;
      await userService.delete(id);

      res.status(204).send();
    } catch (error) {
      next(error);
    }
  }
}

export default new UserController();
```

## Best Practices

### 1. Use Consistent Naming Conventions

```
✅ GOOD:
- Lowercase URLs: /api/v1/users (not /api/v1/Users)
- Plural nouns: /users, /orders (not /user, /order)
- Kebab-case for multi-word: /shipping-addresses (not /shipping_addresses)
- snake_case for JSON fields: { "first_name": "Alice" }
```

### 2. Include Timestamps

```json
{
  "id": 123,
  "name": "Alice",
  "created_at": "2025-12-10T10:30:00Z",  // ISO 8601 UTC
  "updated_at": "2025-12-10T11:00:00Z"
}
```

### 3. Support Content Negotiation

```http
GET /api/v1/users/123
Accept: application/json

# Also support:
Accept: application/xml
Accept: text/csv
```

### 4. Implement Idempotency

```http
POST /api/v1/orders
Idempotency-Key: order_abc123xyz  # Client-generated unique key

# Server stores result keyed by idempotency key
# Retry with same key returns same response (201 or 200)
```

### 5. HATEOAS (Hypermedia)

```json
{
  "id": 123,
  "name": "Alice",
  "links": {
    "self": "/api/v1/users/123",
    "orders": "/api/v1/users/123/orders",
    "edit": "/api/v1/users/123",
    "delete": "/api/v1/users/123"
  }
}
```

## Common Anti-Patterns

### ❌ Anti-Pattern 1: Verbs in URLs

```
❌ BAD:
POST /api/createUser
GET  /api/getUser/123
POST /api/deleteUser/123

✅ GOOD:
POST   /api/users
GET    /api/users/123
DELETE /api/users/123
```

### ❌ Anti-Pattern 2: Exposing Internal IDs

```
❌ BAD: Sequential integer IDs expose count
GET /api/users/1
GET /api/users/2  # Competitor knows you have 2 users

✅ GOOD: Use UUIDs or opaque IDs
GET /api/users/usr_4f8b3c9d2a1e
```

### ❌ Anti-Pattern 3: No API Versioning

```
❌ BAD: Breaking changes break clients
GET /api/users

✅ GOOD: Version from day one
GET /api/v1/users
```

### ❌ Anti-Pattern 4: Inconsistent Error Formats

```
❌ BAD: Different error formats
{ "error": "Not found" }
{ "message": "Invalid input", "code": 400 }
{ "errors": [...] }

✅ GOOD: Consistent error schema
{
  "error": {
    "code": "...",
    "message": "...",
    "details": {...}
  }
}
```

## Quality Standards

- **HTTP Semantics**: Correct use of methods and status codes
- **Consistency**: Uniform URL structure, naming, and response formats
- **Documentation**: OpenAPI 3.0+ specification for all endpoints
- **Security**: Authentication, authorization, rate limiting, input validation
- **Performance**: Pagination for lists, field selection, caching headers
- **Observability**: Request IDs, structured logging, metrics

---

**Skill Type**: API - REST Design
**Complexity**: Moderate
**Typical Usage**: Activated when API platform engineers design or review REST APIs
**Standards**: RESTful principles, HTTP/1.1, OpenAPI 3.0+
