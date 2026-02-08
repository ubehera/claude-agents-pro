---
name: secure-coding-practices
description: Apply secure coding principles and patterns to prevent common vulnerabilities across languages including input validation, error handling, authentication, and data protection.
trigger_keywords: [secure coding, input validation, sql injection, xss prevention, csrf, secure session, password hashing, encryption, sanitization, owasp]
tags:
  - security
  - secure-coding
  - best-practices
  - code-security
  - defensive-programming
category: security
version: 1.0.0
---

# Secure Coding Practices

Language-agnostic secure coding principles and patterns to prevent vulnerabilities at the source. Covers input validation, authentication, authorization, cryptography, error handling, and defensive programming.

## When to Use This Skill

- **New Feature Development**: Apply security from design phase
- **Code Review**: Identify security anti-patterns
- **Refactoring**: Improve security posture of existing code
- **Training**: Teach secure coding to development teams
- **Security Hardening**: Strengthen application defenses
- **Vulnerability Remediation**: Fix identified security issues

## Core Concepts

- **Input Validation**: All external input is untrusted. Validate at boundaries using whitelisting (not blacklisting). Sanitize for the specific output context (HTML, SQL, shell). Reject invalid input rather than attempting to "fix" it.

- **Least Privilege**: Code should run with minimal permissions. Database connections should use read-only accounts when writes aren't needed. Service accounts should have scoped permissions. Drop privileges as early as possible.

- **Fail Securely**: When errors occur, fail closed (deny access) not open. Never expose sensitive information in error messages. Log security events for monitoring. Handle exceptions explicitly - never swallow security-related errors.

- **Defense in Depth**: Never rely on a single security control. Layer authentication, authorization, input validation, output encoding, and encryption. If one layer fails, others should still protect the system.

- **Secure by Default**: Ship with security enabled. Disable dangerous features unless explicitly enabled. Use secure defaults for all configurations. Make the secure path the easy path.

## Core Secure Coding Principles

### 1. Defense in Depth (Multiple Security Layers)

```typescript
// ❌ Single layer of security
function processPayment(amount: number) {
  // Only validates on client-side
  return stripe.charge(amount);
}

// ✅ Multiple layers of defense
function processPayment(userId: string, amount: number) {
  // Layer 1: Authentication
  if (!isAuthenticated(userId)) {
    throw new UnauthorizedError("User not authenticated");
  }

  // Layer 2: Authorization
  if (!hasPaymentPermission(userId)) {
    throw new ForbiddenError("User lacks payment permission");
  }

  // Layer 3: Input validation
  if (amount <= 0 || amount > 10000) {
    throw new ValidationError("Invalid payment amount");
  }

  // Layer 4: Rate limiting
  if (exceedsRateLimit(userId, 'payment')) {
    throw new RateLimitError("Too many payment attempts");
  }

  // Layer 5: Fraud detection
  if (isFraudulent(userId, amount)) {
    throw new FraudError("Suspicious activity detected");
  }

  // Layer 6: Idempotency
  const idempotencyKey = generateIdempotencyKey(userId, amount);

  // Layer 7: Audit logging
  logSecurityEvent({
    type: 'PAYMENT_ATTEMPT',
    userId,
    amount,
    timestamp: new Date()
  });

  return stripe.charge(amount, { idempotencyKey });
}
```

### 2. Principle of Least Privilege

```typescript
// ❌ Overly permissive
class DatabaseConnection {
  constructor() {
    // Connects as admin with full privileges
    this.connection = mysql.createConnection({
      user: 'admin',
      password: 'admin123',
      database: 'production'
    });
  }

  query(sql: string) {
    return this.connection.query(sql); // Can execute ANY query
  }
}

// ✅ Minimal necessary privileges
class DatabaseConnection {
  private readConnection: Connection;
  private writeConnection: Connection;

  constructor() {
    // Read-only user for queries
    this.readConnection = mysql.createConnection({
      user: 'app_reader',
      password: process.env.DB_READ_PASSWORD,
      database: 'production',
      // Read-only privileges
    });

    // Limited write user
    this.writeConnection = mysql.createConnection({
      user: 'app_writer',
      password: process.env.DB_WRITE_PASSWORD,
      database: 'production',
      // Only INSERT, UPDATE on specific tables
    });
  }

  async query(sql: string, params: any[]) {
    // Use prepared statements
    return this.readConnection.execute(sql, params);
  }

  async insert(table: string, data: object) {
    // Whitelist tables
    const allowedTables = ['users', 'orders', 'products'];
    if (!allowedTables.includes(table)) {
      throw new Error('Unauthorized table access');
    }

    // Build safe parameterized query
    const columns = Object.keys(data).join(', ');
    const placeholders = Object.keys(data).map(() => '?').join(', ');
    const sql = `INSERT INTO ${mysql.escapeId(table)} (${columns}) VALUES (${placeholders})`;

    return this.writeConnection.execute(sql, Object.values(data));
  }
}
```

### 3. Fail Securely

```typescript
// ❌ Fails open (insecure default)
function checkPermission(user: User, resource: string): boolean {
  try {
    return permissionService.hasAccess(user, resource);
  } catch (error) {
    // Error - default to allowing access
    console.error("Permission check failed", error);
    return true; // DANGEROUS!
  }
}

// ✅ Fails closed (secure default)
function checkPermission(user: User, resource: string): boolean {
  try {
    const hasAccess = permissionService.hasAccess(user, resource);

    // Log successful authorization
    logger.info({
      event: 'AUTHORIZATION_CHECK',
      userId: user.id,
      resource,
      granted: hasAccess
    });

    return hasAccess;
  } catch (error) {
    // Error - default to denying access
    logger.error({
      event: 'AUTHORIZATION_ERROR',
      userId: user.id,
      resource,
      error: error.message
    });

    // Fail securely - deny access on error
    return false;
  }
}
```

### 4. Don't Trust User Input (Always Validate)

```typescript
// ❌ Trusts user input
app.post('/api/users/:id', (req, res) => {
  const userId = req.params.id;
  const updates = req.body; // DANGEROUS - no validation

  // Directly updates database with user input
  await db.users.update(userId, updates);
  res.json({ success: true });
});

// ✅ Validates all input
import { z } from 'zod';

const UserUpdateSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  age: z.number().int().min(18).max(120).optional(),
  role: z.enum(['user', 'admin']).optional() // ❌ Should not allow role update!
});

// Even better - separate schemas for different user types
const UserSelfUpdateSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  age: z.number().int().min(18).max(120).optional()
  // No role field - users can't change their own role
});

const AdminUserUpdateSchema = UserSelfUpdateSchema.extend({
  role: z.enum(['user', 'admin', 'moderator']),
  isActive: z.boolean()
});

app.post('/api/users/:id', async (req, res) => {
  try {
    // Validate user ID
    const userId = z.string().uuid().parse(req.params.id);

    // Determine which schema to use
    const schema = req.user.isAdmin
      ? AdminUserUpdateSchema
      : UserSelfUpdateSchema;

    // Validate input
    const validatedData = schema.parse(req.body);

    // Authorization check
    if (userId !== req.user.id && !req.user.isAdmin) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    // Update with validated data only
    await db.users.update(userId, validatedData);

    res.json({ success: true });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return res.status(400).json({
        error: 'Validation failed',
        details: error.errors
      });
    }
    throw error;
  }
});
```

## Secure Coding Patterns

### Input Validation & Sanitization

```typescript
import validator from 'validator';
import DOMPurify from 'isomorphic-dompurify';

class InputValidator {
  /**
   * Validate and sanitize email
   */
  static email(input: string): string {
    // Normalize
    const normalized = input.trim().toLowerCase();

    // Validate format
    if (!validator.isEmail(normalized)) {
      throw new ValidationError('Invalid email format');
    }

    // Length check
    if (normalized.length > 254) {
      throw new ValidationError('Email too long');
    }

    return normalized;
  }

  /**
   * Validate and sanitize URL
   */
  static url(input: string): string {
    // Validate URL format
    if (!validator.isURL(input, {
      protocols: ['http', 'https'],
      require_protocol: true
    })) {
      throw new ValidationError('Invalid URL');
    }

    // Parse URL
    const url = new URL(input);

    // Whitelist allowed domains
    const allowedDomains = ['example.com', 'api.example.com'];
    if (!allowedDomains.some(domain => url.hostname.endsWith(domain))) {
      throw new ValidationError('Domain not allowed');
    }

    return url.toString();
  }

  /**
   * Sanitize HTML to prevent XSS
   */
  static html(input: string): string {
    // Configure DOMPurify
    const config = {
      ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br'],
      ALLOWED_ATTR: ['href'],
      ALLOW_DATA_ATTR: false
    };

    // Sanitize
    return DOMPurify.sanitize(input, config);
  }

  /**
   * Validate filename to prevent path traversal
   */
  static filename(input: string): string {
    // Remove path traversal attempts
    const sanitized = input.replace(/\.\./g, '').replace(/\//g, '');

    // Validate format
    if (!/^[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+$/.test(sanitized)) {
      throw new ValidationError('Invalid filename');
    }

    // Max length
    if (sanitized.length > 255) {
      throw new ValidationError('Filename too long');
    }

    return sanitized;
  }

  /**
   * Validate integer input
   */
  static integer(input: any, min?: number, max?: number): number {
    const num = parseInt(input, 10);

    if (isNaN(num) || !Number.isInteger(num)) {
      throw new ValidationError('Invalid integer');
    }

    if (min !== undefined && num < min) {
      throw new ValidationError(`Value must be at least ${min}`);
    }

    if (max !== undefined && num > max) {
      throw new ValidationError(`Value must be at most ${max}`);
    }

    return num;
  }
}
```

### SQL Injection Prevention

```typescript
// ❌ String concatenation (VULNERABLE)
async function findUser(email: string) {
  const query = `SELECT * FROM users WHERE email = '${email}'`;
  return db.query(query);
  // Attacker input: ' OR '1'='1' --
}

// ✅ Parameterized queries
async function findUser(email: string) {
  const query = 'SELECT * FROM users WHERE email = ?';
  return db.execute(query, [email]);
}

// ✅ ORM usage
async function findUser(email: string) {
  return db.users.findOne({
    where: { email }
  });
}

// ✅ Query builder
async function searchUsers(filters: {
  name?: string;
  email?: string;
  minAge?: number;
}) {
  let query = db.users.createQueryBuilder('user');

  if (filters.name) {
    query = query.andWhere('user.name LIKE :name', {
      name: `%${filters.name}%`
    });
  }

  if (filters.email) {
    query = query.andWhere('user.email = :email', {
      email: filters.email
    });
  }

  if (filters.minAge) {
    query = query.andWhere('user.age >= :minAge', {
      minAge: filters.minAge
    });
  }

  return query.getMany();
}
```

### XSS Prevention

```typescript
import DOMPurify from 'isomorphic-dompurify';
import { escape } from 'html-escaper';

// ❌ Directly inserting user content
function renderComment(comment: string) {
  return `<div class="comment">${comment}</div>`;
  // Attacker input: <script>alert('XSS')</script>
}

// ✅ Escape HTML entities
function renderComment(comment: string) {
  const escaped = escape(comment);
  return `<div class="comment">${escaped}</div>`;
}

// ✅ Use framework's escaping (React example)
function Comment({ comment }: { comment: string }) {
  // React automatically escapes
  return <div className="comment">{comment}</div>;
}

// ✅ Sanitize rich text
function renderRichComment(html: string) {
  const clean = DOMPurify.sanitize(html, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'ul', 'li'],
    ALLOWED_ATTR: ['href'],
    ALLOWED_URI_REGEXP: /^https:\/\/example\.com\//
  });

  return `<div class="comment">${clean}</div>`;
}

// ✅ Content Security Policy (CSP)
app.use((req, res, next) => {
  res.setHeader('Content-Security-Policy',
    "default-src 'self'; " +
    "script-src 'self' https://cdn.example.com; " +
    "style-src 'self' 'unsafe-inline'; " +
    "img-src 'self' data: https:; " +
    "font-src 'self'; " +
    "connect-src 'self' https://api.example.com; " +
    "frame-ancestors 'none'; " +
    "base-uri 'self'; " +
    "form-action 'self'"
  );
  next();
});
```

### Authentication & Password Handling

```typescript
import bcrypt from 'bcrypt';
import crypto from 'crypto';

class AuthenticationService {
  private readonly SALT_ROUNDS = 12;
  private readonly MAX_LOGIN_ATTEMPTS = 5;
  private readonly LOCKOUT_DURATION = 15 * 60 * 1000; // 15 minutes

  /**
   * Hash password securely
   */
  async hashPassword(password: string): Promise<string> {
    // Validate password strength
    this.validatePasswordStrength(password);

    // Hash with bcrypt
    return bcrypt.hash(password, this.SALT_ROUNDS);
  }

  /**
   * Verify password
   */
  async verifyPassword(password: string, hash: string): Promise<boolean> {
    try {
      return await bcrypt.compare(password, hash);
    } catch (error) {
      // Log error but don't leak information
      logger.error('Password verification error', { error });
      return false;
    }
  }

  /**
   * Validate password strength
   */
  private validatePasswordStrength(password: string): void {
    const errors: string[] = [];

    if (password.length < 12) {
      errors.push('Password must be at least 12 characters');
    }

    if (!/[A-Z]/.test(password)) {
      errors.push('Password must contain uppercase letters');
    }

    if (!/[a-z]/.test(password)) {
      errors.push('Password must contain lowercase letters');
    }

    if (!/[0-9]/.test(password)) {
      errors.push('Password must contain numbers');
    }

    if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
      errors.push('Password must contain special characters');
    }

    // Check against common passwords
    if (this.isCommonPassword(password)) {
      errors.push('Password is too common');
    }

    if (errors.length > 0) {
      throw new ValidationError(errors.join('; '));
    }
  }

  /**
   * Generate secure random token
   */
  generateToken(length: number = 32): string {
    return crypto.randomBytes(length).toString('base64url');
  }

  /**
   * Handle failed login attempt
   */
  async handleFailedLogin(userId: string): Promise<void> {
    const key = `login_attempts:${userId}`;
    const attempts = await redis.incr(key);

    if (attempts === 1) {
      // Set expiration on first attempt
      await redis.expire(key, this.LOCKOUT_DURATION / 1000);
    }

    if (attempts >= this.MAX_LOGIN_ATTEMPTS) {
      // Lock account
      await this.lockAccount(userId, this.LOCKOUT_DURATION);

      // Send alert
      await this.sendSecurityAlert(userId, 'ACCOUNT_LOCKED');

      throw new AccountLockedError(
        `Account locked for ${this.LOCKOUT_DURATION / 60000} minutes`
      );
    }

    // Progressive delay
    const delay = Math.min(1000 * Math.pow(2, attempts - 1), 10000);
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  /**
   * Constant-time string comparison (prevent timing attacks)
   */
  constantTimeCompare(a: string, b: string): boolean {
    if (a.length !== b.length) {
      return false;
    }

    return crypto.timingSafeEqual(
      Buffer.from(a),
      Buffer.from(b)
    );
  }
}
```

### Secure Session Management

```typescript
import session from 'express-session';
import RedisStore from 'connect-redis';
import { randomBytes } from 'crypto';

// ✅ Secure session configuration
const sessionConfig: session.SessionOptions = {
  store: new RedisStore({
    client: redisClient,
    prefix: 'sess:'
  }),

  // Generate cryptographically strong session ID
  genid: () => randomBytes(32).toString('base64url'),

  // Session secret (from environment)
  secret: process.env.SESSION_SECRET!,

  // Don't save session if unmodified
  resave: false,

  // Don't create session until something stored
  saveUninitialized: false,

  // Cookie settings
  cookie: {
    secure: true,          // HTTPS only
    httpOnly: true,        // No JavaScript access
    sameSite: 'strict',    // CSRF protection
    maxAge: 15 * 60 * 1000, // 15 minutes
    domain: process.env.COOKIE_DOMAIN,
    path: '/'
  },

  // Rolling sessions (reset expiry on activity)
  rolling: true,

  // Session name (hide technology)
  name: 'sessionId'
};

app.use(session(sessionConfig));

// Session regeneration after authentication
app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;

  // Authenticate user
  const user = await authService.authenticate(email, password);

  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // Regenerate session to prevent fixation
  req.session.regenerate((err) => {
    if (err) {
      return res.status(500).json({ error: 'Session error' });
    }

    // Store user info in new session
    req.session.userId = user.id;
    req.session.loginTime = Date.now();

    res.json({
      success: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name
      }
    });
  });
});

// Destroy session on logout
app.post('/api/auth/logout', (req, res) => {
  req.session.destroy((err) => {
    if (err) {
      return res.status(500).json({ error: 'Logout failed' });
    }

    res.clearCookie('sessionId');
    res.json({ success: true });
  });
});
```

### Cryptography Best Practices

```typescript
import { createCipheriv, createDecipheriv, randomBytes, scryptSync } from 'crypto';

class CryptoService {
  private readonly ALGORITHM = 'aes-256-gcm';
  private readonly KEY_LENGTH = 32;
  private readonly IV_LENGTH = 16;
  private readonly TAG_LENGTH = 16;
  private readonly SALT_LENGTH = 64;

  /**
   * Encrypt data with authenticated encryption
   */
  encrypt(plaintext: string, masterKey: Buffer): string {
    // Generate unique IV
    const iv = randomBytes(this.IV_LENGTH);

    // Generate salt
    const salt = randomBytes(this.SALT_LENGTH);

    // Derive key from master key
    const key = scryptSync(masterKey, salt, this.KEY_LENGTH);

    // Create cipher
    const cipher = createCipheriv(this.ALGORITHM, key, iv);

    // Encrypt
    const encrypted = Buffer.concat([
      cipher.update(plaintext, 'utf8'),
      cipher.final()
    ]);

    // Get authentication tag
    const tag = cipher.getAuthTag();

    // Combine salt + iv + tag + ciphertext
    const combined = Buffer.concat([salt, iv, tag, encrypted]);

    return combined.toString('base64');
  }

  /**
   * Decrypt data with verification
   */
  decrypt(ciphertext: string, masterKey: Buffer): string {
    const combined = Buffer.from(ciphertext, 'base64');

    // Extract components
    const salt = combined.slice(0, this.SALT_LENGTH);
    const iv = combined.slice(this.SALT_LENGTH, this.SALT_LENGTH + this.IV_LENGTH);
    const tag = combined.slice(
      this.SALT_LENGTH + this.IV_LENGTH,
      this.SALT_LENGTH + this.IV_LENGTH + this.TAG_LENGTH
    );
    const encrypted = combined.slice(this.SALT_LENGTH + this.IV_LENGTH + this.TAG_LENGTH);

    // Derive same key
    const key = scryptSync(masterKey, salt, this.KEY_LENGTH);

    // Create decipher
    const decipher = createDecipheriv(this.ALGORITHM, key, iv);
    decipher.setAuthTag(tag);

    // Decrypt and verify
    try {
      const decrypted = Buffer.concat([
        decipher.update(encrypted),
        decipher.final()
      ]);
      return decrypted.toString('utf8');
    } catch (error) {
      throw new Error('Decryption failed - data may be tampered');
    }
  }

  /**
   * Generate secure random key
   */
  generateKey(): Buffer {
    return randomBytes(this.KEY_LENGTH);
  }

  /**
   * Hash data (one-way)
   */
  hash(data: string): string {
    const hash = createHash('sha256');
    hash.update(data);
    return hash.digest('hex');
  }
}
```

### Error Handling (Don't Leak Information)

```typescript
// ❌ Leaks sensitive information
app.post('/api/auth/login', async (req, res) => {
  try {
    const user = await db.query('SELECT * FROM users WHERE email = ?', [req.body.email]);

    if (!user) {
      return res.status(404).json({
        error: 'User not found',
        email: req.body.email // LEAKS USER EXISTENCE
      });
    }

    const valid = await bcrypt.compare(req.body.password, user.password);

    if (!valid) {
      return res.status(401).json({
        error: 'Incorrect password',
        attempts: user.loginAttempts // LEAKS ACCOUNT INFO
      });
    }

    res.json({ token: generateToken(user) });
  } catch (error) {
    // LEAKS INTERNAL ERRORS
    res.status(500).json({
      error: error.message,
      stack: error.stack,
      query: 'SELECT * FROM users...'
    });
  }
});

// ✅ Generic error messages, detailed logging
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Validate input
    if (!email || !password) {
      return res.status(400).json({
        error: 'Missing required fields'
      });
    }

    const user = await db.users.findByEmail(email);

    // Constant-time check to prevent timing attacks
    const userExists = !!user;
    const passwordValid = user
      ? await bcrypt.compare(password, user.passwordHash)
      : await bcrypt.compare(password, '$2b$12$dummy'); // Dummy hash

    if (!userExists || !passwordValid) {
      // Log detailed info server-side
      logger.warn({
        event: 'LOGIN_FAILED',
        email,
        reason: !userExists ? 'USER_NOT_FOUND' : 'INVALID_PASSWORD',
        ip: req.ip,
        userAgent: req.headers['user-agent']
      });

      // Generic error to client
      return res.status(401).json({
        error: 'Invalid credentials'
      });
    }

    // Success
    const token = generateToken(user);

    logger.info({
      event: 'LOGIN_SUCCESS',
      userId: user.id,
      ip: req.ip
    });

    res.json({ token });
  } catch (error) {
    // Log detailed error server-side
    logger.error({
      event: 'LOGIN_ERROR',
      error: error.message,
      stack: error.stack
    });

    // Generic error to client
    res.status(500).json({
      error: 'An error occurred. Please try again.'
    });
  }
});
```

## Secure Coding Checklist

```markdown
## Input Validation
- [ ] Validate all user input (whitelist, not blacklist)
- [ ] Sanitize HTML to prevent XSS
- [ ] Use parameterized queries to prevent SQL injection
- [ ] Validate file uploads (type, size, content)
- [ ] Validate and sanitize URLs
- [ ] Check for path traversal in file paths

## Authentication & Authorization
- [ ] Hash passwords with bcrypt/Argon2 (min 12 rounds)
- [ ] Enforce strong password policy (12+ chars, complexity)
- [ ] Implement rate limiting on authentication
- [ ] Use constant-time comparison for secrets
- [ ] Regenerate session after authentication
- [ ] Implement account lockout after failed attempts
- [ ] Verify authorization on every request
- [ ] Prevent IDOR vulnerabilities

## Session Management
- [ ] Use secure, httpOnly, sameSite cookies
- [ ] Set appropriate session timeouts
- [ ] Regenerate session IDs after privilege changes
- [ ] Implement CSRF protection
- [ ] Destroy sessions on logout

## Cryptography
- [ ] Use strong algorithms (AES-256-GCM, RSA-4096)
- [ ] Never roll your own crypto
- [ ] Use authenticated encryption (GCM mode)
- [ ] Generate cryptographically secure random values
- [ ] Store secrets in environment variables, not code
- [ ] Use TLS 1.3 for data in transit

## Error Handling
- [ ] Don't leak sensitive information in errors
- [ ] Log detailed errors server-side
- [ ] Return generic errors to clients
- [ ] Implement proper exception handling
- [ ] Fail securely (deny by default)

## Data Protection
- [ ] Encrypt sensitive data at rest
- [ ] Use HTTPS for data in transit
- [ ] Implement proper access controls
- [ ] Sanitize data in logs (mask PII)
- [ ] Implement data retention policies

## Security Headers
- [ ] Content-Security-Policy
- [ ] X-Content-Type-Options: nosniff
- [ ] X-Frame-Options: DENY
- [ ] Strict-Transport-Security
- [ ] Referrer-Policy

## Code Quality
- [ ] Follow principle of least privilege
- [ ] Implement defense in depth
- [ ] Use security linters (Semgrep, ESLint)
- [ ] Perform code reviews
- [ ] Write security tests
```

## Language-Specific Guidelines

### JavaScript/TypeScript
- Use strict mode (`'use strict'`)
- Avoid `eval()`, `Function()`, `setTimeout(string)`
- Validate all inputs with libraries like Zod, Joi
- Use Content Security Policy to prevent XSS
- Sanitize HTML with DOMPurify
- Use secure dependencies (npm audit)

### Python
- Use parameterized queries (avoid string concatenation)
- Validate input with Pydantic, marshmallow
- Use secrets module for cryptographic randomness
- Avoid `eval()`, `exec()`, `pickle.loads()` on untrusted input
- Use bandit for security linting

### Java
- Use PreparedStatement for SQL queries
- Validate input with Bean Validation (JSR 380)
- Use OWASP Java Encoder for output encoding
- Avoid reflection on untrusted input
- Use strong cryptography (JCA/JCE)

### Go
- Use parameterized queries (`database/sql`)
- Validate input with validator libraries
- Use `crypto/rand` for randomness
- Avoid `eval`-like constructs
- Use gosec for security scanning

## Resources

- **OWASP Secure Coding Practices**: https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/
- **CWE Top 25**: https://cwe.mitre.org/top25/
- **CERT Secure Coding Standards**: https://wiki.sei.cmu.edu/confluence/display/seccode

Secure coding is a continuous practice, not a one-time effort. Build security into every phase of development to create resilient, trustworthy applications.
