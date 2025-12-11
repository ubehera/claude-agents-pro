---
name: owasp-top-10
description: Load when user needs OWASP Top 10 security vulnerabilities, mitigation strategies, secure coding patterns, authentication, authorization, or input validation
trigger_keywords: [owasp, security, vulnerability, sql injection, xss, cross-site scripting, csrf, authentication, authorization, injection, broken access control, security misconfiguration, sensitive data exposure]
---

# OWASP Top 10 Security Vulnerabilities

Production-ready security patterns and mitigation strategies for the OWASP Top 10 web application vulnerabilities (2021 edition).

## Core Concepts

### OWASP Top 10 (2021)

1. **A01:2021 - Broken Access Control** (⬆️ from #5)
2. **A02:2021 - Cryptographic Failures** (formerly Sensitive Data Exposure)
3. **A03:2021 - Injection** (⬇️ from #1)
4. **A04:2021 - Insecure Design** (NEW)
5. **A05:2021 - Security Misconfiguration**
6. **A06:2021 - Vulnerable and Outdated Components**
7. **A07:2021 - Identification and Authentication Failures**
8. **A08:2021 - Software and Data Integrity Failures** (NEW)
9. **A09:2021 - Security Logging and Monitoring Failures**
10. **A10:2021 - Server-Side Request Forgery (SSRF)** (NEW)

## Implementation Patterns

### 1. Broken Access Control

**Vulnerability**: Users can access resources they shouldn't.

**Attack Examples**:
```
# Direct object reference
GET /api/users/999/orders  # Accessing another user's orders

# Path traversal
GET /api/files?path=../../etc/passwd

# Elevation of privilege
POST /api/users/123/promote  # Regular user promoting themselves to admin
```

**Mitigation**:

```typescript
// Express.js middleware for authorization
import { Request, Response, NextFunction } from 'express';

interface AuthRequest extends Request {
  user?: {
    id: string;
    role: string;
  };
}

// Check resource ownership
async function requireOwnership(req: AuthRequest, res: Response, next: NextFunction) {
  const resourceUserId = req.params.userId;
  const currentUserId = req.user?.id;

  if (resourceUserId !== currentUserId && req.user?.role !== 'admin') {
    return res.status(403).json({
      error: {
        code: 'FORBIDDEN',
        message: 'You do not have permission to access this resource'
      }
    });
  }

  next();
}

// Apply to routes
router.get('/users/:userId/orders', authenticate, requireOwnership, getOrders);

// Role-based access control
function requireRole(...allowedRoles: string[]) {
  return (req: AuthRequest, res: Response, next: NextFunction) => {
    if (!req.user || !allowedRoles.includes(req.user.role)) {
      return res.status(403).json({
        error: {
          code: 'FORBIDDEN',
          message: 'Insufficient permissions'
        }
      });
    }
    next();
  };
}

router.delete('/users/:id', authenticate, requireRole('admin'), deleteUser);
```

### 2. Cryptographic Failures

**Vulnerability**: Weak encryption, exposed secrets, insecure data transmission.

**Attack Examples**:
```
# Passwords stored in plaintext
SELECT * FROM users WHERE password = 'hunter2'

# Sensitive data in URLs
GET /reset-password?token=secret123&ssn=123-45-6789

# Data transmitted over HTTP (not HTTPS)
```

**Mitigation**:

```typescript
import bcrypt from 'bcrypt';
import crypto from 'crypto';

// Password hashing (NEVER store plaintext)
async function hashPassword(password: string): Promise<string> {
  const saltRounds = 12;  // Adjust based on performance requirements
  return bcrypt.hash(password, saltRounds);
}

async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

// Encrypt sensitive data at rest
class EncryptionService {
  private algorithm = 'aes-256-gcm';
  private key: Buffer;

  constructor(secretKey: string) {
    // Derive 256-bit key from secret
    this.key = crypto.scryptSync(secretKey, 'salt', 32);
  }

  encrypt(plaintext: string): string {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv(this.algorithm, this.key, iv);

    let encrypted = cipher.update(plaintext, 'utf8', 'hex');
    encrypted += cipher.final('hex');

    const authTag = cipher.getAuthTag();

    // Return: iv:authTag:ciphertext
    return `${iv.toString('hex')}:${authTag.toString('hex')}:${encrypted}`;
  }

  decrypt(ciphertext: string): string {
    const [ivHex, authTagHex, encrypted] = ciphertext.split(':');

    const iv = Buffer.from(ivHex, 'hex');
    const authTag = Buffer.from(authTagHex, 'hex');

    const decipher = crypto.createDecipheriv(this.algorithm, this.key, iv);
    decipher.setAuthTag(authTag);

    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return decrypted;
  }
}

// Use environment variables for secrets
const encryptionKey = process.env.ENCRYPTION_KEY;
if (!encryptionKey) {
  throw new Error('ENCRYPTION_KEY environment variable must be set');
}

const encryption = new EncryptionService(encryptionKey);
```

### 3. Injection (SQL, NoSQL, Command)

**Vulnerability**: Untrusted data executed as code.

**Attack Examples**:
```sql
-- SQL Injection
SELECT * FROM users WHERE username = 'admin' -- ' AND password = 'anything'

-- Command Injection
curl https://api.example.com/ping?host=google.com; cat /etc/passwd

-- NoSQL Injection
db.users.find({ username: { $ne: null }, password: { $ne: null } })
```

**Mitigation**:

```typescript
// ✅ SQL: Use parameterized queries (prepared statements)
import { Pool } from 'pg';

const pool = new Pool();

// ❌ VULNERABLE
async function getUserVulnerable(username: string) {
  const query = `SELECT * FROM users WHERE username = '${username}'`;
  return pool.query(query);  // DON'T DO THIS!
}

// ✅ SECURE
async function getUserSecure(username: string) {
  const query = 'SELECT * FROM users WHERE username = $1';
  return pool.query(query, [username]);  // Parameterized query
}

// ✅ NoSQL: Validate and sanitize input
import { ObjectId } from 'mongodb';

async function getUserByIdSecure(userId: string) {
  // Validate ObjectId format
  if (!ObjectId.isValid(userId)) {
    throw new Error('Invalid user ID format');
  }

  return db.collection('users').findOne({ _id: new ObjectId(userId) });
}

// ✅ Command execution: Avoid shell, use safe APIs
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

// ❌ VULNERABLE
async function pingVulnerable(host: string) {
  const { exec } = require('child_process');
  return exec(`ping -c 4 ${host}`);  // DON'T DO THIS!
}

// ✅ SECURE
async function pingSecure(host: string) {
  // Whitelist validation
  if (!/^[a-zA-Z0-9.-]+$/.test(host)) {
    throw new Error('Invalid hostname');
  }

  // Use execFile (no shell)
  return execFileAsync('ping', ['-c', '4', host]);
}
```

### 4. Insecure Design

**Vulnerability**: Missing security controls in design phase.

**Mitigation**: Threat modeling and secure design patterns.

```typescript
// Example: Secure password reset flow
import crypto from 'crypto';

class PasswordResetService {
  async initiateReset(email: string): Promise<void> {
    const user = await userRepository.findByEmail(email);

    // ALWAYS send generic message (prevent user enumeration)
    const genericMessage = 'If an account exists, a reset link has been sent';

    if (!user) {
      // Don't reveal if user exists
      return;
    }

    // Generate cryptographically secure token
    const token = crypto.randomBytes(32).toString('hex');
    const expiresAt = new Date(Date.now() + 15 * 60 * 1000); // 15 min

    // Hash token before storing (even tokens should be hashed)
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    await passwordResetRepository.create({
      userId: user.id,
      tokenHash: hashedToken,
      expiresAt
    });

    // Send unhashed token via email (one-time use)
    await emailService.sendPasswordReset(user.email, token);
  }

  async resetPassword(token: string, newPassword: string): Promise<void> {
    // Hash submitted token
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    const resetRequest = await passwordResetRepository.findByToken(hashedToken);

    // Check validity
    if (!resetRequest || resetRequest.expiresAt < new Date()) {
      throw new Error('Invalid or expired reset token');
    }

    // Rate limit reset attempts
    await this.checkResetAttempts(resetRequest.userId);

    // Update password
    const hashedPassword = await hashPassword(newPassword);
    await userRepository.updatePassword(resetRequest.userId, hashedPassword);

    // Invalidate all reset tokens for this user
    await passwordResetRepository.invalidateForUser(resetRequest.userId);

    // Invalidate all sessions (force re-login)
    await sessionRepository.invalidateForUser(resetRequest.userId);
  }
}
```

### 5. Security Misconfiguration

**Vulnerability**: Default credentials, verbose errors, unnecessary features enabled.

**Mitigation**:

```typescript
// ✅ Secure Express.js configuration
import express from 'express';
import helmet from 'helmet';

const app = express();

// Security headers
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
    }
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  }
}));

// Hide server information
app.disable('x-powered-by');

// Environment-specific error handling
if (process.env.NODE_ENV === 'production') {
  // ✅ Generic error messages in production
  app.use((err, req, res, next) => {
    console.error(err);  // Log full error
    res.status(500).json({
      error: {
        code: 'INTERNAL_SERVER_ERROR',
        message: 'An unexpected error occurred'
        // DON'T expose stack trace or internal details
      }
    });
  });
} else {
  // ✅ Detailed errors in development only
  app.use((err, req, res, next) => {
    res.status(500).json({
      error: {
        message: err.message,
        stack: err.stack
      }
    });
  });
}

// Disable directory listing
app.use(express.static('public', {
  dotfiles: 'deny',
  index: false
}));
```

### 6. Vulnerable and Outdated Components

**Mitigation**:

```bash
# Audit dependencies regularly
npm audit
npm audit fix

# Use Snyk or Dependabot for automated monitoring
# package.json
{
  "scripts": {
    "security-check": "npm audit --audit-level=moderate"
  }
}

# Keep dependencies updated
npm outdated
npm update

# Use lock files
# npm: package-lock.json
# yarn: yarn.lock
```

### 7. Identification and Authentication Failures

**Vulnerability**: Weak passwords, broken session management, credential stuffing.

**Mitigation**:

```typescript
import rateLimit from 'express-rate-limit';
import slowDown from 'express-slow-down';

// Rate limiting for authentication endpoints
const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 5,  // Max 5 attempts
  message: {
    error: {
      code: 'TOO_MANY_LOGIN_ATTEMPTS',
      message: 'Too many login attempts, please try again later'
    }
  }
});

// Progressive delay on repeated attempts
const loginSlowDown = slowDown({
  windowMs: 15 * 60 * 1000,
  delayAfter: 2,  // Start delaying after 2 requests
  delayMs: 500    // Add 500ms delay per request
});

router.post('/login', loginLimiter, loginSlowDown, async (req, res) => {
  const { email, password } = req.body;

  const user = await userRepository.findByEmail(email);

  // Use constant-time comparison to prevent timing attacks
  if (!user || !(await bcrypt.compare(password, user.passwordHash))) {
    // Generic message (don't reveal if user exists)
    return res.status(401).json({
      error: {
        code: 'INVALID_CREDENTIALS',
        message: 'Invalid email or password'
      }
    });
  }

  // Check if account is locked
  if (user.lockedUntil && user.lockedUntil > new Date()) {
    return res.status(403).json({
      error: {
        code: 'ACCOUNT_LOCKED',
        message: 'Account is temporarily locked'
      }
    });
  }

  // Reset failed login attempts
  await userRepository.resetFailedLoginAttempts(user.id);

  // Generate session token
  const sessionToken = await sessionService.create(user.id);

  res.json({
    token: sessionToken,
    user: { id: user.id, email: user.email, role: user.role }
  });
});

// Password complexity requirements
import validator from 'validator';

function validatePassword(password: string): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (password.length < 12) {
    errors.push('Password must be at least 12 characters');
  }

  if (!/[a-z]/.test(password)) {
    errors.push('Password must contain lowercase letter');
  }

  if (!/[A-Z]/.test(password)) {
    errors.push('Password must contain uppercase letter');
  }

  if (!/[0-9]/.test(password)) {
    errors.push('Password must contain number');
  }

  if (!/[^a-zA-Z0-9]/.test(password)) {
    errors.push('Password must contain special character');
  }

  // Check against common passwords
  if (isCommonPassword(password)) {
    errors.push('Password is too common');
  }

  return { valid: errors.length === 0, errors };
}
```

### 8. Software and Data Integrity Failures

**Vulnerability**: Unsigned packages, insecure CI/CD, unverified updates.

**Mitigation**:

```typescript
// Verify package integrity
import crypto from 'crypto';

async function verifyFileIntegrity(filePath: string, expectedHash: string): Promise<boolean> {
  const hash = crypto.createHash('sha256');
  const stream = fs.createReadStream(filePath);

  return new Promise((resolve, reject) => {
    stream.on('data', (data) => hash.update(data));
    stream.on('end', () => resolve(hash.digest('hex') === expectedHash));
    stream.on('error', reject);
  });
}

// Signed JWTs for data integrity
import jwt from 'jsonwebtoken';

function createSignedToken(payload: object): string {
  const secret = process.env.JWT_SECRET!;
  return jwt.sign(payload, secret, {
    algorithm: 'HS256',
    expiresIn: '1h'
  });
}

function verifySignedToken(token: string): object {
  const secret = process.env.JWT_SECRET!;
  return jwt.verify(token, secret, { algorithms: ['HS256'] });
}
```

### 9. Security Logging and Monitoring Failures

**Mitigation**:

```typescript
import winston from 'winston';

// Structured logging
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  defaultMeta: { service: 'user-service' },
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Log security events
function logSecurityEvent(event: {
  type: 'LOGIN_SUCCESS' | 'LOGIN_FAILURE' | 'ACCESS_DENIED' | 'SUSPICIOUS_ACTIVITY';
  userId?: string;
  ip: string;
  userAgent: string;
  details?: object;
}) {
  logger.warn('Security Event', {
    ...event,
    timestamp: new Date().toISOString()
  });

  // Alert on suspicious patterns
  if (event.type === 'SUSPICIOUS_ACTIVITY') {
    alertService.notify(event);
  }
}

// Middleware to log all authentication attempts
app.use((req, res, next) => {
  const originalSend = res.send;

  res.send = function(data) {
    if (req.path === '/api/auth/login') {
      logSecurityEvent({
        type: res.statusCode === 200 ? 'LOGIN_SUCCESS' : 'LOGIN_FAILURE',
        userId: req.body?.email,
        ip: req.ip,
        userAgent: req.headers['user-agent'] || 'unknown'
      });
    }

    return originalSend.call(this, data);
  };

  next();
});
```

### 10. Server-Side Request Forgery (SSRF)

**Vulnerability**: Server makes requests to attacker-controlled URLs.

**Attack Examples**:
```
# Internal service access
GET /api/fetch?url=http://localhost:9200/admin

# Cloud metadata access
GET /api/fetch?url=http://169.254.169.254/latest/meta-data/iam/security-credentials/

# Port scanning
GET /api/fetch?url=http://internal-host:22
```

**Mitigation**:

```typescript
import axios from 'axios';
import { parse } from 'url';

// Whitelist approach
const ALLOWED_DOMAINS = ['api.example.com', 'cdn.example.com'];

async function fetchUrlSecure(url: string): Promise<any> {
  const parsed = parse(url);

  // Validate protocol
  if (parsed.protocol !== 'https:') {
    throw new Error('Only HTTPS URLs are allowed');
  }

  // Validate domain
  if (!ALLOWED_DOMAINS.includes(parsed.hostname || '')) {
    throw new Error('Domain not allowed');
  }

  // Prevent redirects to internal resources
  return axios.get(url, {
    maxRedirects: 0,
    timeout: 5000
  });
}

// Blacklist approach (less secure, use whitelist if possible)
const BLOCKED_IPS = [
  '127.0.0.0/8',      // Loopback
  '10.0.0.0/8',       // Private
  '172.16.0.0/12',    // Private
  '192.168.0.0/16',   // Private
  '169.254.0.0/16'    // Link-local (AWS metadata)
];

function isBlockedIP(ip: string): boolean {
  // Check against blocked ranges
  // Use a library like 'ip-range-check' for production
  return BLOCKED_IPS.some(range => ipInRange(ip, range));
}
```

## Quality Standards

- **Input Validation**: Validate all user input (whitelist approach)
- **Output Encoding**: Encode data before rendering (prevent XSS)
- **Authentication**: Strong passwords, MFA, rate limiting
- **Authorization**: Principle of least privilege, check on every request
- **Cryptography**: Use proven algorithms (AES-256, bcrypt, scrypt)
- **Logging**: Log security events, protect sensitive data in logs
- **Dependencies**: Automated scanning, regular updates
- **Testing**: Security unit tests, penetration testing

---

**Skill Type**: Security - Application Security
**Complexity**: Moderate
**Typical Usage**: Activated when security architects review code or design secure systems
**Standards**: OWASP Top 10 2021, NIST, CWE/SANS Top 25
