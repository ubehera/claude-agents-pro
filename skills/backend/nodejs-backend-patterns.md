---
name: nodejs-backend-patterns
description: Load when user needs Node.js backend architecture, Express/Fastify setup, layered architecture, middleware patterns, dependency injection, or database integration
trigger_keywords: [nodejs, node.js, express, fastify, backend, middleware, dependency injection, repository pattern, service layer, controller, postgresql, mongodb, mongoose]
---

# Node.js Backend Patterns

Production-grade Node.js backend architecture with Express/Fastify, layered design, middleware patterns, dependency injection, and database integration for scalable, maintainable applications.

## Core Concepts

- **Layered Architecture**: Separate controllers (HTTP handling), services (business logic), and repositories (data access). Controllers never access databases directly; services never know about HTTP. This separation enables unit testing without mocking Express.

- **Middleware Pipeline**: Express/Fastify middleware executes in order. Place security middleware (helmet, cors) first, then parsing, then authentication, then routes, then error handlers last. Order matters for both security and performance.

- **Dependency Injection**: Avoid `require()` for cross-cutting concerns. Use a DI container or constructor injection to wire dependencies. This makes testing trivial (inject mocks) and supports different configurations per environment.

- **Connection Pooling**: Never create database connections per request. Use connection pools (pg.Pool, mongoose connection) with appropriate pool sizes (typically 10-20). Monitor pool exhaustion in production.

- **Error Handling Strategy**: Distinguish operational errors (user input, network) from programmer errors (null reference, type error). Operational errors return HTTP 4xx/5xx; programmer errors crash and restart the process.

- **Anti-pattern - Callback Hell**: Avoid nested callbacks. Use async/await consistently. Wrap callback-based APIs with `util.promisify()`. Never mix callbacks and promises in the same flow.

## Core Frameworks

### Express.js - Minimalist Framework

```typescript
import express, { Request, Response, NextFunction } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';

const app = express();

// Security middleware
app.use(helmet());
app.use(cors({
    origin: process.env.ALLOWED_ORIGINS?.split(','),
    credentials: true
}));
app.use(compression());

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging
app.use((req: Request, res: Response, next: NextFunction) => {
    const start = Date.now();
    res.on('finish', () => {
        const duration = Date.now() - start;
        console.log(`${req.method} ${req.path} ${res.statusCode} ${duration}ms`);
    });
    next();
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
```

### Fastify - High Performance Framework

```typescript
import Fastify from 'fastify';
import helmet from '@fastify/helmet';
import cors from '@fastify/cors';
import compress from '@fastify/compress';

const fastify = Fastify({
    logger: {
        level: process.env.LOG_LEVEL || 'info',
        transport: {
            target: 'pino-pretty',
            options: { colorize: true }
        }
    }
});

// Plugins
await fastify.register(helmet);
await fastify.register(cors, { origin: true });
await fastify.register(compress);

// Type-safe routes with schema validation
fastify.post<{
    Body: { name: string; email: string };
    Reply: { id: string; name: string };
}>('/users', {
    schema: {
        body: {
            type: 'object',
            required: ['name', 'email'],
            properties: {
                name: { type: 'string', minLength: 1 },
                email: { type: 'string', format: 'email' }
            }
        }
    }
}, async (request, reply) => {
    const { name, email } = request.body;
    const user = await userService.create({ name, email });
    return { id: user.id, name: user.name };
});

await fastify.listen({ port: 3000, host: '0.0.0.0' });
```

## Layered Architecture

### Project Structure

```
src/
├── controllers/     # Handle HTTP requests/responses
├── services/        # Business logic
├── repositories/    # Data access layer
├── models/          # Data models
├── middleware/      # Express/Fastify middleware
├── routes/          # Route definitions
├── utils/           # Helper functions
├── config/          # Configuration
└── types/           # TypeScript types
```

### Controller Layer

```typescript
// controllers/user.controller.ts
import { Request, Response, NextFunction } from 'express';
import { UserService } from '../services/user.service';
import { CreateUserDTO, UpdateUserDTO } from '../types/user.types';

export class UserController {
    constructor(private userService: UserService) {}

    async createUser(req: Request, res: Response, next: NextFunction) {
        try {
            const userData: CreateUserDTO = req.body;
            const user = await this.userService.createUser(userData);
            res.status(201).json(user);
        } catch (error) {
            next(error);
        }
    }

    async getUser(req: Request, res: Response, next: NextFunction) {
        try {
            const { id } = req.params;
            const user = await this.userService.getUserById(id);
            res.json(user);
        } catch (error) {
            next(error);
        }
    }

    async updateUser(req: Request, res: Response, next: NextFunction) {
        try {
            const { id } = req.params;
            const updates: UpdateUserDTO = req.body;
            const user = await this.userService.updateUser(id, updates);
            res.json(user);
        } catch (error) {
            next(error);
        }
    }

    async deleteUser(req: Request, res: Response, next: NextFunction) {
        try {
            const { id } = req.params;
            await this.userService.deleteUser(id);
            res.status(204).send();
        } catch (error) {
            next(error);
        }
    }
}
```

### Service Layer

```typescript
// services/user.service.ts
import { UserRepository } from '../repositories/user.repository';
import { CreateUserDTO, UpdateUserDTO, User } from '../types/user.types';
import { NotFoundError, ValidationError } from '../utils/errors';
import bcrypt from 'bcrypt';

export class UserService {
    constructor(private userRepository: UserRepository) {}

    async createUser(userData: CreateUserDTO): Promise<User> {
        // Validation
        const existingUser = await this.userRepository.findByEmail(userData.email);
        if (existingUser) {
            throw new ValidationError('Email already exists');
        }

        // Hash password
        const hashedPassword = await bcrypt.hash(userData.password, 10);

        // Create user
        const user = await this.userRepository.create({
            ...userData,
            password: hashedPassword
        });

        // Remove password from response
        const { password, ...userWithoutPassword } = user;
        return userWithoutPassword as User;
    }

    async getUserById(id: string): Promise<User> {
        const user = await this.userRepository.findById(id);
        if (!user) {
            throw new NotFoundError('User not found');
        }
        const { password, ...userWithoutPassword } = user;
        return userWithoutPassword as User;
    }

    async updateUser(id: string, updates: UpdateUserDTO): Promise<User> {
        const user = await this.userRepository.update(id, updates);
        if (!user) {
            throw new NotFoundError('User not found');
        }
        const { password, ...userWithoutPassword } = user;
        return userWithoutPassword as User;
    }

    async deleteUser(id: string): Promise<void> {
        const deleted = await this.userRepository.delete(id);
        if (!deleted) {
            throw new NotFoundError('User not found');
        }
    }
}
```

### Repository Layer

```typescript
// repositories/user.repository.ts
import { Pool } from 'pg';
import { CreateUserDTO, UpdateUserDTO, UserEntity } from '../types/user.types';

export class UserRepository {
    constructor(private db: Pool) {}

    async create(userData: CreateUserDTO & { password: string }): Promise<UserEntity> {
        const query = `
            INSERT INTO users (name, email, password)
            VALUES ($1, $2, $3)
            RETURNING id, name, email, password, created_at, updated_at
        `;
        const { rows } = await this.db.query(query, [
            userData.name,
            userData.email,
            userData.password
        ]);
        return rows[0];
    }

    async findById(id: string): Promise<UserEntity | null> {
        const query = 'SELECT * FROM users WHERE id = $1';
        const { rows } = await this.db.query(query, [id]);
        return rows[0] || null;
    }

    async findByEmail(email: string): Promise<UserEntity | null> {
        const query = 'SELECT * FROM users WHERE email = $1';
        const { rows } = await this.db.query(query, [email]);
        return rows[0] || null;
    }

    async update(id: string, updates: UpdateUserDTO): Promise<UserEntity | null> {
        const fields = Object.keys(updates);
        const values = Object.values(updates);

        const setClause = fields
            .map((field, idx) => `${field} = $${idx + 2}`)
            .join(', ');

        const query = `
            UPDATE users
            SET ${setClause}, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            RETURNING *
        `;

        const { rows } = await this.db.query(query, [id, ...values]);
        return rows[0] || null;
    }

    async delete(id: string): Promise<boolean> {
        const query = 'DELETE FROM users WHERE id = $1';
        const { rowCount } = await this.db.query(query, [id]);
        return rowCount > 0;
    }
}
```

## Dependency Injection Container

```typescript
// di-container.ts
import { Pool } from 'pg';
import { UserRepository } from './repositories/user.repository';
import { UserService } from './services/user.service';
import { UserController } from './controllers/user.controller';

class Container {
    private instances = new Map<string, any>();

    register<T>(key: string, factory: () => T): void {
        this.instances.set(key, factory);
    }

    resolve<T>(key: string): T {
        const factory = this.instances.get(key);
        if (!factory) {
            throw new Error(`No factory registered for ${key}`);
        }
        return factory();
    }

    singleton<T>(key: string, factory: () => T): void {
        let instance: T;
        this.instances.set(key, () => {
            if (!instance) {
                instance = factory();
            }
            return instance;
        });
    }
}

export const container = new Container();

// Register dependencies
container.singleton('db', () => new Pool({
    host: process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT || '5432'),
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
}));

container.singleton('userRepository', () =>
    new UserRepository(container.resolve('db'))
);

container.singleton('userService', () =>
    new UserService(container.resolve('userRepository'))
);

container.register('userController', () =>
    new UserController(container.resolve('userService'))
);
```

## Middleware Patterns

### Authentication Middleware

```typescript
// middleware/auth.middleware.ts
import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { UnauthorizedError } from '../utils/errors';

interface JWTPayload {
    userId: string;
    email: string;
    roles?: string[];
}

declare global {
    namespace Express {
        interface Request {
            user?: JWTPayload;
        }
    }
}

export const authenticate = async (
    req: Request,
    res: Response,
    next: NextFunction
) => {
    try {
        const token = req.headers.authorization?.replace('Bearer ', '');

        if (!token) {
            throw new UnauthorizedError('No token provided');
        }

        const payload = jwt.verify(
            token,
            process.env.JWT_SECRET!
        ) as JWTPayload;

        req.user = payload;
        next();
    } catch (error) {
        next(new UnauthorizedError('Invalid token'));
    }
};

export const authorize = (...roles: string[]) => {
    return async (req: Request, res: Response, next: NextFunction) => {
        if (!req.user) {
            return next(new UnauthorizedError('Not authenticated'));
        }

        const hasRole = roles.some(role =>
            req.user?.roles?.includes(role)
        );

        if (!hasRole) {
            return next(new UnauthorizedError('Insufficient permissions'));
        }

        next();
    };
};
```

### Validation Middleware

```typescript
// middleware/validation.middleware.ts
import { Request, Response, NextFunction } from 'express';
import { AnyZodObject, ZodError } from 'zod';
import { ValidationError } from '../utils/errors';

export const validate = (schema: AnyZodObject) => {
    return async (req: Request, res: Response, next: NextFunction) => {
        try {
            await schema.parseAsync({
                body: req.body,
                query: req.query,
                params: req.params
            });
            next();
        } catch (error) {
            if (error instanceof ZodError) {
                const errors = error.errors.map(err => ({
                    field: err.path.join('.'),
                    message: err.message
                }));
                next(new ValidationError('Validation failed', errors));
            } else {
                next(error);
            }
        }
    };
};

// Usage with Zod
import { z } from 'zod';

const createUserSchema = z.object({
    body: z.object({
        name: z.string().min(1),
        email: z.string().email(),
        password: z.string().min(8)
    })
});

router.post('/users', validate(createUserSchema), userController.createUser);
```

### Rate Limiting Middleware

```typescript
// middleware/rate-limit.middleware.ts
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import Redis from 'ioredis';

const redis = new Redis({
    host: process.env.REDIS_HOST,
    port: parseInt(process.env.REDIS_PORT || '6379')
});

export const apiLimiter = rateLimit({
    store: new RedisStore({
        client: redis,
        prefix: 'rl:',
    }),
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // Limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP, please try again later',
    standardHeaders: true,
    legacyHeaders: false,
});

export const authLimiter = rateLimit({
    store: new RedisStore({
        client: redis,
        prefix: 'rl:auth:',
    }),
    windowMs: 15 * 60 * 1000,
    max: 5, // Stricter limit for auth endpoints
    skipSuccessfulRequests: true,
});
```

## Error Handling

### Custom Error Classes

```typescript
// utils/errors.ts
export class AppError extends Error {
    constructor(
        public message: string,
        public statusCode: number = 500,
        public isOperational: boolean = true
    ) {
        super(message);
        Object.setPrototypeOf(this, AppError.prototype);
        Error.captureStackTrace(this, this.constructor);
    }
}

export class ValidationError extends AppError {
    constructor(message: string, public errors?: any[]) {
        super(message, 400);
    }
}

export class NotFoundError extends AppError {
    constructor(message: string = 'Resource not found') {
        super(message, 404);
    }
}

export class UnauthorizedError extends AppError {
    constructor(message: string = 'Unauthorized') {
        super(message, 401);
    }
}

export class ForbiddenError extends AppError {
    constructor(message: string = 'Forbidden') {
        super(message, 403);
    }
}

export class ConflictError extends AppError {
    constructor(message: string) {
        super(message, 409);
    }
}
```

### Global Error Handler

```typescript
// middleware/error-handler.ts
import { Request, Response, NextFunction } from 'express';
import { AppError, ValidationError } from '../utils/errors';

export const errorHandler = (
    err: Error,
    req: Request,
    res: Response,
    next: NextFunction
) => {
    if (err instanceof AppError) {
        return res.status(err.statusCode).json({
            status: 'error',
            message: err.message,
            ...(err instanceof ValidationError && { errors: err.errors })
        });
    }

    // Log unexpected errors
    console.error('Unexpected error:', err);

    // Don't leak error details in production
    const message = process.env.NODE_ENV === 'production'
        ? 'Internal server error'
        : err.message;

    res.status(500).json({
        status: 'error',
        message
    });
};

// Async error wrapper
export const asyncHandler = (
    fn: (req: Request, res: Response, next: NextFunction) => Promise<any>
) => {
    return (req: Request, res: Response, next: NextFunction) => {
        Promise.resolve(fn(req, res, next)).catch(next);
    };
};
```

## Database Patterns

### PostgreSQL with Connection Pool

```typescript
// config/database.ts
import { Pool, PoolConfig } from 'pg';

const poolConfig: PoolConfig = {
    host: process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT || '5432'),
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    max: 20,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
};

export const pool = new Pool(poolConfig);

pool.on('connect', () => {
    console.log('Database connected');
});

pool.on('error', (err) => {
    console.error('Unexpected database error', err);
    process.exit(-1);
});

export const closeDatabase = async () => {
    await pool.end();
    console.log('Database connection closed');
};
```

### MongoDB with Mongoose

```typescript
// config/mongoose.ts
import mongoose from 'mongoose';

export const connectDB = async () => {
    try {
        await mongoose.connect(process.env.MONGODB_URI!, {
            maxPoolSize: 10,
            serverSelectionTimeoutMS: 5000,
            socketTimeoutMS: 45000,
        });

        console.log('MongoDB connected');
    } catch (error) {
        console.error('MongoDB connection error:', error);
        process.exit(1);
    }
};

mongoose.connection.on('disconnected', () => {
    console.log('MongoDB disconnected');
});

mongoose.connection.on('error', (err) => {
    console.error('MongoDB error:', err);
});

// Model example
import { Schema, model, Document } from 'mongoose';

interface IUser extends Document {
    name: string;
    email: string;
    password: string;
    createdAt: Date;
    updatedAt: Date;
}

const userSchema = new Schema<IUser>({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
}, {
    timestamps: true
});

userSchema.index({ email: 1 });

export const User = model<IUser>('User', userSchema);
```

### Transaction Pattern

```typescript
// services/order.service.ts
import { Pool } from 'pg';

export class OrderService {
    constructor(private db: Pool) {}

    async createOrder(userId: string, items: any[]) {
        const client = await this.db.connect();

        try {
            await client.query('BEGIN');

            // Create order
            const orderResult = await client.query(
                'INSERT INTO orders (user_id, total) VALUES ($1, $2) RETURNING id',
                [userId, calculateTotal(items)]
            );
            const orderId = orderResult.rows[0].id;

            // Create order items
            for (const item of items) {
                await client.query(
                    'INSERT INTO order_items (order_id, product_id, quantity, price) VALUES ($1, $2, $3, $4)',
                    [orderId, item.productId, item.quantity, item.price]
                );

                // Update inventory
                await client.query(
                    'UPDATE products SET stock = stock - $1 WHERE id = $2',
                    [item.quantity, item.productId]
                );
            }

            await client.query('COMMIT');
            return orderId;
        } catch (error) {
            await client.query('ROLLBACK');
            throw error;
        } finally {
            client.release();
        }
    }
}
```

## Caching with Redis

```typescript
// utils/cache.ts
import Redis from 'ioredis';

const redis = new Redis({
    host: process.env.REDIS_HOST,
    port: parseInt(process.env.REDIS_PORT || '6379'),
    retryStrategy: (times) => {
        const delay = Math.min(times * 50, 2000);
        return delay;
    }
});

export class CacheService {
    async get<T>(key: string): Promise<T | null> {
        const data = await redis.get(key);
        return data ? JSON.parse(data) : null;
    }

    async set(key: string, value: any, ttl?: number): Promise<void> {
        const serialized = JSON.stringify(value);
        if (ttl) {
            await redis.setex(key, ttl, serialized);
        } else {
            await redis.set(key, serialized);
        }
    }

    async delete(key: string): Promise<void> {
        await redis.del(key);
    }

    async invalidatePattern(pattern: string): Promise<void> {
        const keys = await redis.keys(pattern);
        if (keys.length > 0) {
            await redis.del(...keys);
        }
    }
}
```

## Best Practices

1. **Use TypeScript**: Type safety prevents runtime errors
2. **Layered Architecture**: Separate concerns (controller/service/repository)
3. **Dependency Injection**: Easier testing and maintenance
4. **Validate Input**: Use libraries like Zod or Joi
5. **Error Handling**: Use custom error classes and global error handler
6. **Environment Variables**: Never hardcode secrets
7. **Logging**: Use structured logging (Pino, Winston)
8. **Rate Limiting**: Prevent abuse
9. **HTTPS**: Always in production
10. **Connection Pooling**: For databases
11. **Graceful Shutdown**: Clean up resources
12. **Health Checks**: For monitoring

---

**Skill Type**: Backend - Node.js Architecture
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when building Node.js backend services
**Frameworks**: Express, Fastify, TypeScript
