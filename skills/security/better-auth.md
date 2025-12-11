---
name: better-auth
description: Implement authentication with Better Auth - a framework-agnostic TypeScript authentication framework. Features email/password, OAuth, 2FA, passkeys, and session management.
---

# Better Auth

Framework-agnostic TypeScript authentication with built-in email/password, social OAuth, and plugin ecosystem.

## When to Use

- Implementing auth in TypeScript/JavaScript applications
- Adding email/password or social OAuth authentication
- Setting up 2FA, passkeys, magic links
- Building multi-tenant apps with organization support
- Any framework (Next.js, Nuxt, SvelteKit, Remix, etc.)

## Quick Start

### Installation

```bash
npm install better-auth
```

### Environment Setup

```env
BETTER_AUTH_SECRET=<generated-secret-32-chars-min>
BETTER_AUTH_URL=http://localhost:3000
```

### Server Setup

```ts
// auth.ts
import { betterAuth } from "better-auth";

export const auth = betterAuth({
  database: { /* See database docs */ },
  emailAndPassword: {
    enabled: true,
    autoSignIn: true
  },
  socialProviders: {
    github: {
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
    }
  }
});
```

### Next.js App Router

```ts
// app/api/auth/[...all]/route.ts
import { auth } from "@/lib/auth";
import { toNextJsHandler } from "better-auth/next-js";

export const { POST, GET } = toNextJsHandler(auth);
```

### Client Setup

```ts
import { createAuthClient } from "better-auth/client";

export const authClient = createAuthClient({
  baseURL: process.env.NEXT_PUBLIC_BETTER_AUTH_URL
});
```

### Usage

```ts
// Sign up
await authClient.signUp.email({
  email: "user@example.com",
  password: "secure123",
  name: "John Doe"
});

// Sign in
await authClient.signIn.email({ email, password });

// OAuth
await authClient.signIn.social({ provider: "github" });

// Session
const { data: session } = authClient.useSession(); // React
const { data: session } = await authClient.getSession(); // Vanilla
```

## Feature Matrix

| Feature | Plugin Required | Use Case |
|---------|----------------|----------|
| Email/Password | No | Basic auth |
| OAuth | No | Social login |
| Email Verification | No | Verify emails |
| Password Reset | No | Forgot password |
| Two-Factor (2FA) | Yes (`twoFactor`) | Enhanced security |
| Passkeys/WebAuthn | Yes (`passkey`) | Passwordless |
| Magic Link | Yes (`magicLink`) | Email-based login |
| Organizations | Yes (`organization`) | Multi-tenant |
| Rate Limiting | No | Prevent abuse |

## Auth Method Selection

- **Email/Password**: Traditional auth, full control
- **OAuth**: Quick signup, minimal friction
- **Passkeys**: Passwordless, modern browsers
- **Magic Link**: Passwordless without WebAuthn complexity

## Database Schema

```bash
npx @better-auth/cli generate  # Generate schema/migrations
npx @better-auth/cli migrate   # Apply migrations
```

## Implementation Checklist

- [ ] Install `better-auth` package
- [ ] Set environment variables
- [ ] Create auth server with database config
- [ ] Run schema migration
- [ ] Mount API handler
- [ ] Create client instance
- [ ] Implement sign-up/sign-in UI
- [ ] Add session management
- [ ] Set up protected routes
- [ ] Configure email sending
- [ ] Enable rate limiting for production

## Resources

- Docs: https://www.better-auth.com/docs
- GitHub: https://github.com/better-auth/better-auth
