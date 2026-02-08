---
name: better-auth
description: Implement authentication with Better Auth - a framework-agnostic TypeScript authentication framework. Features email/password, OAuth, 2FA, passkeys, and session management.
trigger_keywords: [better-auth, better auth, typescript auth, passkeys, magic link, oauth typescript, 2fa typescript, session management]
---

# Better Auth

Framework-agnostic TypeScript authentication with built-in email/password, social OAuth, and plugin ecosystem.

## When to Use

- Implementing auth in TypeScript/JavaScript applications
- Adding email/password or social OAuth authentication
- Setting up 2FA, passkeys, magic links
- Building multi-tenant apps with organization support
- Any framework (Next.js, Nuxt, SvelteKit, Remix, etc.)

## Core Concepts

- **Session Security**: Better Auth uses secure, httpOnly cookies by default. Sessions are cryptographically signed and can be configured for sliding expiration. Always set `BETTER_AUTH_SECRET` to a strong random value (32+ characters).

- **Defense in Depth**: Layer multiple auth methods (email/password + 2FA + rate limiting). Better Auth's plugin system enables progressive security enhancement without rewriting core auth logic.

- **Token Lifecycle**: Access tokens should be short-lived (15-30 min), refresh tokens longer (7-30 days). Better Auth handles token rotation automatically, but configure `session.expiresIn` and `session.updateAge` appropriately for your risk profile.

- **Attack Mitigation**: Enable rate limiting in production to prevent credential stuffing and brute force attacks. Use email verification to prevent account enumeration. PKCE is automatic for OAuth flows to prevent authorization code interception.

- **Secure Defaults**: Better Auth implements OWASP session management best practices out of the box, including secure cookie flags, CSRF protection, and password hashing with bcrypt. Override defaults only when you understand the security implications.

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
