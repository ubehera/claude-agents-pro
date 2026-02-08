---
name: nextjs-patterns
description: Load when user needs Next.js App Router patterns including server actions, route handlers, middleware, and streaming
trigger_keywords: [nextjs, next.js, app router, server action, route handler, middleware, next auth, next config, page.tsx, layout.tsx, loading.tsx]
---

# Next.js Patterns Skill

Production patterns for Next.js 14+ App Router including server actions, route handlers, middleware, caching, and deployment.

## Overview

Next.js App Router uses React Server Components by default, with file-system routing, nested layouts, and built-in data fetching. This skill covers patterns that maximize App Router capabilities.

**When to Use**:
- Building full-stack Next.js applications
- Implementing server actions for mutations
- Configuring caching and revalidation strategies
- Setting up authentication middleware

## Core Concepts

### File-System Routing

```
app/
├── layout.tsx          # Root layout (wraps all pages)
├── page.tsx            # Home page (/)
├── loading.tsx         # Loading UI (Suspense boundary)
├── error.tsx           # Error boundary
├── not-found.tsx       # 404 page
├── dashboard/
│   ├── layout.tsx      # Dashboard layout (nested)
│   ├── page.tsx        # /dashboard
│   └── settings/
│       └── page.tsx    # /dashboard/settings
├── blog/
│   ├── page.tsx        # /blog (list)
│   └── [slug]/
│       └── page.tsx    # /blog/my-post (dynamic)
└── api/
    └── webhooks/
        └── route.ts    # API route handler
```

### Server Actions

```tsx
// app/actions.ts
'use server';

import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';
import { z } from 'zod';

const CreatePostSchema = z.object({
  title: z.string().min(1).max(200),
  content: z.string().min(1),
});

export async function createPost(formData: FormData) {
  const parsed = CreatePostSchema.safeParse({
    title: formData.get('title'),
    content: formData.get('content'),
  });

  if (!parsed.success) {
    return { error: parsed.error.flatten().fieldErrors };
  }

  const post = await db.posts.create({ data: parsed.data });
  revalidatePath('/blog');
  redirect(`/blog/${post.slug}`);
}

// app/blog/new/page.tsx — Server Component using server action
import { createPost } from '../actions';

export default function NewPost() {
  return (
    <form action={createPost}>
      <input name="title" placeholder="Post title" required />
      <textarea name="content" placeholder="Content" required />
      <button type="submit">Publish</button>
    </form>
  );
}
```

### Route Handlers (API Routes)

```tsx
// app/api/posts/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const page = parseInt(searchParams.get('page') ?? '1');
  const limit = 20;

  const posts = await db.posts.findMany({
    skip: (page - 1) * limit,
    take: limit,
    orderBy: { createdAt: 'desc' },
  });

  return NextResponse.json({ posts, page, limit });
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const post = await db.posts.create({ data: body });
  return NextResponse.json(post, { status: 201 });
}
```

### Middleware

```tsx
// middleware.ts (project root)
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const token = request.cookies.get('session-token')?.value;

  // Protect dashboard routes
  if (request.nextUrl.pathname.startsWith('/dashboard')) {
    if (!token) {
      return NextResponse.redirect(new URL('/login', request.url));
    }
  }

  // Add security headers
  const response = NextResponse.next();
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  return response;
}

export const config = {
  matcher: ['/dashboard/:path*', '/api/:path*'],
};
```

## Caching & Revalidation

### Data Fetching with Cache Control

```tsx
// Default: cached forever (static)
const posts = await fetch('https://api.example.com/posts');

// Revalidate every 60 seconds (ISR)
const posts = await fetch('https://api.example.com/posts', {
  next: { revalidate: 60 },
});

// Never cache (dynamic)
const user = await fetch('https://api.example.com/me', {
  cache: 'no-store',
});

// On-demand revalidation via server action
'use server';
import { revalidateTag, revalidatePath } from 'next/cache';

export async function publishPost(id: string) {
  await db.posts.update({ where: { id }, data: { published: true } });
  revalidateTag('posts');        // Invalidate by tag
  revalidatePath('/blog');       // Invalidate by path
}
```

### Streaming with Loading States

```tsx
// app/dashboard/loading.tsx — automatic Suspense boundary
export default function Loading() {
  return <DashboardSkeleton />;
}

// app/dashboard/page.tsx — streams when ready
export default async function Dashboard() {
  const data = await getExpensiveData(); // Streams when resolved
  return <DashboardContent data={data} />;
}
```

## Patterns

### Parallel Data Fetching

```tsx
// Fetch in parallel — don't waterfall
export default async function Dashboard() {
  const [revenue, orders, users] = await Promise.all([
    getRevenue(),
    getRecentOrders(),
    getActiveUsers(),
  ]);

  return (
    <div className="grid grid-cols-3 gap-4">
      <RevenueCard data={revenue} />
      <OrdersTable data={orders} />
      <UsersChart data={users} />
    </div>
  );
}
```

### Optimistic Updates with useOptimistic

```tsx
'use client';
import { useOptimistic } from 'react';
import { toggleLike } from './actions';

function LikeButton({ post }: { post: Post }) {
  const [optimisticLiked, setOptimisticLiked] = useOptimistic(
    post.isLiked,
    (_, newState: boolean) => newState
  );

  return (
    <form action={async () => {
      setOptimisticLiked(!optimisticLiked);
      await toggleLike(post.id);
    }}>
      <button>{optimisticLiked ? '❤️' : '🤍'} {post.likes}</button>
    </form>
  );
}
```

## Best Practices

1. **Server Components by default** — add `'use client'` only for interactivity
2. **Server Actions for mutations** — replace API routes for form submissions
3. **Parallel data fetching** — `Promise.all` to avoid waterfalls
4. **Granular Suspense** — wrap each async section independently
5. **Revalidate on mutation** — `revalidatePath` / `revalidateTag` after writes
6. **Middleware for auth** — check session before rendering protected pages

---

**Skill Type**: TypeScript — Next.js
**Complexity**: Moderate
**Typical Usage**: Next.js App Router development, server actions, caching strategies
