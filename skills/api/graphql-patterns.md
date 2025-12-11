---
name: graphql-patterns
description: Load when user needs GraphQL schema design, resolvers, N+1 problem, batching, subscriptions, or GraphQL API patterns
trigger_keywords: [graphql, schema, resolver, query, mutation, subscription, n+1 problem, dataloader, batching, graphql api, apollo, strawberry, graphene]
---

# GraphQL Patterns Skill

Production-grade GraphQL API design with schema-first patterns, efficient resolvers, N+1 solution strategies, and real-time subscriptions.

## Overview

GraphQL provides a strongly-typed query language for APIs, enabling clients to request exactly the data they need. Solves over-fetching and under-fetching problems of REST while introducing new challenges (N+1, complexity analysis).

**When to Use**:
- Clients need flexible data fetching (mobile, web with varying needs)
- Reducing API round-trips and bandwidth
- Strong typing and schema introspection required
- Real-time updates with subscriptions

## Core Concepts

### Schema Definition Language (SDL)

**Basic Types**:
```graphql
# Object types
type User {
  id: ID!           # Non-null ID
  email: String!
  name: String!
  age: Int
  isActive: Boolean!
  createdAt: DateTime!
}

# Enums
enum Role {
  ADMIN
  USER
  GUEST
}

# Input types (for mutations)
input CreateUserInput {
  email: String!
  name: String!
  age: Int
}

# Interfaces
interface Node {
  id: ID!
}

type User implements Node {
  id: ID!
  email: String!
}
```

**Queries and Mutations**:
```graphql
type Query {
  # Fetch single user
  user(id: ID!): User

  # List users with pagination
  users(
    first: Int = 10
    after: String
    filter: UserFilter
  ): UserConnection!

  # Search
  searchUsers(query: String!): [User!]!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(id: ID!, input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeleteUserPayload!
}

type CreateUserPayload {
  user: User
  errors: [UserError!]
}
```

**Subscriptions**:
```graphql
type Subscription {
  userCreated: User!
  userUpdated(id: ID!): User!
  messageReceived(chatId: ID!): Message!
}
```

### Relay Cursor Pagination

```graphql
type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type UserEdge {
  cursor: String!
  node: User!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

# Query
query {
  users(first: 10, after: "cursor123") {
    edges {
      cursor
      node {
        id
        name
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

## Python Implementation (Strawberry)

### Basic Setup

```python
import strawberry
from typing import Optional, List
from datetime import datetime

@strawberry.type
class User:
    id: strawberry.ID
    email: str
    name: str
    age: Optional[int] = None
    is_active: bool = True
    created_at: datetime

@strawberry.input
class CreateUserInput:
    email: str
    name: str
    age: Optional[int] = None

@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: strawberry.ID) -> Optional[User]:
        return get_user_by_id(id)

    @strawberry.field
    def users(self) -> List[User]:
        return get_all_users()

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_user(self, input: CreateUserInput) -> User:
        return create_user_in_db(input)

schema = strawberry.Schema(query=Query, mutation=Mutation)
```

### Resolvers with DataLoader (N+1 Solution)

**The N+1 Problem**:
```python
# ❌ N+1 queries (1 query for posts, N queries for authors)
type Post {
  id: ID!
  title: String!
  author: User!  # Separate query per post
}

# If fetching 100 posts → 101 database queries!
```

**Solution: DataLoader**:
```python
from strawberry.dataloader import DataLoader
from typing import List
import asyncio

# Batch loading function
async def load_users(keys: List[int]) -> List[User]:
    """Load multiple users in single query"""
    # SELECT * FROM users WHERE id IN (1, 2, 3, ...)
    users = await db.query(User).filter(User.id.in_(keys)).all()
    user_map = {user.id: user for user in users}
    return [user_map.get(key) for key in keys]

# Create DataLoader
user_loader = DataLoader(load_fn=load_users)

@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    author_id: strawberry.Private[int]

    @strawberry.field
    async def author(self) -> User:
        # DataLoader batches and caches requests
        return await user_loader.load(self.author_id)

# Fetching 100 posts → 2 queries (1 for posts, 1 batched for authors)
```

### Field-Level Resolvers

```python
@strawberry.type
class User:
    id: strawberry.ID
    email: str
    name: str

    @strawberry.field
    async def full_name(self) -> str:
        """Computed field"""
        return f"{self.first_name} {self.last_name}"

    @strawberry.field
    async def posts(
        self,
        info: strawberry.Info,
        first: int = 10
    ) -> List["Post"]:
        """Resolve nested relationship"""
        post_loader = info.context["post_loader"]
        return await post_loader.load(self.id)

    @strawberry.field
    def is_admin(self, info: strawberry.Info) -> bool:
        """Access context (current user)"""
        current_user = info.context["current_user"]
        return self.id == current_user.id and self.role == "ADMIN"
```

### Authentication & Authorization

```python
from strawberry.permission import BasePermission
from strawberry.types import Info

class IsAuthenticated(BasePermission):
    message = "User is not authenticated"

    def has_permission(self, source, info: Info, **kwargs) -> bool:
        return info.context.get("current_user") is not None

class IsAdmin(BasePermission):
    message = "User is not an admin"

    def has_permission(self, source, info: Info, **kwargs) -> bool:
        user = info.context.get("current_user")
        return user and user.role == "ADMIN"

@strawberry.type
class Query:
    @strawberry.field(permission_classes=[IsAuthenticated])
    def me(self, info: Info) -> User:
        return info.context["current_user"]

    @strawberry.field(permission_classes=[IsAdmin])
    def all_users(self) -> List[User]:
        return get_all_users()
```

### Error Handling

```python
from typing import Union, List
import strawberry

@strawberry.type
class UserError:
    field: str
    message: str

@strawberry.type
class CreateUserSuccess:
    user: User

@strawberry.type
class CreateUserError:
    errors: List[UserError]

CreateUserResult = strawberry.union(
    "CreateUserResult",
    (CreateUserSuccess, CreateUserError)
)

@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_user(self, input: CreateUserInput) -> CreateUserResult:
        # Validate
        errors = validate_user_input(input)
        if errors:
            return CreateUserError(errors=errors)

        # Create user
        user = create_user_in_db(input)
        return CreateUserSuccess(user=user)

# Client query
"""
mutation {
  createUser(input: {email: "test@example.com", name: "Test"}) {
    ... on CreateUserSuccess {
      user { id name }
    }
    ... on CreateUserError {
      errors { field message }
    }
  }
}
"""
```

## Advanced Patterns

### Subscriptions (Real-Time)

```python
import asyncio
from typing import AsyncGenerator
import strawberry

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def user_created(self) -> AsyncGenerator[User, None]:
        """Stream new users as they're created"""
        queue = asyncio.Queue()
        # Register queue with event system
        user_events.register(queue)

        try:
            while True:
                user = await queue.get()
                yield user
        finally:
            user_events.unregister(queue)

    @strawberry.subscription
    async def message_received(
        self,
        chat_id: strawberry.ID,
        info: Info
    ) -> AsyncGenerator[Message, None]:
        """Stream messages for specific chat"""
        user = info.context["current_user"]
        if not has_access_to_chat(user, chat_id):
            raise PermissionError("Access denied")

        queue = asyncio.Queue()
        message_events.register(chat_id, queue)

        try:
            while True:
                message = await queue.get()
                yield message
        finally:
            message_events.unregister(chat_id, queue)
```

### Complexity Analysis (Prevent DoS)

```python
from strawberry.extensions import QueryDepthLimiter, MaxTokensLimiter

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[
        QueryDepthLimiter(max_depth=10),      # Prevent deeply nested queries
        MaxTokensLimiter(max_token_count=1000)  # Limit query size
    ]
)

# Blocked query (depth > 10):
# query {
#   user {
#     posts {
#       author {
#         posts {
#           author { ... }  # Too deep
#         }
#       }
#     }
#   }
# }
```

### Custom Scalars

```python
from datetime import datetime, date
import strawberry

@strawberry.scalar(
    serialize=lambda v: v.isoformat(),
    parse_value=lambda v: datetime.fromisoformat(v)
)
class DateTime:
    __slots__ = ()

@strawberry.scalar(
    serialize=lambda v: v.isoformat(),
    parse_value=lambda v: date.fromisoformat(v)
)
class Date:
    __slots__ = ()

@strawberry.type
class Event:
    id: strawberry.ID
    name: str
    scheduled_at: DateTime
    event_date: Date
```

### File Uploads

```python
from strawberry.file_uploads import Upload

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def upload_avatar(
        self,
        file: Upload,
        info: Info
    ) -> User:
        user = info.context["current_user"]
        contents = await file.read()

        # Save to S3/storage
        avatar_url = await storage.upload(contents, file.filename)

        # Update user
        user.avatar_url = avatar_url
        await db.save(user)

        return user

# Client mutation (multipart form data)
"""
mutation($file: Upload!) {
  uploadAvatar(file: $file) {
    id
    avatarUrl
  }
}
"""
```

## Production-Ready Examples

### FastAPI Integration

```python
from fastapi import FastAPI, Depends
from strawberry.fastapi import GraphQLRouter
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

async def get_context(session: AsyncSession = Depends(get_db_session)):
    """Provide context to GraphQL resolvers"""
    return {
        "db": session,
        "user_loader": DataLoader(load_fn=load_users),
        "post_loader": DataLoader(load_fn=load_posts),
    }

graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
)

app.include_router(graphql_app, prefix="/graphql")
```

### Pagination (Relay Cursor)

```python
import base64
from typing import List, Optional
import strawberry

def encode_cursor(value: str) -> str:
    """Encode cursor as base64"""
    return base64.b64encode(value.encode()).decode()

def decode_cursor(cursor: str) -> str:
    """Decode base64 cursor"""
    return base64.b64decode(cursor.encode()).decode()

@strawberry.type
class PageInfo:
    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str]
    end_cursor: Optional[str]

@strawberry.type
class UserEdge:
    cursor: str
    node: User

@strawberry.type
class UserConnection:
    edges: List[UserEdge]
    page_info: PageInfo
    total_count: int

@strawberry.type
class Query:
    @strawberry.field
    async def users(
        self,
        first: int = 10,
        after: Optional[str] = None
    ) -> UserConnection:
        # Decode cursor
        offset = 0
        if after:
            offset = int(decode_cursor(after))

        # Fetch users (offset + 1 to check has_next_page)
        users = await db.query(User).offset(offset).limit(first + 1).all()

        has_next_page = len(users) > first
        users = users[:first]

        # Build edges
        edges = [
            UserEdge(
                cursor=encode_cursor(str(offset + i)),
                node=user
            )
            for i, user in enumerate(users)
        ]

        # Build page info
        page_info = PageInfo(
            has_next_page=has_next_page,
            has_previous_page=offset > 0,
            start_cursor=edges[0].cursor if edges else None,
            end_cursor=edges[-1].cursor if edges else None
        )

        total_count = await db.query(User).count()

        return UserConnection(
            edges=edges,
            page_info=page_info,
            total_count=total_count
        )
```

### Batched Mutations

```python
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_users(
        self,
        inputs: List[CreateUserInput]
    ) -> List[User]:
        """Batch create multiple users"""
        users = [
            User(
                email=input.email,
                name=input.name,
                age=input.age
            )
            for input in inputs
        ]

        # Bulk insert
        await db.bulk_insert(users)
        return users
```

## Best Practices

### 1. Schema-First Design
```graphql
# ✅ Design schema before implementation
type User {
  id: ID!
  email: String!
  fullName: String!  # Abstraction, not direct DB field
  posts(first: Int): PostConnection!
}
```

### 2. Use DataLoader for N+1
```python
# ✅ Always batch related data
@strawberry.field
async def author(self, info: Info) -> User:
    return await info.context["user_loader"].load(self.author_id)
```

### 3. Limit Query Depth/Complexity
```python
# ✅ Prevent malicious queries
schema = strawberry.Schema(
    query=Query,
    extensions=[QueryDepthLimiter(max_depth=10)]
)
```

### 4. Explicit Error Types
```python
# ✅ Typed errors (not just strings)
@strawberry.type
class ValidationError:
    field: str
    message: str
```

### 5. Paginate Lists
```python
# ❌ Unbounded list
@strawberry.field
def users(self) -> List[User]:  # Could return millions
    return get_all_users()

# ✅ Paginated
@strawberry.field
def users(self, first: int = 10, after: Optional[str] = None) -> UserConnection:
    ...
```

## Common Pitfalls

❌ **N+1 queries without DataLoader**
```python
# ❌ Separate query per item
@strawberry.field
def author(self) -> User:
    return db.query(User).filter_by(id=self.author_id).first()
```

❌ **Exposing database structure directly**
```graphql
# ❌ Exposes implementation details
type User {
  id: ID!
  first_name: String  # DB column name
  last_name: String
  created_at: String
}
```
✅ Use domain language
```graphql
type User {
  id: ID!
  fullName: String
  createdAt: DateTime
}
```

❌ **No complexity limits**
```graphql
# ❌ Allows DoS attacks
query {
  users { posts { comments { author { posts { ... } } } } }
}
```

❌ **Returning null for errors**
```python
# ❌ Silent failures
@strawberry.mutation
def create_user(self, input: CreateUserInput) -> Optional[User]:
    if not valid:
        return None  # Client doesn't know why
```

## Quality Standards

- **N+1 Prevention**: Use DataLoader for all relationships
- **Complexity Limits**: Max depth ≤10, max tokens ≤1000
- **Pagination**: All lists must support cursor pagination
- **Error Handling**: Explicit error types in union results
- **Authorization**: Field-level permissions for sensitive data
- **Documentation**: Schema descriptions for all types and fields

---

**Skill Type**: API - GraphQL
**Complexity**: Moderate to Advanced
**Typical Usage**: Activated when designing flexible, type-safe GraphQL APIs
**Performance**: DataLoader reduces N+1 queries to O(1) batched queries
