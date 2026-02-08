---
description: Fetch documentation for libraries, frameworks, and APIs
args: <library-name> [topic]
tools: Task
model: sonnet
---

# Documentation Fetcher

Fetch documentation for: **$ARGUMENTS**

## Documentation Strategy

### Step 1: Library Identification
Parse the request to identify:
- Library/framework name (e.g., React, Django, Express)
- Specific topic or API (e.g., hooks, routing, middleware)
- Version if specified (e.g., React 18, Vue 3)

### Step 2: Source Priority

Search documentation in order:
1. **Official Documentation**
   - Project website (e.g., reactjs.org, djangoproject.com)
   - GitHub repository README and docs/
   - NPM/PyPI/crates.io package pages

2. **API References**
   - Generated API docs (JSDoc, Sphinx, RustDoc)
   - TypeScript definitions
   - OpenAPI/Swagger specifications

3. **Community Resources**
   - MDN Web Docs for web APIs
   - DevDocs.io for aggregated docs
   - Stack Overflow documentation

4. **Examples & Tutorials**
   - Official examples repository
   - Getting started guides
   - Best practices documentation

### Step 3: Content Extraction

Focus on extracting:
- **Quick Start**: Installation and setup
- **Core Concepts**: Key abstractions and patterns
- **API Reference**: Methods, parameters, return values
- **Examples**: Working code samples
- **Best Practices**: Recommended patterns
- **Troubleshooting**: Common issues and solutions

## Popular Library Shortcuts

### Frontend Frameworks
- `react` → React hooks, components, state management
- `vue` → Vue 3 composition API, directives, reactivity
- `angular` → Angular CLI, services, RxJS
- `svelte` → SvelteKit, stores, transitions
- `next` → Next.js routing, API routes, SSR/SSG

### Backend Frameworks
- `express` → Express middleware, routing, error handling
- `django` → Django ORM, views, admin
- `fastapi` → FastAPI schemas, dependency injection
- `rails` → Ruby on Rails ActiveRecord, controllers
- `spring` → Spring Boot, dependency injection

### Databases
- `postgres` → PostgreSQL queries, indexes, extensions
- `mongodb` → MongoDB aggregation, indexes, replication
- `redis` → Redis data types, pub/sub, persistence
- `elasticsearch` → Elasticsearch queries, mappings

### Cloud Platforms
- `aws` → AWS services (S3, Lambda, DynamoDB, etc.)
- `gcp` → Google Cloud Platform services
- `azure` → Microsoft Azure services
- `vercel` → Vercel deployment, edge functions
- `netlify` → Netlify functions, forms, identity

### Development Tools
- `git` → Git commands, workflows, configuration
- `docker` → Dockerfile, compose, networking
- `kubernetes` → K8s resources, kubectl commands
- `terraform` → Terraform providers, modules
- `github-actions` → GitHub Actions workflow syntax

## Advanced Documentation Features

### Version-Specific Docs
```
/docs "react 17" "concurrent features"
/docs "python 3.11" "new features"
/docs "node 20" "performance improvements"
```

### Migration Guides
```
/docs "vue 2 to 3 migration"
/docs "angular upgrade guide"
/docs "django 3 to 4"
```

### Comparison Docs
```
/docs "react vs vue"
/docs "postgresql vs mysql"
/docs "aws lambda vs google cloud functions"
```

## Output Format

Present documentation as:
1. **Summary**: Brief overview of the library/topic
2. **Installation**: Quick setup instructions
3. **Core API**: Most important methods/functions
4. **Examples**: 2-3 practical code samples
5. **Links**: Official docs, tutorials, references

## Offline Fallback

If online docs unavailable:
1. Check local node_modules for README
2. Parse package.json for repository URL
3. Use `Task` to invoke research-librarian
4. Search local documentation cache

## Integration with Development

After fetching docs:
1. Suggest relevant code examples
2. Identify applicable patterns for current project
3. Highlight breaking changes or deprecations
4. Recommend best practices for implementation

## Examples

```
/docs react hooks
/docs fastapi "dependency injection"
/docs aws s3 "presigned urls"
/docs typescript "generics"
/docs tailwind "custom utilities"
```

## Follow-up Actions

- Save important docs to project documentation
- Create code snippets from examples
- Update dependencies if using outdated version
- Share findings with team