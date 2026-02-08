---
name: create-feature
description: Load when user needs to implement new features with proper planning, architecture, testing, and documentation
trigger_keywords: [feature, implement, add, create, build, new functionality, user story, requirement, epic, sprint]
---

# Feature Development Workflow

End-to-end feature development workflow from requirements to deployment, ensuring consistent quality and maintainability.

## Core Concepts

### Feature Development Lifecycle

```yaml
1. Requirements
   - Clarify user story/acceptance criteria
   - Identify edge cases and constraints
   - Define success metrics

2. Design
   - Architecture decision (if needed)
   - API contract design
   - Data model changes
   - UI/UX wireframes (if applicable)

3. Implementation
   - Create feature branch
   - Build in vertical slices
   - Write tests alongside code
   - Regular commits with clear messages

4. Review & Testing
   - Self-review checklist
   - Code review
   - QA testing
   - Performance validation

5. Deployment
   - Merge to main
   - Feature flag (if gradual rollout)
   - Monitor and iterate
```

### Complexity Estimation

| Size | Scope | Timeline | Approach |
|------|-------|----------|----------|
| **Small** | 1 file, <100 LOC | 1-2 hours | Direct implementation |
| **Medium** | 2-5 files, 100-500 LOC | 1-2 days | Design doc optional |
| **Large** | 5+ files, >500 LOC | 1+ week | Full design review |
| **Epic** | Cross-service, new domain | 2+ weeks | ADR + phased delivery |

## Implementation Patterns

### 1. Feature Branch Workflow

```bash
#!/bin/bash
# Create feature branch with proper naming

FEATURE_NAME="$1"
TICKET_ID="${2:-}"

if [ -z "$FEATURE_NAME" ]; then
  echo "Usage: ./create-feature.sh <feature-name> [ticket-id]"
  exit 1
fi

# Format: feature/TICKET-123-brief-description
if [ -n "$TICKET_ID" ]; then
  BRANCH_NAME="feature/${TICKET_ID}-${FEATURE_NAME}"
else
  BRANCH_NAME="feature/${FEATURE_NAME}"
fi

# Ensure we're up to date
git fetch origin
git checkout main
git pull origin main

# Create and switch to feature branch
git checkout -b "$BRANCH_NAME"

echo "Created branch: $BRANCH_NAME"
echo "Ready to implement feature: $FEATURE_NAME"
```

### 2. Feature Planning Template

```markdown
## Feature: [Feature Name]

### User Story
As a [user type], I want to [action] so that [benefit].

### Acceptance Criteria
- [ ] Given [context], when [action], then [expected result]
- [ ] Given [context], when [action], then [expected result]
- [ ] Given [context], when [action], then [expected result]

### Technical Design

#### Architecture Changes
- [ ] New components: [list]
- [ ] Modified components: [list]
- [ ] Database changes: [migrations]
- [ ] API changes: [endpoints]

#### Data Flow
```
User Action → API Endpoint → Service → Repository → Database
                    ↓
              Response → UI Update
```

#### Dependencies
- External services: [list]
- Internal services: [list]
- Libraries: [list]

### Implementation Plan

#### Phase 1: Foundation
- [ ] Create database migrations
- [ ] Add API endpoint stubs
- [ ] Set up feature flag

#### Phase 2: Core Logic
- [ ] Implement service layer
- [ ] Add business logic
- [ ] Write unit tests

#### Phase 3: Integration
- [ ] Connect frontend to API
- [ ] Add integration tests
- [ ] Update documentation

### Testing Strategy
- Unit tests: [scope]
- Integration tests: [scope]
- E2E tests: [critical paths]

### Rollout Plan
- [ ] Deploy behind feature flag
- [ ] Enable for 10% of users
- [ ] Monitor error rates
- [ ] Gradual rollout to 100%

### Success Metrics
- [Metric 1]: [target]
- [Metric 2]: [target]
```

### 3. Vertical Slice Implementation

```typescript
/**
 * Vertical slice: Implement one complete user flow at a time
 * Instead of: All models → All services → All controllers → All UI
 * Do: Feature 1 complete → Feature 2 complete → Feature 3 complete
 */

// Slice 1: User can view their profile
// ─────────────────────────────────────

// 1. Database model (if new)
interface UserProfile {
  id: string;
  email: string;
  displayName: string;
  avatarUrl?: string;
}

// 2. Repository
async function getUserProfile(userId: string): Promise<UserProfile> {
  return await db.userProfiles.findUnique({ where: { id: userId } });
}

// 3. Service
async function getProfile(userId: string): Promise<UserProfile> {
  const profile = await getUserProfile(userId);
  if (!profile) throw new NotFoundError('Profile not found');
  return profile;
}

// 4. API endpoint
router.get('/api/users/:id/profile', async (req, res) => {
  const profile = await getProfile(req.params.id);
  res.json(profile);
});

// 5. UI component
function ProfilePage({ userId }: { userId: string }) {
  const { data: profile, isLoading } = useQuery(['profile', userId],
    () => fetchProfile(userId));

  if (isLoading) return <Spinner />;
  return <ProfileCard profile={profile} />;
}

// 6. Tests
describe('Profile Feature', () => {
  it('displays user profile', async () => {
    render(<ProfilePage userId="123" />);
    expect(await screen.findByText('John Doe')).toBeInTheDocument();
  });
});
```

### 4. Feature Flag Pattern

```typescript
// Feature flag configuration
interface FeatureFlags {
  newProfilePage: boolean;
  darkMode: boolean;
  betaFeatures: boolean;
}

// Check feature flag
function isFeatureEnabled(flag: keyof FeatureFlags, userId?: string): boolean {
  const flags = getFeatureFlags();
  const flagConfig = flags[flag];

  if (typeof flagConfig === 'boolean') return flagConfig;

  // Percentage rollout
  if (flagConfig.percentage && userId) {
    return hashUser(userId) < flagConfig.percentage;
  }

  // User allowlist
  if (flagConfig.allowlist?.includes(userId)) return true;

  return flagConfig.default ?? false;
}

// Usage in code
if (isFeatureEnabled('newProfilePage', user.id)) {
  return <NewProfilePage />;
} else {
  return <LegacyProfilePage />;
}
```

### 5. PR Template for Features

```markdown
## Summary
[2-3 sentences describing what this PR does]

## Related Issues
- Closes #[issue_number]
- Related to #[issue_number]

## Changes Made
- [Change 1]
- [Change 2]
- [Change 3]

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No console.log or debug statements
- [ ] Feature flag added (if applicable)
- [ ] Database migrations are reversible

## Screenshots (if UI changes)
| Before | After |
|--------|-------|
| [img]  | [img] |

## Deployment Notes
[Any special deployment considerations]
```

## Best Practices

### Planning Phase

```yaml
DO:
  - Write clear acceptance criteria before coding
  - Break large features into shippable increments
  - Identify dependencies and blockers early
  - Get design review for architectural changes

DON'T:
  - Start coding without clear requirements
  - Plan for months without shipping
  - Skip edge case analysis
  - Ignore non-functional requirements
```

### Implementation Phase

```yaml
DO:
  - Commit frequently with clear messages
  - Write tests as you go
  - Keep PRs small (<400 LOC ideally)
  - Ask for early feedback on approach

DON'T:
  - Create massive PRs that are hard to review
  - Leave tests until the end
  - Ignore linting/type errors
  - Hard-code values that should be configurable
```

### Review Phase

```yaml
DO:
  - Self-review before requesting review
  - Respond to feedback promptly
  - Test in staging environment
  - Update documentation

DON'T:
  - Merge without approval
  - Skip QA for "simple" features
  - Ignore edge cases found in review
  - Rush to meet arbitrary deadlines
```

## Self-Review Checklist

Before requesting review, verify:

```markdown
## Code Quality
- [ ] No commented-out code
- [ ] No TODO comments without issue links
- [ ] Error handling is comprehensive
- [ ] Edge cases are handled

## Testing
- [ ] Unit tests for business logic
- [ ] Integration tests for API endpoints
- [ ] Tests are readable and maintainable
- [ ] Test coverage is adequate (>80%)

## Security
- [ ] No secrets in code
- [ ] Input validation present
- [ ] Authorization checks in place
- [ ] SQL injection prevention

## Performance
- [ ] No N+1 queries
- [ ] Indexes added for new queries
- [ ] Large lists paginated
- [ ] Expensive operations cached

## Documentation
- [ ] API endpoints documented
- [ ] Complex logic commented
- [ ] README updated if needed
- [ ] Changelog entry added
```

## Quality Standards

- **Requirements**: Clear, testable acceptance criteria
- **Design**: Reviewed for features touching >3 files
- **Code**: Passes linting, type checking, tests
- **Coverage**: >80% for new code
- **Documentation**: API docs, inline comments for complex logic
- **Performance**: No regressions, meets SLA targets

---

**Skill Type**: Workflow - Development
**Complexity**: Moderate
**Typical Usage**: Activated when implementing new features
**Tools**: Git, issue trackers, CI/CD, testing frameworks
