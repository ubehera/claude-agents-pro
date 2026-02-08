---
name: e2e-testing
description: Load when implementing end-to-end tests with Playwright or Cypress, debugging flaky tests, or establishing E2E testing standards for critical user workflows
trigger_keywords: [e2e, end-to-end, playwright, cypress, browser testing, user flow, integration testing, flaky test, visual regression, accessibility testing]
---

# E2E Testing Patterns

Production-grade end-to-end testing with Playwright and Cypress for reliable, maintainable test suites that catch bugs before users do.

## Overview

End-to-end testing validates complete user workflows from UI to backend, ensuring critical paths work correctly across browsers and devices.

**When to Use**:
- Testing critical user journeys (authentication, checkout, signup)
- Cross-browser compatibility validation
- Complex multi-step workflows
- Real API integration testing
- Accessibility and visual regression testing

**When NOT to Use**:
- Unit-level logic (use unit tests)
- API contract validation (use integration tests)
- Edge cases requiring many permutations (too slow)

## Core Concepts

- **Test User Behavior, Not Implementation**: E2E tests validate what users experience, not how code works internally - use semantic selectors (roles, labels, data-testid) over CSS classes
- **Test Independence**: Each test must set up its own data, run without dependencies on other tests, and clean up after itself - shared state causes flaky suites
- **Page Object Model**: Encapsulate page interactions in reusable classes to reduce duplication and isolate selector changes to single locations
- **Smart Waiting**: Use auto-waiting assertions (`expect(element).toBeVisible()`) and network idle states instead of fixed timeouts - condition-based waits eliminate flakiness
- **Critical Paths Only**: E2E tests are slow and expensive - focus on 20-30 critical user journeys (login, checkout, signup), not comprehensive coverage

## The Testing Pyramid

```
        /\
       /E2E\         ← 10%: Critical paths only
      /─────\
     /Integr\        ← 20%: Service interactions
    /────────\
   /Unit Tests\      ← 70%: Fast, isolated, extensive
  /────────────\
```

**E2E Test Principles**:
- Test user behavior, not implementation
- Keep tests independent and deterministic
- Optimize for speed (mock when appropriate)
- Use semantic selectors (data-testid, ARIA roles)
- Clean up test data after each test

## Playwright Patterns

### Configuration

```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
    testDir: './e2e',
    timeout: 30000,
    expect: { timeout: 5000 },
    fullyParallel: true,
    forbidOnly: !!process.env.CI,
    retries: process.env.CI ? 2 : 0,
    workers: process.env.CI ? 1 : undefined,

    use: {
        baseURL: 'http://localhost:3000',
        trace: 'on-first-retry',
        screenshot: 'only-on-failure',
        video: 'retain-on-failure',
    },

    projects: [
        { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
        { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
        { name: 'webkit', use: { ...devices['Desktop Safari'] } },
        { name: 'mobile', use: { ...devices['iPhone 13'] } },
    ],
});
```

### Page Object Model

```typescript
// pages/LoginPage.ts
import { Page, Locator } from '@playwright/test';

export class LoginPage {
    readonly page: Page;
    readonly emailInput: Locator;
    readonly passwordInput: Locator;
    readonly loginButton: Locator;
    readonly errorMessage: Locator;

    constructor(page: Page) {
        this.page = page;
        this.emailInput = page.getByLabel('Email');
        this.passwordInput = page.getByLabel('Password');
        this.loginButton = page.getByRole('button', { name: 'Login' });
        this.errorMessage = page.getByRole('alert');
    }

    async goto() {
        await this.page.goto('/login');
    }

    async login(email: string, password: string) {
        await this.emailInput.fill(email);
        await this.passwordInput.fill(password);
        await this.loginButton.click();
    }

    async getErrorMessage(): Promise<string> {
        return await this.errorMessage.textContent() ?? '';
    }
}

// Test using Page Object
import { test, expect } from '@playwright/test';

test('successful login redirects to dashboard', async ({ page }) => {
    const loginPage = new LoginPage(page);
    await loginPage.goto();
    await loginPage.login('user@example.com', 'password123');

    await expect(page).toHaveURL('/dashboard');
    await expect(page.getByRole('heading', { name: 'Dashboard' }))
        .toBeVisible();
});
```

### Test Fixtures for Data Management

```typescript
// fixtures/test-data.ts
import { test as base } from '@playwright/test';

type TestData = {
    testUser: {
        email: string;
        password: string;
        name: string;
    };
};

export const test = base.extend<TestData>({
    testUser: async ({}, use) => {
        const user = {
            email: `test-${Date.now()}@example.com`,
            password: 'Test123!@#',
            name: 'Test User',
        };

        // Setup: Create user in database
        await createTestUser(user);

        await use(user);

        // Teardown: Clean up user
        await deleteTestUser(user.email);
    },
});

// Usage in tests
test('user can update profile', async ({ page, testUser }) => {
    await page.goto('/login');
    await page.getByLabel('Email').fill(testUser.email);
    await page.getByLabel('Password').fill(testUser.password);
    await page.getByRole('button', { name: 'Login' }).click();

    await page.goto('/profile');
    await page.getByLabel('Name').fill('Updated Name');
    await page.getByRole('button', { name: 'Save' }).click();

    await expect(page.getByText('Profile updated')).toBeVisible();
});
```

### Waiting Strategies

```typescript
// ❌ Bad: Fixed timeouts (flaky!)
await page.waitForTimeout(3000);

// ✅ Good: Wait for specific conditions
await page.waitForLoadState('networkidle');
await page.waitForURL('/dashboard');
await page.waitForSelector('[data-testid="user-profile"]');

// ✅ Better: Auto-waiting with assertions
await expect(page.getByText('Welcome')).toBeVisible();
await expect(page.getByRole('button', { name: 'Submit' }))
    .toBeEnabled();

// Wait for API response
const responsePromise = page.waitForResponse(
    response => response.url().includes('/api/users') &&
                response.status() === 200
);
await page.getByRole('button', { name: 'Load Users' }).click();
const response = await responsePromise;
const data = await response.json();
expect(data.users).toHaveLength(10);
```

### Network Mocking

```typescript
// Mock API failures
test('displays error when API fails', async ({ page }) => {
    await page.route('**/api/users', route => {
        route.fulfill({
            status: 500,
            contentType: 'application/json',
            body: JSON.stringify({ error: 'Internal Server Error' }),
        });
    });

    await page.goto('/users');
    await expect(page.getByText('Failed to load users')).toBeVisible();
});

// Mock third-party services
test('payment flow with mocked Stripe', async ({ page }) => {
    await page.route('**/api/stripe/**', route => {
        route.fulfill({
            status: 200,
            body: JSON.stringify({
                id: 'mock_payment_id',
                status: 'succeeded',
            }),
        });
    });

    // Test payment flow with mocked response
});
```

## Cypress Patterns

### Custom Commands

```typescript
// cypress/support/commands.ts
declare global {
    namespace Cypress {
        interface Chainable {
            login(email: string, password: string): Chainable<void>;
            dataCy(value: string): Chainable<JQuery<HTMLElement>>;
        }
    }
}

Cypress.Commands.add('login', (email: string, password: string) => {
    cy.visit('/login');
    cy.get('[data-testid="email"]').type(email);
    cy.get('[data-testid="password"]').type(password);
    cy.get('[data-testid="login-button"]').click();
    cy.url().should('include', '/dashboard');
});

Cypress.Commands.add('dataCy', (value: string) => {
    return cy.get(`[data-cy="${value}"]`);
});

// Usage
cy.login('user@example.com', 'password');
cy.dataCy('submit-button').click();
```

### Intercept Pattern

```typescript
// Mock API calls
cy.intercept('GET', '/api/users', {
    statusCode: 200,
    body: [
        { id: 1, name: 'John' },
        { id: 2, name: 'Jane' },
    ],
}).as('getUsers');

cy.visit('/users');
cy.wait('@getUsers');
cy.get('[data-testid="user-list"]').children().should('have.length', 2);

// Simulate slow network
cy.intercept('GET', '/api/data', (req) => {
    req.reply((res) => {
        res.delay(3000);  // 3 second delay
        res.send();
    });
});
```

## Advanced Patterns

### Visual Regression Testing

```typescript
import { test, expect } from '@playwright/test';

test('homepage visual snapshot', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveScreenshot('homepage.png', {
        fullPage: true,
        maxDiffPixels: 100,
    });
});

test('button states', async ({ page }) => {
    await page.goto('/components');
    const button = page.getByRole('button', { name: 'Submit' });

    // Default state
    await expect(button).toHaveScreenshot('button-default.png');

    // Hover state
    await button.hover();
    await expect(button).toHaveScreenshot('button-hover.png');

    // Disabled state
    await button.evaluate(el => el.setAttribute('disabled', 'true'));
    await expect(button).toHaveScreenshot('button-disabled.png');
});
```

### Accessibility Testing

```typescript
// Install: npm install @axe-core/playwright
import AxeBuilder from '@axe-core/playwright';

test('page should not have accessibility violations', async ({ page }) => {
    await page.goto('/');

    const accessibilityScanResults = await new AxeBuilder({ page })
        .exclude('#third-party-widget')
        .analyze();

    expect(accessibilityScanResults.violations).toEqual([]);
});

test('form is accessible', async ({ page }) => {
    await page.goto('/signup');

    const results = await new AxeBuilder({ page })
        .include('form')
        .analyze();

    expect(results.violations).toEqual([]);
});
```

### Parallel Testing with Sharding

```typescript
// playwright.config.ts
export default defineConfig({
    projects: [
        {
            name: 'shard-1/4',
            use: { ...devices['Desktop Chrome'] },
            shard: { current: 1, total: 4 },
        },
        {
            name: 'shard-2/4',
            use: { ...devices['Desktop Chrome'] },
            shard: { current: 2, total: 4 },
        },
        // ... more shards
    ],
});

// Run in CI: npx playwright test --shard=1/4
```

## Best Practices

### Selector Strategy

```typescript
// ❌ Bad: Brittle selectors
cy.get('.btn.btn-primary.submit-button').click();
cy.get('div > form > div:nth-child(2) > input').type('text');

// ✅ Good: Semantic selectors
page.getByRole('button', { name: 'Submit' }).click();
page.getByLabel('Email address').type('user@example.com');
page.get('[data-testid="email-input"]').type('user@example.com');
```

### Test Independence

```typescript
// Each test should:
// 1. Set up its own data
// 2. Run independently of other tests
// 3. Clean up after itself

test.beforeEach(async ({ page }) => {
    // Fresh state for each test
    await setupTestData();
});

test.afterEach(async ({ page }) => {
    // Clean up test data
    await cleanupTestData();
});
```

### Debugging Failing Tests

```bash
# Playwright debugging
npx playwright test --headed           # Visual mode
npx playwright test --debug            # Debug mode
npx playwright test --trace on         # Trace recording

# Use test.step for better reporting
test('checkout flow', async ({ page }) => {
    await test.step('Add item to cart', async () => {
        await page.goto('/products');
        await page.getByRole('button', { name: 'Add to Cart' }).click();
    });

    await test.step('Proceed to checkout', async () => {
        await page.goto('/cart');
        await page.getByRole('button', { name: 'Checkout' }).click();
    });
});
```

## Common Pitfalls

❌ **Flaky Tests**: Use proper waits, not fixed timeouts
❌ **Slow Tests**: Mock external APIs, use parallel execution
❌ **Over-Testing**: Don't test every edge case with E2E
❌ **Coupled Tests**: Tests should not depend on each other
❌ **Poor Selectors**: Avoid CSS classes and nth-child
❌ **No Cleanup**: Clean up test data after each test
❌ **Testing Implementation**: Test user behavior, not internals

## Quality Standards

- **Test Speed**: E2E suite should complete in <10 minutes
- **Flakiness**: <1% flaky test rate (ideally 0%)
- **Coverage**: Focus on critical user journeys (20-30 tests)
- **Browser Support**: Test on primary browsers (Chrome, Firefox, Safari)
- **CI Integration**: Run on every PR with failure screenshots

---

**Skill Type**: Testing - E2E
**Complexity**: Moderate to High
**Typical Usage**: Activated when implementing E2E test suites or debugging flaky tests
**Performance**: Optimized with parallel execution, network mocking, and smart waits
