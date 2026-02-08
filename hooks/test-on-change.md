---
name: test-on-change
description: Automatically runs related tests after code changes to catch regressions early
event: PostToolUse
tools: ["Write", "Edit"]
---

# Test-on-Change Hook

Runs relevant tests after code modifications to provide immediate feedback on regressions.

## Test Discovery Strategy

1. **Direct test file**: If `src/auth/login.ts` is modified, look for `src/auth/login.test.ts` or `tests/auth/login.test.ts`
2. **Directory tests**: Run all tests in the same directory as the modified file
3. **Related tests**: Search for test files that import the modified module

## Test Runner Selection

| Project Type | Detection | Command |
|-------------|-----------|---------|
| Node.js (Jest) | `package.json` has jest | `npx jest --findRelatedTests {file}` |
| Node.js (Vitest) | `vitest.config.*` exists | `npx vitest run --reporter=verbose {test_file}` |
| Python (pytest) | `pytest.ini` or `pyproject.toml` | `pytest {test_file} -x --tb=short` |
| Go | `go.mod` exists | `go test ./{dir}/... -run {test_name}` |
| Rust | `Cargo.toml` exists | `cargo test --lib {module}` |
| Ruby (RSpec) | `Gemfile` has rspec | `bundle exec rspec {test_file}` |
| PHP (Pest/PHPUnit) | `phpunit.xml` exists | `./vendor/bin/pest --filter={test_name}` |
| Java (JUnit) | `pom.xml` or `build.gradle` | `mvn test -pl {module} -Dtest={TestClass}` |

## Behavior

1. After Write/Edit completes on a source file, identify related tests
2. If test file found, run it with the appropriate runner
3. Report results concisely (pass/fail count, failure details)
4. If no related tests found, note this as a coverage gap

## Configuration

```yaml
# Disable for specific file patterns
exclude_paths:
  - "**/*.md"
  - "**/*.json"
  - "**/*.yaml"
  - "**/migrations/**"
  - "**/generated/**"

# Maximum test runtime before timeout
timeout_seconds: 60
```
