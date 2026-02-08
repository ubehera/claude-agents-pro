---
name: defense-in-depth-validation
description: Load when user needs multi-layer validation patterns to make bugs structurally impossible by validating at every layer data passes through
trigger_keywords: [validation, defense in depth, multi-layer validation, input validation, boundary validation, structural validation, fail fast]
---

# Defense-in-Depth Validation

Multi-layer validation pattern that validates at every layer data passes through, making bugs structurally impossible rather than just handled.

## Core Concepts

- **Layer Independence**: Each validation layer must work independently. If entry validation is bypassed (mocks, internal calls), business logic validation still catches invalid data. Never assume upstream validation occurred.

- **Fail-Fast Principle**: Validate as early as possible in the call stack. Rejecting bad data at the API boundary is cheaper than catching it in database transactions. Early failures produce clearer error messages.

- **Environment-Aware Guards**: Production and test environments need different protections. Tests should refuse operations on real data; production should refuse operations on system directories. Use environment guards, not just generic validation.

- **Validation as Documentation**: Explicit validation checks document invariants. When code says `if (!path) throw`, it documents that null paths are never acceptable. Future developers understand constraints without reading docs.

- **Anti-pattern - Single Point of Validation**: Relying on one validation layer creates fragile systems. Code paths change, refactoring moves logic, mocks bypass checks. Multiple independent layers ensure at least one catches bad data.

## Core Principle

**Single validation**: "We fixed the bug"
**Multiple layers**: "We made the bug impossible"

When you fix a bug caused by invalid data, adding validation at one place feels sufficient. But that single check can be bypassed by different code paths, refactoring, or mocks. Defense-in-depth validation ensures that no matter how data enters the system, it's validated at every checkpoint.

## Why Multiple Layers

Different layers catch different cases:
- **Entry validation**: Catches most bugs at API boundary
- **Business logic**: Catches edge cases in operations
- **Environment guards**: Prevents context-specific dangers (test vs production)
- **Debug logging**: Helps when other layers fail

## The Four Layers

### Layer 1: Entry Point Validation

Reject obviously invalid input at API boundary.

```typescript
function createProject(name: string, workingDirectory: string) {
    // Validate not empty
    if (!workingDirectory || workingDirectory.trim() === '') {
        throw new Error('workingDirectory cannot be empty');
    }

    // Validate exists
    if (!existsSync(workingDirectory)) {
        throw new Error(`workingDirectory does not exist: ${workingDirectory}`);
    }

    // Validate is directory
    if (!statSync(workingDirectory).isDirectory()) {
        throw new Error(`workingDirectory is not a directory: ${workingDirectory}`);
    }

    // Validate writable
    try {
        accessSync(workingDirectory, constants.W_OK);
    } catch (e) {
        throw new Error(`workingDirectory is not writable: ${workingDirectory}`);
    }

    // ... proceed with project creation
}
```

**Python Example:**
```python
def create_project(name: str, working_directory: str) -> Project:
    """Create new project with validated directory."""
    # Entry validation
    if not working_directory:
        raise ValueError("working_directory cannot be empty")

    if not os.path.exists(working_directory):
        raise ValueError(f"working_directory does not exist: {working_directory}")

    if not os.path.isdir(working_directory):
        raise ValueError(f"working_directory is not a directory: {working_directory}")

    if not os.access(working_directory, os.W_OK):
        raise ValueError(f"working_directory is not writable: {working_directory}")

    return Project(name, working_directory)
```

### Layer 2: Business Logic Validation

Ensure data makes sense for this specific operation.

```typescript
function initializeWorkspace(projectDir: string, sessionId: string) {
    // Business logic validation
    if (!projectDir) {
        throw new Error('projectDir required for workspace initialization');
    }

    if (!sessionId) {
        throw new Error('sessionId required for workspace initialization');
    }

    // Validate projectDir looks like absolute path
    if (!path.isAbsolute(projectDir)) {
        throw new Error(`projectDir must be absolute path, got: ${projectDir}`);
    }

    // ... proceed with workspace initialization
}
```

**Python Example:**
```python
def initialize_workspace(project_dir: str, session_id: str) -> Workspace:
    """Initialize workspace with validated parameters."""
    # Business logic validation
    if not project_dir:
        raise ValueError("project_dir required for workspace initialization")

    if not session_id:
        raise ValueError("session_id required for workspace initialization")

    if not os.path.isabs(project_dir):
        raise ValueError(f"project_dir must be absolute path, got: {project_dir}")

    return Workspace(project_dir, session_id)
```

### Layer 3: Environment Guards

Prevent dangerous operations in specific contexts (test vs production).

```typescript
async function gitInit(directory: string) {
    // Environment guard for tests
    if (process.env.NODE_ENV === 'test') {
        const normalized = normalize(resolve(directory));
        const tmpDir = normalize(resolve(tmpdir()));

        if (!normalized.startsWith(tmpDir)) {
            throw new Error(
                `Refusing git init outside temp dir during tests: ${directory}\n` +
                `Expected path starting with: ${tmpDir}\n` +
                `Got: ${normalized}`
            );
        }
    }

    // Production guard
    if (process.env.NODE_ENV === 'production') {
        // Refuse git operations on system directories
        const systemDirs = ['/usr', '/bin', '/etc', '/var', '/sys'];
        const normalized = normalize(resolve(directory));

        for (const systemDir of systemDirs) {
            if (normalized.startsWith(systemDir)) {
                throw new Error(
                    `Refusing git init in system directory: ${directory}`
                );
            }
        }
    }

    // ... proceed with git init
}
```

**Python Example:**
```python
def git_init(directory: str) -> None:
    """Initialize git repository with environment guards."""
    import os
    import tempfile

    # Environment guard for tests
    if os.getenv("ENV") == "test":
        normalized = os.path.normpath(os.path.abspath(directory))
        tmp_dir = os.path.normpath(tempfile.gettempdir())

        if not normalized.startswith(tmp_dir):
            raise RuntimeError(
                f"Refusing git init outside temp dir during tests: {directory}\n"
                f"Expected path starting with: {tmp_dir}\n"
                f"Got: {normalized}"
            )

    # Production guard
    if os.getenv("ENV") == "production":
        system_dirs = ["/usr", "/bin", "/etc", "/var", "/sys"]
        normalized = os.path.normpath(os.path.abspath(directory))

        for system_dir in system_dirs:
            if normalized.startswith(system_dir):
                raise RuntimeError(
                    f"Refusing git init in system directory: {directory}"
                )

    # Proceed with git init
    subprocess.run(["git", "init", directory], check=True)
```

### Layer 4: Debug Instrumentation

Capture context for forensics when issues occur.

```typescript
async function gitInit(directory: string) {
    // Debug instrumentation
    const stack = new Error().stack;
    logger.debug('About to git init', {
        directory,
        cwd: process.cwd(),
        env: process.env.NODE_ENV,
        caller: stack
    });

    // ... proceed with git init

    logger.debug('Git init completed', {
        directory,
        duration: Date.now() - startTime
    });
}
```

**Python Example:**
```python
import logging
import traceback
import time

logger = logging.getLogger(__name__)

def git_init(directory: str) -> None:
    """Initialize git repository with debug instrumentation."""
    # Debug instrumentation
    start_time = time.time()
    stack = ''.join(traceback.format_stack())

    logger.debug(
        "About to git init",
        extra={
            "directory": directory,
            "cwd": os.getcwd(),
            "env": os.getenv("ENV"),
            "caller": stack
        }
    )

    # Proceed with git init
    subprocess.run(["git", "init", directory], check=True)

    # Log completion
    duration = time.time() - start_time
    logger.debug(
        "Git init completed",
        extra={"directory": directory, "duration": duration}
    )
```

## Applying the Pattern

When you find a bug caused by invalid data:

1. **Trace the data flow**: Where does bad value originate? Where is it used?
2. **Map all checkpoints**: List every point data passes through
3. **Add validation at each layer**: Entry, business, environment, debug
4. **Test each layer**: Try to bypass layer 1, verify layer 2 catches it

### Example Data Flow

**Bug**: Empty `projectDir` caused `git init` in source code directory

**Data Flow:**
1. Test setup → empty string
2. `Project.create(name, '')`
3. `WorkspaceManager.createWorkspace('')`
4. `git init` runs in `process.cwd()`

**Four Layers Added:**
- **Layer 1**: `Project.create()` validates not empty/exists/writable
- **Layer 2**: `WorkspaceManager` validates projectDir not empty
- **Layer 3**: `WorktreeManager` refuses git init outside tmpdir in tests
- **Layer 4**: Stack trace logging before git init

**Result**: Bug impossible to reproduce, all tests passed

## Implementation Patterns

### Validation Chain

```typescript
class ValidationChain<T> {
    private validators: Array<(value: T) => void> = [];

    add(validator: (value: T) => void): this {
        this.validators.push(validator);
        return this;
    }

    validate(value: T, context: string = 'value'): void {
        for (const validator of this.validators) {
            try {
                validator(value);
            } catch (error) {
                throw new Error(
                    `Validation failed for ${context}: ${error.message}`
                );
            }
        }
    }
}

// Usage
const projectDirValidation = new ValidationChain<string>()
    .add(dir => {
        if (!dir) throw new Error('Cannot be empty');
    })
    .add(dir => {
        if (!existsSync(dir)) throw new Error('Must exist');
    })
    .add(dir => {
        if (!statSync(dir).isDirectory()) throw new Error('Must be directory');
    })
    .add(dir => {
        try {
            accessSync(dir, constants.W_OK);
        } catch {
            throw new Error('Must be writable');
        }
    });

projectDirValidation.validate(userInput, 'projectDir');
```

### Validation Decorator

```python
from functools import wraps
from typing import Callable, Any

def validate_args(**validators: Callable[[Any], None]):
    """Decorator to validate function arguments."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate each argument
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    try:
                        validator(value)
                    except Exception as e:
                        raise ValueError(
                            f"Validation failed for {param_name}: {e}"
                        )

            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
def validate_project_dir(directory: str) -> None:
    """Validate project directory."""
    if not directory:
        raise ValueError("Cannot be empty")
    if not os.path.exists(directory):
        raise ValueError("Must exist")
    if not os.path.isdir(directory):
        raise ValueError("Must be directory")
    if not os.access(directory, os.W_OK):
        raise ValueError("Must be writable")

@validate_args(project_dir=validate_project_dir)
def create_project(name: str, project_dir: str) -> Project:
    """Create project with validated directory."""
    return Project(name, project_dir)
```

## Key Insights

All four layers are necessary. During testing, each layer caught bugs the others missed:
- Different code paths bypassed entry validation
- Mocks bypassed business logic checks
- Edge cases on different platforms needed environment guards
- Debug logging identified structural misuse

**Don't stop at one validation point.** Add checks at every layer.

## Common Use Cases

### 1. File Path Validation

```typescript
function validateFilePath(path: string, context: string) {
    // Layer 1: Entry validation
    if (!path) throw new Error(`${context}: path cannot be empty`);

    // Layer 2: Business validation
    if (!path.isAbsolute(path)) {
        throw new Error(`${context}: path must be absolute`);
    }

    // Layer 3: Environment guard
    if (process.env.NODE_ENV === 'test' && !path.startsWith(tmpdir())) {
        throw new Error(`${context}: test paths must be in temp directory`);
    }

    // Layer 4: Debug logging
    logger.debug(`Validated file path: ${path}`, { context });
}
```

### 2. Database Query Validation

```python
def execute_query(query: str, params: dict) -> list:
    """Execute database query with multi-layer validation."""
    # Layer 1: Entry validation
    if not query:
        raise ValueError("Query cannot be empty")
    if not isinstance(params, dict):
        raise TypeError("Params must be dictionary")

    # Layer 2: Business validation
    if "DROP" in query.upper() and os.getenv("ENV") == "production":
        raise ValueError("DROP queries forbidden in production")

    # Layer 3: Environment guard
    if os.getenv("ENV") == "test" and "production_db" in query:
        raise RuntimeError("Cannot query production DB in tests")

    # Layer 4: Debug logging
    logger.debug(f"Executing query: {query}", extra={"params": params})

    return db.execute(query, params)
```

### 3. API Input Validation

```typescript
function handleCreateUser(req: Request, res: Response) {
    // Layer 1: Entry validation
    const { email, password, name } = req.body;
    if (!email || !password || !name) {
        throw new ValidationError('Missing required fields');
    }

    // Layer 2: Business validation
    if (!isValidEmail(email)) {
        throw new ValidationError('Invalid email format');
    }
    if (password.length < 8) {
        throw new ValidationError('Password too short');
    }

    // Layer 3: Environment guard
    if (process.env.NODE_ENV === 'production' && email.includes('test')) {
        throw new ValidationError('Test emails not allowed in production');
    }

    // Layer 4: Debug logging
    logger.debug('Creating user', { email, name });

    const user = userService.create({ email, password, name });
    res.json(user);
}
```

## Best Practices

1. **Validate at every boundary**: Entry point, business logic, environment-specific
2. **Fail fast**: Validate early in the call stack
3. **Be specific**: Clear error messages with context
4. **Log for debugging**: Include stack traces and context
5. **Test bypass attempts**: Try to break layer 1, verify layer 2 catches it
6. **Document invariants**: Explain what each layer protects against

## Quality Standards

- **Comprehensive**: Validation at all layers (entry, business, environment, debug)
- **Specific Errors**: Clear messages explaining what failed and why
- **Testable**: Each layer can be tested independently
- **Maintainable**: Validation logic is reusable and composable
- **Observable**: Debug logging provides forensic context

---

**Skill Type**: Backend - Validation Patterns
**Complexity**: Moderate
**Typical Usage**: Activated when implementing robust validation to prevent bugs
**Languages**: TypeScript, Python, universal pattern
