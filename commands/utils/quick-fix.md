---
description: Quick fixes for common development issues
args: <issue-type> [--auto-fix]
tools: Task, Read, Grep
model: sonnet
---

# Quick Fix Assistant

Apply quick fix for: **$ARGUMENTS**

## Common Issues & Solutions

### Linting Issues
```bash
# ESLint/Prettier (JavaScript/TypeScript)
npm run lint -- --fix
npx prettier --write "**/*.{js,jsx,ts,tsx,json,css,md}"

# Python (Black/Ruff)
black . --line-length 88
ruff check . --fix

# Go
gofmt -w .
golangci-lint run --fix
```

### Type Errors
1. Check for missing type definitions
2. Install type packages: `npm i -D @types/...`
3. Update tsconfig.json if needed
4. Generate types from schemas

### Import/Module Issues
- Fix import paths (relative vs absolute)
- Update module resolution in config
- Check for circular dependencies
- Verify package installations

### Git Issues
```bash
# Merge conflicts
git status --short | grep "^UU" | awk '{print $2}' | xargs -I {} sh -c 'echo "Resolving: {}" && git checkout --theirs {} || git checkout --ours {}'

# Large file issues
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch path/to/large/file' --prune-empty --tag-name-filter cat -- --all

# Reset to clean state
git reset --hard HEAD
git clean -fd
```

### Package/Dependency Issues
```bash
# Node.js
rm -rf node_modules package-lock.json
npm install

# Python
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# Clear caches
npm cache clean --force
pip cache purge
```

### Build Failures
1. Clear build artifacts
2. Reinstall dependencies
3. Check environment variables
4. Verify build configuration

### Test Failures
- Clear test cache
- Reset test database
- Update snapshots
- Fix timing issues with proper waits

### Performance Quick Wins
1. **React**: Add React.memo, useMemo, useCallback
2. **Database**: Add missing indexes
3. **API**: Implement caching headers
4. **Bundle**: Enable code splitting

### Security Vulnerabilities
```bash
# npm
npm audit fix
npm audit fix --force  # Use with caution

# Python
pip-audit --fix
safety check --json

# Secrets in git
git secrets --scan
trufflehog git file://. --only-verified
```

## Auto-Fix Mode

When `--auto-fix` is provided:
1. Detect issue type automatically
2. Delegate fixes to appropriate specialist agents via Task tool
3. Report changes made with validation
4. Suggest manual fixes for complex issues

## Issue Detection

Automatically detect and fix:
- Linting violations
- Import errors
- Type mismatches
- Formatting issues
- Simple merge conflicts
- Dependency vulnerabilities

## Prevention Tips

After fixing, suggest:
1. Pre-commit hooks setup
2. CI/CD validation rules
3. IDE configuration
4. Team coding standards

## Examples

```
/quick-fix "eslint errors"
/quick-fix "typescript compilation" --auto-fix
/quick-fix "merge conflicts"
/quick-fix "slow bundle size"
/quick-fix "security vulnerabilities"
```

## Complex Issues

For issues requiring deeper analysis:
- Use `/debug` for detailed debugging
- Use `/agent error-diagnostician` for complex errors
- Use `/test` to validate fixes