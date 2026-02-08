---
description: Automated release preparation with quality gates and validation
args: [--version major|minor|patch|x.y.z] [--pre-release alpha|beta|rc] [--dry-run]
tools: Bash(git:*), Bash(npm version:*), Read, Write, Grep
model: sonnet
---

## Objective
Automate release preparation process with comprehensive validation, quality gates, and automated changelog generation.

## Before You Run
- Ensure main branch is up to date: `git pull origin main`
- Verify all tests pass: `/verify-agents && /score-agents`
- Review pending changes and issues
- Confirm release branch strategy

## Execution
Prepare release with automated validation:

```bash
# Semantic version bump
!/release-prepare --version minor

# Pre-release preparation
!/release-prepare --version 2.1.0 --pre-release beta

# Dry-run to preview changes
!/release-prepare --version patch --dry-run
```

## Release Preparation Workflow

### 1. Version Management
- **Semantic Versioning**: Automatic version calculation
- **Tag Management**: Create annotated release tags
- **Branch Strategy**: Release branch creation and management
- **Version Files**: Update version references across codebase

### 2. Quality Validation
- **Agent Verification**: Full agent validation suite
- **Quality Scoring**: Minimum quality thresholds
- **Security Scanning**: Vulnerability assessment
- **Compliance Checking**: Policy and standard adherence

### 3. Documentation Generation
- **Changelog**: Automated generation from commit history
- **Release Notes**: Feature highlights and breaking changes
- **API Documentation**: Updated agent specifications
- **Migration Guides**: Version upgrade instructions

### 4. Asset Preparation
- **Agent Packaging**: Compress and validate agent bundles
- **Checksum Generation**: Integrity verification files
- **Signature Creation**: Code signing for security
- **Distribution Preparation**: Upload artifacts to staging

## Quality Gates

### Gate 1: Code Quality
- [ ] All agents pass validation
- [ ] Quality scores above threshold (85%)
- [ ] No critical security issues
- [ ] Documentation completeness check

### Gate 2: Integration Testing
- [ ] Installation scripts work correctly
- [ ] Cross-platform compatibility verified
- [ ] Performance benchmarks met
- [ ] Backward compatibility maintained

### Gate 3: Release Readiness
- [ ] Changelog generated and reviewed
- [ ] Release notes completed
- [ ] Assets packaged and verified
- [ ] Distribution channels prepared

## Automated Checks

### Version Consistency
```bash
# Verify version references are updated
grep -r "version:" agents/ configs/ scripts/
```

### Breaking Changes Detection
```bash
# Analyze API changes and compatibility
git diff --name-only HEAD~10..HEAD | grep -E "\.(md|json|yaml)$"
```

### Release Branch Health
```bash
# Validate release branch state
git log --oneline --since="2 weeks ago" --no-merges
```

## Rollback Procedures

### Pre-Release Rollback
- Delete release branch
- Remove version tags
- Reset version files
- Clean up artifacts

### Post-Release Rollback
- Create hotfix branch
- Revert problematic changes
- Emergency patch release
- Communication plan activation

## Output Artifacts

### Release Package
- `agents-v{version}.tar.gz`: Complete agent bundle
- `CHANGELOG.md`: Version-specific changes
- `RELEASE_NOTES.md`: Human-readable highlights
- `checksums.txt`: File integrity verification

### Metadata
- Release tag: `v{version}`
- Release branch: `release/v{version}`
- Build metadata: `build.json`
- Quality report: `quality-v{version}.json`

## Follow Up
- Review generated changelog for accuracy
- Test installation on clean environment
- Coordinate with stakeholders for release timing
- Prepare release announcement
- Monitor post-release metrics and feedback

## Integration Points
- **CI/CD Pipeline**: Trigger automated builds
- **Issue Tracking**: Link resolved issues
- **Documentation**: Update external docs
- **Distribution**: Publish to repositories
