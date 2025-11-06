# Changelog

All notable changes to agent-forge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-05

### Added
- **Quality Framework**: Automated agent quality scoring system (`scripts/quality-scorer.py`)
  - Rubric-based evaluation (frontmatter, description, tools, structure, examples, specificity)
  - Minimum score requirements: 70/100 for new agents, 85/100 for production
  - Individual agent scoring and bulk directory analysis
- **Enhanced Documentation Standards**:
  - `CONTRIBUTING.md` now includes "Agent Development Standards" section
  - Comprehensive frontmatter requirements and validation workflow
  - Tool optimization guidelines with performance considerations
- **Agent Expansion**: Grew from 24 to 34 specialized agents across 8 tiers
  - Added Tier 5 (Utilities) for search and developer tools
  - Expanded domain coverage in Foundation, Specialist, and Expert tiers
- **CI/CD Badges**: Repository now displays agent count, command count, and quality validation status
- **Developer Quick Start**: Enhanced README with separate user/developer workflows
- **Agent Discovery Section**: Clear invocation patterns (direct, implicit, orchestration)

### Changed
- **README.md**: Updated agent statistics (24 → 34), added quality framework component
- **Agent Tier Structure**: Refined tier descriptions for clarity
  - Tier 0: Multi-agent orchestration (was: 1 agent description)
  - Added Tier 5 (Utilities) to hierarchy
  - Improved tier purpose descriptions across all levels
- **Installation Workflow**: Clarified `--user` vs `--project` scope in Quick Start
- **Validation Process**: Integrated quality scoring into standard development workflow

### Fixed
- Agent count accuracy across documentation files
- Tier hierarchy completeness (added missing Tier 5)

### Documentation
- **CHANGELOG.md** (this file): Created to track version history
- **CONTRIBUTING.md**: Added 80+ lines of agent development standards
- **README.md**: Enhanced with badges, updated stats, developer quick start

### Quality Improvements
- Established minimum quality thresholds for agent acceptance
- Automated validation prevents regression in agent quality
- Tool optimization guidelines reduce context overhead and improve performance

---

## [1.0.0] - 2024-12-XX

### Added
- Initial release of agent-forge
- 24 specialized agents across 7 tiers (Tier 0-4, 6-7)
- 35 slash commands for workflow automation
- MCP integration (memory, sequential-thinking)
- Installation scripts (`install-agents.sh`, `verify-agents.sh`)
- Core documentation (README, CONTRIBUTING, SYSTEM_OVERVIEW)
- Agent checklist (`agents/AGENT_CHECKLIST.md`)

### Features
- Multi-agent orchestration via `agent-coordinator`
- DDD workflow patterns with quality gates
- Foundation tier (API, domain modeling, testing, review, debugging, performance, system design)
- Development tier (frontend, mobile, Python, TypeScript specialists)
- Specialist tier (cloud, backend, database, DevOps, observability, SRE, data, full-stack)
- Expert tier (ML/MLOps)
- Integration tier (research, technical documentation)
- Quality tier (security architecture)

---

## Version History

- **2.0.0** (2025-11-05): Quality framework, documentation standards, 34 agents
- **1.0.0** (2024-12-XX): Initial release with 24 agents and core infrastructure

---

## Upgrade Guide

### From 1.x to 2.0

**For Users**:
1. Pull latest changes: `git pull origin main`
2. Reinstall agents: `./scripts/install-agents.sh --user`
3. Restart Claude Code

**For Contributors**:
1. Review new agent development standards in `CONTRIBUTING.md`
2. Run quality scorer on your agents: `python3 scripts/quality-scorer.py --agent path/to/agent.md`
3. Ensure agents meet minimum score (70/100 new, 85/100 production)
4. Update agent frontmatter if validation fails
5. Follow enhanced validation workflow before submitting PRs

**Breaking Changes**:
- None. Version 2.0 is backward compatible with 1.x agents.
- New quality standards apply to new/modified agents only.

---

## Contributing

See `CONTRIBUTING.md` for detailed contribution guidelines, including:
- Agent development standards
- Validation workflow (structural, quality, functional, documentation)
- Tool optimization strategies
- Quality rubric and scoring criteria
