# Changelog

All notable changes to Claude Agents Pro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-02-08 Gap Analysis & Expansion

### Added
- **Language Specialist Agents (Phase 1)**: 8 new agents in `02-development/`:
  `java-expert`, `csharp-expert`, `kotlin-expert`, `angular-expert`, `django-expert`, `rails-expert`, `spring-boot-expert`, `laravel-expert`
- **Cloud Parity Agents (Phase 2)**: 2 new agents in `03-specialists/`:
  `azure-cloud-architect`, `gcp-cloud-architect`
- **Hooks System (Phase 3)**: 10 event-driven automation hooks in `hooks/`:
  `secrets-scanner`, `auto-format`, `commit-message-validator`, `branch-protection`, `test-on-change`, `session-context`, `dependency-audit`, `file-protection`, `lint-on-save`, `pr-template`
- **MCP Server Registry (Phase 3)**: Curated catalog of 20+ MCP servers in `mcp-servers/README.md` with preset configs (minimal, development, full-stack)
- **Agentic Skills Domain (Phase 3-5)**: 6 new skills in `skills/agentic/`:
  `subagent-catalog`, `multi-agent-patterns`, `hook-development`, `mcp-server-development`, `memory-taxonomy`, `plugin-architecture`
- **TypeScript Skills (Phase 4)**: 4 new skills in `skills/typescript/`:
  `react-patterns`, `nextjs-patterns`, `type-system-advanced`, `state-management`
- **Testing Skills (Phase 4)**: 2 new skills: `contract-testing`, `load-testing`
- **Architecture Skill (Phase 4)**: `event-sourcing-cqrs`
- **Backend Skill (Phase 4)**: `websocket-patterns`
- **DevOps Skill (Phase 4)**: `github-actions-patterns`
- **Workflow Skill (Phase 5)**: `spec-driven-development`

### Changed
- **Agent Count**: 51 → 61 agents across 9 tiers
- **Skill Count**: 89 → 120 skills across 17 domains (was 16)
- **SYSTEM_OVERVIEW.md**: Added Hooks System section, Skills System section, MCP Server Registry reference
- **Documentation**: Updated all counts and references

### Summary
Gap analysis against 12 reference repositories (VoltAgent, davepoon, microck, zhsama, et al) identified 5 phases of improvements. All 5 phases completed:
1. Language specialists (Java, C#, Kotlin, Angular, Django, Rails, Spring Boot, Laravel)
2. Cloud parity (Azure, GCP alongside existing AWS)
3. Structural upgrades (hooks, MCP registry, agentic skills)
4. Skills expansion (TypeScript, testing, architecture, backend, DevOps)
5. Workflow patterns (spec-driven dev, memory taxonomy, plugin architecture)

---

## [2.2.0] - 2026-02-07 Tiger Team Review

### Changed
- **Model Updates**: Updated all 51 agent model IDs to `claude-opus-4-6`
- **Shell Compatibility**: Fixed macOS Bash 3.2 compatibility in `verify-agents.sh`, `verify-catalog.sh`, `verify-skills.sh`
- **Quality Scorer**: Fixed `quality-scorer.py` tier threshold math (was comparing 0-1 scores against 6.5-9.0); added `--min-score` argument
- **CLI Score**: Fixed `score.py` weight display and added missing Security metric
- **Schema Validation**: Consolidated schema validation (`agent-schema.json`, `agent-frontmatter.schema.json`); added "platform" and "security" to valid agent categories
- **GitHub Actions**: Pinned 17 GitHub Actions to commit SHAs for supply chain security
- **Documentation**: Updated `agents/README.md` catalog, `SYSTEM_OVERVIEW.md` tiers, CLI docs, `skills/README.md` (all 89 skills across 16 domains)

### Fixed
- **Ghost References**: Renamed `agent-coordinator` to `orchestration-coordinator` across 13+ files
- **Stale References**: Removed references to non-existent files (`AGENTS.md`, `IMPLEMENTATION_ROADMAP.md`, `verify-commands.sh`)

### Added
- **Tier READMEs**: Created README files for `00-meta`, `06-integration`, `07-quality`
- **Requirements**: Created `requirements.txt`

### Removed
- **Junk Script**: Deleted `fix_model_rationale.py`
- **Duplicate Changelog**: Merged `UPDATES.md` into this file and deleted it

---

## [2.1.0] - 2025-12-11

### Added
- **New Agents**:
  - `llm-architect` (04-experts): LLM system design and prompt architecture
  - `chaos-engineer` (03-specialists): Chaos engineering and resilience testing
  - `prompt-engineer` (04-experts): Prompt engineering and LLM optimization
  - `git-workflow-manager` (05-platform): Git workflow automation
  - `dx-optimizer` (05-platform): Developer experience optimization
- **New Tier**: 05-platform for developer platform tooling
- **New Workflow Skills**: `standup-report`, `bug-fix`, `create-feature`
- **Skill Verification**: Added `scripts/verify-skills.sh`
- **Documentation Enhancements**:
  - Quick Selection Guide in `agents/README.md`
  - Common Agent Combinations section
  - Troubleshooting section for common issues
  - Tier-level README files for `01-foundation`, `02-development`, `03-specialists`
- **Security Scoring**: Added security pattern detection in `verify-agents.sh` and security scoring dimension (15% weight) in `quality-scorer.py`

### Changed
- Expanded `CONTRIBUTING.md` with table of contents and testing section
- Improved documentation structure across all tiers

### Fixed
- Documentation gaps identified by expert review panel
- Added `trigger_keywords` to 18 skills missing them across security, devops, debugging, backend, testing, architecture, ml, finance, and data domains

---

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
- **Agent Expansion**: Grew from 24 to 45 specialized agents across 8 tiers
  - Added Tier 5 (Utilities) for search and developer tools
  - Expanded domain coverage in Foundation, Specialist, and Expert tiers
  - Added Tier 8 (Finance) with 11 trading and quantitative analysis agents
- **CI/CD Badges**: Repository now displays agent count, command count, and quality validation status
- **Developer Quick Start**: Enhanced README with separate user/developer workflows
- **Agent Discovery Section**: Clear invocation patterns (direct, implicit, orchestration)

### Changed
- **README.md**: Updated agent statistics, added quality framework component
- **Agent Tier Structure**: Refined tier descriptions for clarity
  - Tier 0: Multi-agent orchestration
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

## [1.0.0] - 2024-12-11

### Added
- Initial release of Claude Agents Pro
- 24 specialized agents across 7 tiers (Tier 0-4, 6-7)
- 35 slash commands for workflow automation
- MCP integration (memory, sequential-thinking)
- Installation scripts (`install-agents.sh`, `verify-agents.sh`)
- Core documentation (README, CONTRIBUTING, SYSTEM_OVERVIEW)
- Agent checklist (`agents/AGENT_CHECKLIST.md`)
- Schema validation (`schema-validator.py`)
- Pre-commit hooks for quality gates

### Tier Structure at Launch
- **00-meta**: Multi-agent orchestration (3 agents)
- **01-foundation**: Core engineering (9 agents)
- **02-development**: Language/platform specialists (4 agents)
- **03-specialists**: Domain experts (4 agents)
- **04-experts**: Advanced specialists (1 agent)
- **06-integration**: Research and documentation (2 agents)
- **07-quality**: Security and quality (1 agent)

### Features
- Multi-agent orchestration via `orchestration-coordinator`
- DDD workflow patterns with quality gates
- Foundation tier (API, domain modeling, testing, review, debugging, performance, system design)
- Development tier (frontend, mobile, Python, TypeScript specialists)
- Specialist tier (cloud, backend, database, DevOps, observability, SRE, data, full-stack)
- Expert tier (ML/MLOps)
- Integration tier (research, technical documentation)
- Quality tier (security architecture)

---

## Version History

| Version | Date | Agents | Highlights |
|---------|------|--------|------------|
| Unreleased | 2026-02-08 | 61 | Gap analysis & expansion: 10 new agents, 31 new skills, hooks, MCP registry |
| 2.1.0 | 2025-12-11 | 51 | New agents (llm-architect, chaos-engineer, prompt-engineer), 05-platform tier, workflow skills |
| 2.0.0 | 2025-11-05 | 45 | Quality framework, documentation standards, finance tier, agent expansion |
| 1.0.0 | 2024-12-11 | 24 | Initial release with core infrastructure |

---

## Upgrade Guide

### From 2.x to Unreleased

**For Users**:
1. Pull latest changes: `git pull origin main`
2. Reinstall agents: `./scripts/install-agents.sh --user`
3. Restart Claude Code (agents now use `claude-opus-4-6`)

**For Contributors**:
1. Run quality scorer: `python3 scripts/quality-scorer.py --agents-dir agents`
2. Shell scripts are now macOS Bash 3.2 compatible
3. Use consolidated schema files for validation

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
- None. All versions are backward compatible with previous agents.
- New quality standards apply to new/modified agents only.

---

## Contributing

See `CONTRIBUTING.md` for detailed contribution guidelines, including:
- Agent development standards
- Validation workflow (structural, quality, functional, documentation)
- Quality rubric and scoring criteria
