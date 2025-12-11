#!/usr/bin/env python3
"""
Add contextual example sections to agent markdown files.

This script:
1. Reads all .md files in agents/ subdirectories
2. Parses YAML frontmatter
3. Generates 2-3 contextual examples for agents without examples
4. Updates frontmatter with generated examples
5. Writes changes back to file

Usage:
    python3 scripts/add-examples.py                    # Apply changes
    python3 scripts/add-examples.py --dry-run          # Preview changes
    python3 scripts/add-examples.py --verbose          # Show detailed output
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install PyYAML")
    sys.exit(1)


class FrontmatterParser:
    """Parse and manipulate YAML frontmatter in markdown files."""

    FRONTMATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n(.*)$', re.DOTALL)

    @staticmethod
    def parse(content: str) -> Tuple[Optional[Dict], str, str]:
        """
        Parse markdown content into frontmatter, body, and original frontmatter text.

        Returns:
            Tuple of (frontmatter_dict, body_content, original_frontmatter_text)
        """
        match = FrontmatterParser.FRONTMATTER_PATTERN.match(content)
        if not match:
            return None, content, ""

        frontmatter_text = match.group(1)
        body = match.group(2)

        try:
            frontmatter = yaml.safe_load(frontmatter_text)
            return frontmatter, body, frontmatter_text
        except yaml.YAMLError as e:
            print(f"Warning: Failed to parse YAML frontmatter: {e}")
            return None, content, ""

    @staticmethod
    def serialize(frontmatter: Dict, body: str) -> str:
        """Serialize frontmatter and body back into markdown content."""
        # Use custom YAML dump settings for clean output
        yaml_content = yaml.dump(
            frontmatter,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120
        )
        return f"---\n{yaml_content}---\n{body}"


class ExampleGenerator:
    """Generate contextual examples for agents based on their metadata."""

    # Domain-specific example templates
    DOMAIN_TEMPLATES = {
        'orchestration': [
            {
                'trigger': 'Coordinate multi-agent workflow for building a payment processing system',
                'commentary': 'Activates orchestration expertise for complex multi-domain project, expecting task decomposition and agent delegation strategy'
            },
            {
                'trigger': 'Break down large e-commerce platform migration into specialized agent tasks',
                'commentary': 'Engages task decomposition and routing capabilities, expecting dependency analysis and parallel execution planning'
            }
        ],
        'algorithmic-trading': [
            {
                'trigger': 'Implement order execution system with TWAP and VWAP algorithms',
                'commentary': 'Triggers execution algorithm expertise, expecting OMS integration with smart order routing and slippage optimization'
            },
            {
                'trigger': 'Set up multi-broker integration for live trading with position reconciliation',
                'commentary': 'Activates broker API integration skills, expecting Alpaca/E*TRADE setup with real-time position tracking'
            }
        ],
        'api': [
            {
                'trigger': 'Design a REST API for user authentication with OAuth 2.0',
                'commentary': 'Triggers API design expertise with authentication focus, expecting OpenAPI spec and security best practices'
            },
            {
                'trigger': 'Review our GraphQL schema for performance bottlenecks',
                'commentary': 'Engages GraphQL expertise for optimization analysis, expecting N+1 query detection and resolver improvements'
            },
            {
                'trigger': 'Set up API gateway with rate limiting for microservices',
                'commentary': 'Activates gateway configuration skills, expecting Kong/Apigee setup with throttling policies'
            }
        ],
        'testing': [
            {
                'trigger': 'Create comprehensive test suite for payment processing module',
                'commentary': 'Invokes test design expertise for critical business logic, expecting unit, integration, and E2E test strategies'
            },
            {
                'trigger': 'Implement property-based tests for our validation library',
                'commentary': 'Engages advanced testing patterns, expecting hypothesis/fast-check implementation with edge case coverage'
            }
        ],
        'database': [
            {
                'trigger': 'Design database schema for multi-tenant SaaS application',
                'commentary': 'Triggers data modeling expertise with tenant isolation focus, expecting schema design with RLS policies'
            },
            {
                'trigger': 'Optimize slow queries causing P95 latency spikes',
                'commentary': 'Activates performance optimization skills, expecting query analysis, indexing strategy, and execution plan review'
            }
        ],
        'security': [
            {
                'trigger': 'Conduct security audit of authentication and authorization flows',
                'commentary': 'Invokes security review expertise, expecting threat modeling, vulnerability assessment, and remediation plan'
            },
            {
                'trigger': 'Implement secure secrets management for production infrastructure',
                'commentary': 'Triggers secrets handling best practices, expecting vault setup, rotation policies, and access controls'
            }
        ],
        'cloud': [
            {
                'trigger': 'Design AWS architecture for high-availability web application',
                'commentary': 'Engages cloud architecture expertise, expecting multi-AZ deployment, auto-scaling, and disaster recovery strategy'
            },
            {
                'trigger': 'Migrate monolith to serverless architecture on AWS',
                'commentary': 'Triggers cloud migration planning, expecting Lambda design, API Gateway setup, and cost optimization'
            }
        ],
        'frontend': [
            {
                'trigger': 'Build responsive dashboard with real-time data visualization',
                'commentary': 'Activates frontend expertise, expecting component architecture, WebSocket integration, and performance optimization'
            },
            {
                'trigger': 'Optimize React app bundle size and loading performance',
                'commentary': 'Engages frontend performance skills, expecting code splitting, lazy loading, and bundle analysis'
            }
        ],
        'backend': [
            {
                'trigger': 'Design event-driven architecture for order processing system',
                'commentary': 'Triggers backend architecture expertise, expecting message queue design, saga patterns, and idempotency handling'
            },
            {
                'trigger': 'Implement distributed transaction handling across microservices',
                'commentary': 'Activates advanced backend patterns, expecting 2PC/saga implementation and consistency guarantees'
            }
        ],
        'performance': [
            {
                'trigger': 'Investigate memory leak causing OOM crashes in production',
                'commentary': 'Engages performance diagnostics, expecting heap analysis, profiling strategy, and memory optimization'
            },
            {
                'trigger': 'Optimize application response time from 2s to <200ms',
                'commentary': 'Triggers performance engineering, expecting bottleneck analysis, caching strategy, and query optimization'
            }
        ],
        'devops': [
            {
                'trigger': 'Set up CI/CD pipeline with automated testing and deployment',
                'commentary': 'Activates DevOps automation, expecting pipeline configuration, quality gates, and rollback strategies'
            },
            {
                'trigger': 'Implement infrastructure as code for multi-environment deployment',
                'commentary': 'Engages IaC expertise, expecting Terraform/CloudFormation, environment parity, and secrets management'
            }
        ],
        'observability': [
            {
                'trigger': 'Design monitoring and alerting strategy for microservices',
                'commentary': 'Triggers observability design, expecting metrics, logging, tracing setup with SLO/SLA definitions'
            },
            {
                'trigger': 'Implement distributed tracing for debugging latency issues',
                'commentary': 'Activates tracing expertise, expecting OpenTelemetry setup, span instrumentation, and correlation IDs'
            }
        ],
        'ml': [
            {
                'trigger': 'Design ML pipeline for fraud detection model',
                'commentary': 'Engages ML engineering, expecting feature engineering, model training pipeline, and monitoring setup'
            },
            {
                'trigger': 'Optimize model inference latency for real-time predictions',
                'commentary': 'Triggers ML performance optimization, expecting quantization, batching, and caching strategies'
            }
        ],
        'trading': [
            {
                'trigger': 'Design low-latency order execution system for algorithmic trading',
                'commentary': 'Activates trading systems expertise, expecting FIX protocol, market data handling, and microsecond optimization'
            },
            {
                'trigger': 'Implement risk management system with real-time position monitoring',
                'commentary': 'Engages trading risk management, expecting PnL calculation, VaR limits, and circuit breakers'
            }
        ],
        'finance': [
            {
                'trigger': 'Build backtesting framework for quantitative trading strategies',
                'commentary': 'Triggers quant engineering, expecting historical data handling, slippage modeling, and performance metrics'
            },
            {
                'trigger': 'Implement options pricing engine with Greeks calculation',
                'commentary': 'Activates derivatives expertise, expecting Black-Scholes/binomial models, volatility surfaces, and sensitivities'
            }
        ],
        'portfolio': [
            {
                'trigger': 'Design portfolio optimization system with risk constraints',
                'commentary': 'Activates portfolio management expertise, expecting mean-variance optimization, risk budgeting, and rebalancing logic'
            },
            {
                'trigger': 'Implement performance attribution analysis for multi-asset portfolio',
                'commentary': 'Engages portfolio analytics, expecting factor decomposition, attribution models, and reporting dashboards'
            }
        ],
        'quant': [
            {
                'trigger': 'Develop statistical arbitrage strategy using cointegration',
                'commentary': 'Triggers quantitative research expertise, expecting pairs trading implementation with statistical testing'
            },
            {
                'trigger': 'Build factor model for equity returns prediction',
                'commentary': 'Activates quant modeling skills, expecting factor construction, regression analysis, and backtesting'
            }
        ],
        'market-data': [
            {
                'trigger': 'Set up real-time market data pipeline with tick-level granularity',
                'commentary': 'Activates market data engineering, expecting WebSocket streaming, normalization, and storage optimization'
            },
            {
                'trigger': 'Implement historical data loader with symbol normalization',
                'commentary': 'Engages data ingestion expertise, expecting corporate action handling and data quality validation'
            }
        ],
        'code-review': [
            {
                'trigger': 'Review pull request for security vulnerabilities and code quality',
                'commentary': 'Activates code review expertise, expecting security analysis, best practices validation, and improvement suggestions'
            },
            {
                'trigger': 'Audit codebase for technical debt and maintainability issues',
                'commentary': 'Engages comprehensive review capabilities, expecting architecture analysis and refactoring recommendations'
            }
        ],
        'domain-modeling': [
            {
                'trigger': 'Design domain model for e-commerce order fulfillment system',
                'commentary': 'Triggers DDD expertise, expecting bounded contexts, aggregates, and ubiquitous language definition'
            },
            {
                'trigger': 'Model complex business rules for insurance underwriting',
                'commentary': 'Activates domain modeling skills, expecting entity relationships, invariants, and business logic encapsulation'
            }
        ],
        'system-design': [
            {
                'trigger': 'Design scalable architecture for social media platform',
                'commentary': 'Activates system design expertise, expecting architecture diagrams, capacity planning, and technology choices'
            },
            {
                'trigger': 'Architect microservices system with event-driven communication',
                'commentary': 'Engages distributed systems knowledge, expecting service boundaries, messaging patterns, and data consistency strategies'
            }
        ]
    }

    @classmethod
    def generate_examples(cls, agent_name: str, description: str, category: str) -> List[Dict[str, str]]:
        """
        Generate 2-3 contextual examples for an agent based on its metadata.

        Args:
            agent_name: Name of the agent
            description: Agent description
            category: Agent category

        Returns:
            List of example dictionaries with 'trigger' and 'commentary' keys
        """
        # Detect domain from name and description
        domain = cls._detect_domain(agent_name, description)

        # Get domain-specific templates or generate generic ones
        if domain and domain in cls.DOMAIN_TEMPLATES:
            templates = cls.DOMAIN_TEMPLATES[domain]
            return templates[:2]  # Return 2 examples

        # Fallback to generic example generation
        return cls._generate_generic_examples(agent_name, description, category)

    @staticmethod
    def _detect_domain(agent_name: str, description: str) -> Optional[str]:
        """Detect agent domain from name and description (most specific first)."""
        text = (agent_name + " " + description).lower()

        # Order matters: check more specific domains before general ones
        domain_keywords = {
            # Most specific domains first
            'orchestration': ['orchestrat', 'multi-agent', 'coordinator', 'task decomposition', 'agent routing'],
            'algorithmic-trading': ['algorithmic trading', 'order execution', 'oms', 'twap', 'vwap', 'execution algorithm'],
            'portfolio': ['portfolio', 'asset allocation', 'rebalancing', 'portfolio optimization'],
            'quant': ['quantitative analyst', 'quant', 'factor model', 'statistical arbitrage', 'alpha generation'],
            'market-data': ['market data', 'tick data', 'market feed', 'real-time data'],
            'code-review': ['code review', 'reviewer', 'pull request review', 'code quality'],
            'domain-modeling': ['domain model', 'ddd', 'bounded context', 'aggregate', 'ubiquitous language'],
            'system-design': ['system design', 'architect', 'scalability', 'distributed system'],

            # More general domains
            'trading': ['trading', 'order', 'fix protocol', 'broker', 'execution'],
            'finance': ['finance', 'options', 'derivatives', 'backtest', 'pnl'],
            'security': ['security', 'authentication', 'authorization', 'oauth', 'jwt', 'vulnerability', 'audit'],
            'testing': ['test engineer', 'testing', 'qa', 'pytest', 'jest', 'test suite'],
            'database': ['database', 'sql', 'postgres', 'mongodb', 'schema', 'query optimization'],
            'api': ['api', 'rest', 'graphql', 'endpoint', 'gateway', 'openapi', 'swagger'],
            'cloud': ['aws', 'cloud', 'lambda', 'ec2', 's3', 'serverless'],
            'frontend': ['frontend', 'react', 'vue', 'angular', 'ui', 'component'],
            'backend': ['backend', 'microservice', 'event-driven', 'saga'],
            'performance': ['performance', 'optimization', 'profiling', 'memory', 'latency'],
            'devops': ['devops', 'ci/cd', 'pipeline', 'deployment', 'infrastructure as code'],
            'observability': ['observability', 'monitoring', 'logging', 'tracing', 'metrics'],
            'ml': ['machine learning', 'ml', 'model', 'training', 'inference', 'neural']
        }

        # Check domains in order (specific to general)
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain

        return None

    @staticmethod
    def _generate_generic_examples(agent_name: str, description: str, category: str) -> List[Dict[str, str]]:
        """Generate generic examples when no domain-specific templates match."""
        # Extract key phrases from description
        name_clean = agent_name.replace('-', ' ').title()

        # Create generic but contextual examples
        examples = [
            {
                'trigger': f'I need help with {agent_name.replace("-", " ")} related tasks',
                'commentary': f'Activates {name_clean} based on explicit agent reference, expecting domain expertise application'
            },
            {
                'trigger': f'Review and improve our {category} implementation',
                'commentary': f'Engages {name_clean} for quality review and optimization, expecting best practices and improvement recommendations'
            }
        ]

        return examples


class AgentUpdater:
    """Update agent markdown files with generated examples."""

    def __init__(self, agents_dir: Path, dry_run: bool = False, verbose: bool = False):
        self.agents_dir = agents_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.stats = {
            'total': 0,
            'skipped_has_examples': 0,
            'skipped_no_frontmatter': 0,
            'skipped_readme': 0,
            'updated': 0,
            'errors': 0
        }

    def process_all_agents(self) -> None:
        """Process all agent markdown files in the agents directory."""
        # Find all .md files in subdirectories
        agent_files = list(self.agents_dir.glob('**/*.md'))

        print(f"Found {len(agent_files)} markdown files in {self.agents_dir}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE UPDATE'}\n")

        for agent_file in sorted(agent_files):
            self._process_agent_file(agent_file)

        self._print_summary()

    def _process_agent_file(self, file_path: Path) -> None:
        """Process a single agent markdown file."""
        self.stats['total'] += 1

        # Skip README and other non-agent files
        if file_path.name in ['README.md', 'AGENT_CHECKLIST.md', 'TESTING.md', 'finance-glossary.md']:
            self.stats['skipped_readme'] += 1
            if self.verbose:
                print(f"⊝ SKIP: {file_path.relative_to(self.agents_dir)} (non-agent file)")
            return

        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')

            # Parse frontmatter
            frontmatter, body, original_fm_text = FrontmatterParser.parse(content)

            if frontmatter is None:
                self.stats['skipped_no_frontmatter'] += 1
                if self.verbose:
                    print(f"⊝ SKIP: {file_path.relative_to(self.agents_dir)} (no frontmatter)")
                return

            # Check if examples already exist
            if 'examples' in frontmatter:
                self.stats['skipped_has_examples'] += 1
                if self.verbose:
                    print(f"⊝ SKIP: {file_path.relative_to(self.agents_dir)} (already has examples)")
                return

            # Generate examples
            agent_name = frontmatter.get('name', file_path.stem)
            description = frontmatter.get('description', '')
            category = frontmatter.get('category', 'general')

            examples = ExampleGenerator.generate_examples(agent_name, description, category)

            # Add examples to frontmatter
            frontmatter['examples'] = examples

            # Serialize updated content
            updated_content = FrontmatterParser.serialize(frontmatter, body)

            # Write back to file (if not dry run)
            if not self.dry_run:
                file_path.write_text(updated_content, encoding='utf-8')

            self.stats['updated'] += 1
            rel_path = file_path.relative_to(self.agents_dir)

            if self.verbose or not self.dry_run:
                print(f"✓ {'WOULD UPDATE' if self.dry_run else 'UPDATED'}: {rel_path}")
                if self.verbose:
                    for i, example in enumerate(examples, 1):
                        print(f"  Example {i}: {example['trigger'][:60]}...")

        except Exception as e:
            self.stats['errors'] += 1
            rel_path = file_path.relative_to(self.agents_dir)
            print(f"✗ ERROR: {rel_path}: {e}")

    def _print_summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total files processed:        {self.stats['total']}")
        print(f"  Skipped (has examples):     {self.stats['skipped_has_examples']}")
        print(f"  Skipped (no frontmatter):   {self.stats['skipped_no_frontmatter']}")
        print(f"  Skipped (non-agent):        {self.stats['skipped_readme']}")
        print(f"  {'Would update' if self.dry_run else 'Updated'}:                {self.stats['updated']}")
        print(f"  Errors:                     {self.stats['errors']}")
        print("=" * 70)

        if self.dry_run and self.stats['updated'] > 0:
            print("\n💡 This was a dry run. Use without --dry-run to apply changes.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Add contextual examples to agent markdown files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/add-examples.py                    # Apply changes
  python3 scripts/add-examples.py --dry-run          # Preview changes
  python3 scripts/add-examples.py --verbose          # Show detailed output
  python3 scripts/add-examples.py --dry-run -v       # Preview with details
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output for each file'
    )

    parser.add_argument(
        '--agents-dir',
        type=Path,
        default=None,
        help='Path to agents directory (default: auto-detect from script location)'
    )

    args = parser.parse_args()

    # Determine agents directory
    if args.agents_dir:
        agents_dir = args.agents_dir
    else:
        # Auto-detect: script is in scripts/, agents is sibling directory
        script_dir = Path(__file__).parent
        agents_dir = script_dir.parent / 'agents'

    # Validate agents directory exists
    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}")
        sys.exit(1)

    if not agents_dir.is_dir():
        print(f"Error: Not a directory: {agents_dir}")
        sys.exit(1)

    # Run the updater
    updater = AgentUpdater(agents_dir, dry_run=args.dry_run, verbose=args.verbose)
    updater.process_all_agents()


if __name__ == '__main__':
    main()
