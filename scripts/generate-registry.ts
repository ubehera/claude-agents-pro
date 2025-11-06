#!/usr/bin/env tsx

import { readFile, writeFile, mkdir } from 'fs/promises';
import { glob } from 'glob';
import matter from 'gray-matter';
import path from 'path';
import { fileURLToPath } from 'url';

// ============================================================================
// Type Definitions
// ============================================================================

interface AgentFrontmatter {
  name: string;
  description: string;
  tools?: string[];
  model?: string;
  model_rationale?: string;
  [key: string]: unknown;
}

interface AgentEntry {
  name: string;
  tier: string;
  category: string;
  complexity: string;
  model: string;
  description: string;
  capabilities: string[];
  tags: string[];
  file: string;
}

interface RegistryStats {
  total: number;
  by_tier: Record<string, number>;
  by_category: Record<string, number>;
  by_complexity: Record<string, number>;
  by_model: Record<string, number>;
}

interface AgentRegistry {
  generated: string;
  version: string;
  agents: AgentEntry[];
  stats: RegistryStats;
}

// ============================================================================
// Configuration
// ============================================================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.join(__dirname, '..');
const AGENTS_DIR = path.join(REPO_ROOT, 'agents');
const OUTPUT_PATH = path.join(REPO_ROOT, 'configs', 'agent-registry.json');

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Extracts tier from file path (e.g., "01-foundation" from "agents/01-foundation/api-platform-engineer.md")
 */
function extractTier(filePath: string): string {
  const parts = filePath.split(path.sep);
  const tierPart = parts.find(part => /^\d{2}-/.test(part));
  return tierPart || 'uncategorized';
}

/**
 * Determines category based on tier
 */
function determineCategory(tier: string): string {
  const tierMap: Record<string, string> = {
    '00-meta': 'orchestration',
    '01-foundation': 'foundation',
    '02-development': 'development',
    '03-specialists': 'specialist',
    '04-experts': 'expert',
    '05-domains': 'domain',
    '06-integration': 'integration',
    '07-quality': 'quality',
    '08-automation': 'automation',
    '08-finance': 'finance',
    '09-enterprise': 'enterprise',
  };
  return tierMap[tier] || 'uncategorized';
}

/**
 * Determines complexity based on description length and tool count
 */
function determineComplexity(description: string, tools: string[]): string {
  const descLength = description.length;
  const toolCount = tools.length;

  if (descLength > 300 || toolCount >= 6) return 'complex';
  if (descLength > 150 || toolCount >= 4) return 'moderate';
  return 'simple';
}

/**
 * Extracts capabilities from description (capitalized keywords)
 */
function extractCapabilities(description: string): string[] {
  // Extract key technical terms and patterns
  const capabilities: string[] = [];
  const keywords = [
    'REST', 'GraphQL', 'gRPC', 'API', 'OpenAPI', 'OAuth', 'JWT',
    'React', 'Vue', 'Angular', 'TypeScript', 'JavaScript', 'Python', 'Go', 'Java',
    'AWS', 'Azure', 'GCP', 'Kubernetes', 'Docker', 'Terraform',
    'PostgreSQL', 'MongoDB', 'Redis', 'DynamoDB',
    'CI/CD', 'DevOps', 'MLOps', 'SRE', 'Security',
    'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
    'Testing', 'Performance', 'Observability', 'Monitoring',
    'Frontend', 'Backend', 'Full Stack', 'Mobile',
    'Microservices', 'Event-Driven', 'CQRS', 'DDD',
  ];

  for (const keyword of keywords) {
    const regex = new RegExp(`\\b${keyword}\\b`, 'i');
    if (regex.test(description)) {
      capabilities.push(keyword);
    }
  }

  return [...new Set(capabilities)].slice(0, 10); // Limit to 10 unique capabilities
}

/**
 * Extracts tags from description (lowercased keywords)
 */
function extractTags(description: string, name: string): string[] {
  const tags = new Set<string>();

  // Add name-based tags
  const nameParts = name.split('-');
  nameParts.forEach(part => {
    if (part.length > 3) tags.add(part);
  });

  // Extract tags from description
  const tagKeywords = [
    'api', 'rest', 'graphql', 'design', 'architecture', 'security',
    'testing', 'performance', 'optimization', 'monitoring', 'observability',
    'cloud', 'aws', 'kubernetes', 'docker', 'devops', 'ci/cd',
    'frontend', 'backend', 'mobile', 'react', 'typescript', 'python',
    'database', 'sql', 'nosql', 'machine learning', 'ml', 'ai',
    'documentation', 'research', 'review', 'debugging',
  ];

  tagKeywords.forEach(keyword => {
    const regex = new RegExp(`\\b${keyword}\\b`, 'i');
    if (regex.test(description)) {
      tags.add(keyword.toLowerCase());
    }
  });

  return Array.from(tags).slice(0, 15); // Limit to 15 tags
}

/**
 * Normalizes model name to standard format
 */
function normalizeModel(model?: string): string {
  if (!model) return 'sonnet';

  const modelLower = model.toLowerCase();
  if (modelLower.includes('opus')) return 'opus';
  if (modelLower.includes('sonnet')) return 'sonnet';
  if (modelLower.includes('haiku')) return 'haiku';

  return 'sonnet'; // Default
}

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Parses a single agent markdown file and extracts metadata
 */
async function parseAgentFile(filePath: string): Promise<AgentEntry | null> {
  try {
    const content = await readFile(filePath, 'utf-8');
    const { data } = matter(content);

    // Validate required fields
    if (!data.name || !data.description) {
      console.warn(`⚠ Skipping ${filePath}: missing required frontmatter fields`);
      return null;
    }

    const frontmatter = data as AgentFrontmatter;
    const tier = extractTier(filePath);
    const category = determineCategory(tier);
    const tools = Array.isArray(frontmatter.tools)
      ? frontmatter.tools
      : typeof frontmatter.tools === 'string'
        ? frontmatter.tools.split(/[,\s]+/).filter(Boolean)
        : [];

    const complexity = determineComplexity(frontmatter.description, tools);
    const model = normalizeModel(frontmatter.model);
    const capabilities = extractCapabilities(frontmatter.description);
    const tags = extractTags(frontmatter.description, frontmatter.name);
    const relativeFile = path.relative(REPO_ROOT, filePath);

    return {
      name: frontmatter.name,
      tier,
      category,
      complexity,
      model,
      description: frontmatter.description.trim(),
      capabilities,
      tags,
      file: relativeFile,
    };
  } catch (error) {
    console.error(`✗ Error parsing ${filePath}:`, error);
    return null;
  }
}

/**
 * Scans all agent files and builds registry entries
 */
async function scanAgents(): Promise<AgentEntry[]> {
  const pattern = path.join(AGENTS_DIR, '**', '*.md');
  const files = await glob(pattern, {
    ignore: ['**/README.md', '**/AGENT_CHECKLIST.md', '**/TESTING.md'],
  });

  console.log(`Found ${files.length} agent files to process...`);

  const agents: AgentEntry[] = [];
  for (const file of files) {
    const agent = await parseAgentFile(file);
    if (agent) {
      agents.push(agent);
    }
  }

  // Sort by tier then name
  return agents.sort((a, b) => {
    const tierCompare = a.tier.localeCompare(b.tier);
    return tierCompare !== 0 ? tierCompare : a.name.localeCompare(b.name);
  });
}

/**
 * Calculates statistics for the registry
 */
function calculateStats(agents: AgentEntry[]): RegistryStats {
  const stats: RegistryStats = {
    total: agents.length,
    by_tier: {},
    by_category: {},
    by_complexity: {},
    by_model: {},
  };

  for (const agent of agents) {
    // Count by tier
    stats.by_tier[agent.tier] = (stats.by_tier[agent.tier] || 0) + 1;

    // Count by category
    stats.by_category[agent.category] = (stats.by_category[agent.category] || 0) + 1;

    // Count by complexity
    stats.by_complexity[agent.complexity] = (stats.by_complexity[agent.complexity] || 0) + 1;

    // Count by model
    stats.by_model[agent.model] = (stats.by_model[agent.model] || 0) + 1;
  }

  return stats;
}

/**
 * Generates the complete registry JSON
 */
async function generateRegistry(): Promise<void> {
  try {
    console.log('🚀 Generating agent registry...\n');

    const agents = await scanAgents();

    if (agents.length === 0) {
      throw new Error('No valid agent files found');
    }

    const stats = calculateStats(agents);

    const registry: AgentRegistry = {
      generated: new Date().toISOString(),
      version: '1.0.0',
      agents,
      stats,
    };

    // Ensure output directory exists
    await mkdir(path.dirname(OUTPUT_PATH), { recursive: true });

    // Write registry with pretty formatting
    await writeFile(OUTPUT_PATH, JSON.stringify(registry, null, 2) + '\n');

    console.log('✓ Registry generated successfully!\n');
    console.log('📊 Statistics:');
    console.log(`  • Total agents: ${stats.total}`);
    console.log(`  • By tier:`);
    Object.entries(stats.by_tier)
      .sort(([a], [b]) => a.localeCompare(b))
      .forEach(([tier, count]) => {
        console.log(`    - ${tier}: ${count}`);
      });
    console.log(`  • By category:`);
    Object.entries(stats.by_category)
      .sort(([, a], [, b]) => b - a)
      .forEach(([category, count]) => {
        console.log(`    - ${category}: ${count}`);
      });
    console.log(`  • By complexity:`);
    Object.entries(stats.by_complexity)
      .sort(([, a], [, b]) => b - a)
      .forEach(([complexity, count]) => {
        console.log(`    - ${complexity}: ${count}`);
      });
    console.log(`  • By model:`);
    Object.entries(stats.by_model)
      .sort(([, a], [, b]) => b - a)
      .forEach(([model, count]) => {
        console.log(`    - ${model}: ${count}`);
      });
    console.log(`\n📄 Output: ${OUTPUT_PATH}`);
  } catch (error) {
    console.error('✗ Error generating registry:', error);
    process.exit(1);
  }
}

// ============================================================================
// Entry Point
// ============================================================================

generateRegistry();
