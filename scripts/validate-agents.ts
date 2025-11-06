#!/usr/bin/env node

/**
 * Agent Validation Script for ubehera/agent-forge
 *
 * Validates agent markdown files against JSON schema and custom rules:
 * - YAML frontmatter structure and completeness
 * - Filename/name field consistency
 * - Tool usage optimization
 * - Description quality
 * - Duplicate name detection
 *
 * Exit codes:
 *   0 - All validations passed
 *   1 - Validation errors found
 */

import * as fs from 'fs';
import * as path from 'path';
import matter from 'gray-matter';
import Ajv, { ValidateFunction } from 'ajv';
import { globSync } from 'glob';

// Initialize AJV for JSON schema validation
const ajv = new Ajv({ allErrors: true, strict: false });

// Load agent schema
const schemaPath = path.join(__dirname, 'agent-schema.json');
const agentSchema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
const validateAgent: ValidateFunction = ajv.compile(agentSchema);

// Validation result tracking
interface ValidationIssue {
  file: string;
  message: string;
  severity: 'error' | 'warning';
  details?: any;
}

const issues: ValidationIssue[] = [];
let filesProcessed = 0;

// Valid Claude Code tools (as of 2025)
const VALID_TOOLS = [
  'Read', 'Write', 'Edit', 'MultiEdit',
  'Bash', 'WebSearch', 'WebFetch',
  'Task', 'TodoWrite',
  'Grep', 'Glob', 'LS',
  'NotebookRead', 'NotebookEdit'
];

const TOOL_OPTIMIZATION_THRESHOLD = 7;

/**
 * Parse frontmatter from agent markdown file
 */
function parseAgentFile(filePath: string): { frontmatter: any; content: string } | null {
  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const parsed = matter(fileContent);

    if (!parsed.data || Object.keys(parsed.data).length === 0) {
      issues.push({
        file: filePath,
        message: 'No frontmatter found (must start with ---)',
        severity: 'error'
      });
      return null;
    }

    return {
      frontmatter: parsed.data,
      content: parsed.content
    };
  } catch (error: any) {
    issues.push({
      file: filePath,
      message: `Failed to parse file: ${error.message}`,
      severity: 'error'
    });
    return null;
  }
}

/**
 * Validate against JSON schema
 */
function validateAgainstSchema(filePath: string, frontmatter: any): boolean {
  const valid = validateAgent(frontmatter);

  if (!valid && validateAgent.errors) {
    for (const error of validateAgent.errors) {
      let message = error.message || 'Unknown schema error';

      // Enhance error messages for clarity
      if (error.instancePath) {
        const field = error.instancePath.replace(/^\//, '');
        message = `Field '${field}' ${message}`;
      } else if (error.keyword === 'required') {
        const missing = error.params.missingProperty;
        message = `Missing required field '${missing}'`;
      } else if (error.keyword === 'pattern') {
        message = `${message} (expected pattern: ${error.params.pattern})`;
      } else if (error.keyword === 'enum') {
        message = `${message}. Valid values: ${error.params.allowedValues?.join(', ')}`;
      }

      issues.push({
        file: filePath,
        message,
        severity: 'error',
        details: error
      });
    }
  }

  return valid;
}

/**
 * Check filename matches name field
 */
function validateFilename(filePath: string, frontmatter: any): void {
  const fileName = path.basename(filePath, '.md');
  const nameField = frontmatter.name;

  if (nameField && fileName !== nameField) {
    issues.push({
      file: filePath,
      message: `Filename '${fileName}.md' doesn't match name field '${nameField}'. Expected '${nameField}.md'`,
      severity: 'error'
    });
  }
}

/**
 * Validate description quality
 */
function validateDescription(filePath: string, frontmatter: any): void {
  const description = frontmatter.description;

  if (!description) {
    return; // Schema validation will catch this
  }

  if (description.length < 50) {
    issues.push({
      file: filePath,
      message: `Description too short (${description.length} chars, minimum 50 recommended)`,
      severity: 'warning'
    });
  }

  if (description.length > 500) {
    issues.push({
      file: filePath,
      message: `Description too long (${description.length} chars, maximum 500 recommended)`,
      severity: 'warning'
    });
  }

  // Check for clarity indicators
  const hasActionVerbs = /\b(design|implement|build|manage|optimize|analyze|create|configure)\b/i.test(description);
  if (!hasActionVerbs) {
    issues.push({
      file: filePath,
      message: 'Description should include action verbs (design, implement, build, etc.) for clarity',
      severity: 'warning'
    });
  }
}

/**
 * Validate tool usage
 */
function validateTools(filePath: string, frontmatter: any): void {
  if (!frontmatter.tools) {
    return; // No tools specified = inherit all (valid pattern)
  }

  // Handle both string and array formats
  let toolsList: string[];
  if (typeof frontmatter.tools === 'string') {
    toolsList = frontmatter.tools
      .split(',')
      .map((t: string) => t.trim())
      .filter((t: string) => t.length > 0);
  } else if (Array.isArray(frontmatter.tools)) {
    toolsList = frontmatter.tools.map((t: any) => String(t).trim()).filter(t => t.length > 0);
  } else {
    issues.push({
      file: filePath,
      message: `Tools field must be a string or array, got ${typeof frontmatter.tools}`,
      severity: 'error'
    });
    return;
  }

  // Check for unknown tools
  for (const tool of toolsList) {
    if (!VALID_TOOLS.includes(tool)) {
      issues.push({
        file: filePath,
        message: `Unknown tool '${tool}'. Valid tools: ${VALID_TOOLS.join(', ')}`,
        severity: 'warning'
      });
    }
  }

  // Warn on excessive tools
  if (toolsList.length > TOOL_OPTIMIZATION_THRESHOLD) {
    issues.push({
      file: filePath,
      message: `Tools list is long (${toolsList.length} tools, recommended: ≤${TOOL_OPTIMIZATION_THRESHOLD}). Consider least-privilege principle.`,
      severity: 'warning'
    });
  }

  // Check for redundant tool combinations
  if (toolsList.includes('WebSearch') && toolsList.includes('WebFetch')) {
    issues.push({
      file: filePath,
      message: 'Both WebSearch and WebFetch specified. Consider using only one unless both are genuinely needed.',
      severity: 'warning'
    });
  }
}

/**
 * Validate model specification
 */
function validateModel(filePath: string, frontmatter: any): void {
  if (frontmatter.model && !frontmatter.model_rationale) {
    issues.push({
      file: filePath,
      message: 'Model specified without model_rationale. Provide justification for model choice.',
      severity: 'error'
    });
  }

  if (!frontmatter.model && frontmatter.model_rationale) {
    issues.push({
      file: filePath,
      message: 'Model rationale provided without specifying a model',
      severity: 'warning'
    });
  }
}

/**
 * Validate content structure
 */
function validateContent(filePath: string, content: string, frontmatter: any): void {
  const contentLower = content.toLowerCase();

  // Check for "You are" opening statement
  if (!contentLower.includes('you are')) {
    issues.push({
      file: filePath,
      message: 'Missing opening statement "You are a..." in agent body',
      severity: 'warning'
    });
  }

  // Check for section structure
  const hasSections = content.includes('## ');
  if (!hasSections) {
    issues.push({
      file: filePath,
      message: 'No markdown sections found. Consider adding ## headings for better structure',
      severity: 'warning'
    });
  }

  // Warn if agent is too short (might be incomplete)
  const wordCount = content.split(/\s+/).length;
  if (wordCount < 500) {
    issues.push({
      file: filePath,
      message: `Agent content is short (${wordCount} words). Consider expanding with examples and guidance.`,
      severity: 'warning'
    });
  }

  // Check for capabilities section if complexity is specified
  if (frontmatter.complexity === 'complex' && !contentLower.includes('capabilit')) {
    issues.push({
      file: filePath,
      message: 'Complex agent should document capabilities explicitly',
      severity: 'warning'
    });
  }
}

/**
 * Check for duplicate agent names
 */
function checkDuplicateNames(agentFiles: string[]): void {
  const nameMap = new Map<string, string[]>();

  for (const file of agentFiles) {
    const parsed = parseAgentFile(file);
    if (!parsed) continue;

    const name = parsed.frontmatter.name;
    if (!name) continue;

    if (!nameMap.has(name)) {
      nameMap.set(name, []);
    }
    nameMap.get(name)!.push(file);
  }

  // Report duplicates
  for (const [name, files] of nameMap.entries()) {
    if (files.length > 1) {
      for (const file of files) {
        issues.push({
          file,
          message: `Duplicate agent name '${name}' found in: ${files.join(', ')}`,
          severity: 'error'
        });
      }
    }
  }
}

/**
 * Validate single agent file
 */
function validateAgentFile(filePath: string): void {
  filesProcessed++;

  const parsed = parseAgentFile(filePath);
  if (!parsed) return;

  const { frontmatter, content } = parsed;

  // Schema validation
  validateAgainstSchema(filePath, frontmatter);

  // Custom validations
  validateFilename(filePath, frontmatter);
  validateDescription(filePath, frontmatter);
  validateTools(filePath, frontmatter);
  validateModel(filePath, frontmatter);
  validateContent(filePath, content, frontmatter);
}

/**
 * Generate validation report
 */
function generateReport(): {
  summary: { total: number; errors: number; warnings: number };
  issues: ValidationIssue[];
} {
  const errors = issues.filter(i => i.severity === 'error');
  const warnings = issues.filter(i => i.severity === 'warning');

  return {
    summary: {
      total: filesProcessed,
      errors: errors.length,
      warnings: warnings.length
    },
    issues
  };
}

/**
 * Print console report
 */
function printConsoleReport(report: ReturnType<typeof generateReport>): void {
  console.log(`\n\x1b[34m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m`);
  console.log(`\x1b[34m  Agent Validation Report\x1b[0m`);
  console.log(`\x1b[34m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\x1b[0m\n`);

  console.log(`Files Processed: ${report.summary.total}`);
  console.log(`Errors: \x1b[31m${report.summary.errors}\x1b[0m`);
  console.log(`Warnings: \x1b[33m${report.summary.warnings}\x1b[0m\n`);

  // Group issues by file
  const issuesByFile = new Map<string, ValidationIssue[]>();
  for (const issue of report.issues) {
    if (!issuesByFile.has(issue.file)) {
      issuesByFile.set(issue.file, []);
    }
    issuesByFile.get(issue.file)!.push(issue);
  }

  // Print errors first
  const filesWithErrors = Array.from(issuesByFile.entries())
    .filter(([_, issues]) => issues.some(i => i.severity === 'error'));

  if (filesWithErrors.length > 0) {
    console.log(`\x1b[31m❌ Errors:\x1b[0m\n`);
    for (const [file, fileIssues] of filesWithErrors) {
      const relPath = path.relative(process.cwd(), file);
      console.log(`  \x1b[31m${relPath}\x1b[0m`);

      for (const issue of fileIssues) {
        if (issue.severity === 'error') {
          console.log(`    • ${issue.message}`);
        }
      }
      console.log();
    }
  }

  // Print warnings
  const filesWithWarnings = Array.from(issuesByFile.entries())
    .filter(([_, issues]) => issues.some(i => i.severity === 'warning'));

  if (filesWithWarnings.length > 0) {
    console.log(`\x1b[33m⚠️  Warnings:\x1b[0m\n`);
    for (const [file, fileIssues] of filesWithWarnings) {
      const relPath = path.relative(process.cwd(), file);
      console.log(`  \x1b[33m${relPath}\x1b[0m`);

      for (const issue of fileIssues) {
        if (issue.severity === 'warning') {
          console.log(`    • ${issue.message}`);
        }
      }
      console.log();
    }
  }

  // Success message
  if (report.summary.errors === 0 && report.summary.warnings === 0) {
    console.log(`\x1b[32m✅ All validations passed!\x1b[0m\n`);
  } else if (report.summary.errors === 0) {
    console.log(`\x1b[32m✅ No errors found (${report.summary.warnings} warnings)\x1b[0m\n`);
  } else {
    console.log(`\x1b[31m❌ Validation failed with ${report.summary.errors} error(s)\x1b[0m\n`);
  }
}

/**
 * Main execution
 */
function main() {
  const args = process.argv.slice(2);

  // Parse command line arguments
  let agentsDir = 'agents';
  let specificAgent: string | null = null;
  let outputFile = 'validation-report.json';

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--agents-dir' && i + 1 < args.length) {
      agentsDir = args[++i];
    } else if (args[i] === '--agent' && i + 1 < args.length) {
      specificAgent = args[++i];
    } else if (args[i] === '--output' && i + 1 < args.length) {
      outputFile = args[++i];
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.log(`
Usage: tsx validate-agents.ts [options]

Options:
  --agents-dir <path>  Directory containing agent files (default: agents)
  --agent <path>       Validate specific agent file
  --output <path>      Output JSON report file (default: validation-report.json)
  --help, -h          Show this help message

Examples:
  tsx validate-agents.ts
  tsx validate-agents.ts --agent agents/01-foundation/api-platform-engineer.md
  tsx validate-agents.ts --agents-dir agents --output report.json
      `);
      process.exit(0);
    }
  }

  console.log(`\x1b[34mValidating agents...\x1b[0m`);

  let agentFiles: string[];

  if (specificAgent) {
    // Validate single agent
    if (!fs.existsSync(specificAgent)) {
      console.error(`\x1b[31mError: Agent file not found: ${specificAgent}\x1b[0m`);
      process.exit(1);
    }
    agentFiles = [specificAgent];
  } else {
    // Find all agent markdown files
    const agentsDirPath = path.resolve(agentsDir);

    if (!fs.existsSync(agentsDirPath)) {
      console.error(`\x1b[31mError: Agents directory not found: ${agentsDirPath}\x1b[0m`);
      process.exit(1);
    }

    agentFiles = globSync(`${agentsDirPath}/**/*.md`, {
      ignore: ['**/README.md', '**/TESTING.md', '**/AGENT_CHECKLIST.md', '**/INDEX.md']
    });
  }

  if (agentFiles.length === 0) {
    console.log(`\x1b[33mNo agent files found in ${agentsDir}\x1b[0m`);
    process.exit(0);
  }

  // Validate each agent file
  for (const file of agentFiles) {
    validateAgentFile(file);
  }

  // Check for duplicate names (only if validating multiple files)
  if (agentFiles.length > 1) {
    checkDuplicateNames(agentFiles);
  }

  // Generate reports
  const report = generateReport();

  // Print console report
  printConsoleReport(report);

  // Write JSON report
  const jsonReport = {
    timestamp: new Date().toISOString(),
    ...report
  };

  fs.writeFileSync(outputFile, JSON.stringify(jsonReport, null, 2));
  console.log(`\x1b[90mDetailed report: ${outputFile}\x1b[0m\n`);

  // Exit with appropriate code
  process.exit(report.summary.errors > 0 ? 1 : 0);
}

// Execute if run directly
if (require.main === module) {
  main();
}

export {
  validateAgentFile,
  generateReport,
  ValidationIssue
};
