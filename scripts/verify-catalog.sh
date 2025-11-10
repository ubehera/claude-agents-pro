#!/bin/bash
# verify-catalog.sh - Ensures agents/README.md catalog matches actual agent files
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$SCRIPT_DIR/../agents"
README="$AGENTS_DIR/README.md"

echo "Verifying agent catalog consistency..."

# Extract agent names from README table (lines with | starting after ## Active Agents)
# Format: | `agent-name` | tier | domain | tools |
readme_agents=$(sed -n '/^## Active Agents/,/^##/p' "$README" | \
  grep '^\|' | \
  grep -v '^| Agent' | \
  grep -v '^|----' | \
  grep -v '^##' | \
  grep -oP '\| `\K[^`]+' | \
  sort -u)

# Find actual agent files (exclude README.md, TESTING.md, AGENT_CHECKLIST.md, finance-glossary.md)
actual_agents=$(find "$AGENTS_DIR" -type f -name '*.md' \
  ! -name 'README.md' \
  ! -name 'TESTING.md' \
  ! -name 'AGENT_CHECKLIST.md' \
  ! -name 'finance-glossary.md' \
  -exec basename {} .md \; | \
  sort -u)

# Convert to arrays for comparison
readarray -t readme_array <<< "$readme_agents"
readarray -t actual_array <<< "$actual_agents"

# Count
readme_count=${#readme_array[@]}
actual_count=${#actual_array[@]}

echo "README catalog: $readme_count agents"
echo "Actual agent files: $actual_count agents"

# Find agents in README but not in files
echo ""
echo "Checking for catalog mismatches..."

missing_files=()
for agent in "${readme_array[@]}"; do
  if [[ ! " ${actual_array[@]} " =~ " ${agent} " ]]; then
    missing_files+=("$agent")
  fi
done

# Find agent files not in README
missing_catalog=()
for agent in "${actual_array[@]}"; do
  if [[ ! " ${readme_array[@]} " =~ " ${agent} " ]]; then
    missing_catalog+=("$agent")
  fi
done

# Report results
errors=0

if [ ${#missing_files[@]} -gt 0 ]; then
  echo ""
  echo "❌ Agents in README but file not found:"
  for agent in "${missing_files[@]}"; do
    echo "   - $agent"
  done
  errors=$((errors + ${#missing_files[@]}))
fi

if [ ${#missing_catalog[@]} -gt 0 ]; then
  echo ""
  echo "❌ Agent files exist but not in README catalog:"
  for agent in "${missing_catalog[@]}"; do
    echo "   - $agent"
    # Try to find the file location
    file_path=$(find "$AGENTS_DIR" -type f -name "${agent}.md" ! -name 'README.md' ! -name 'TESTING.md' ! -name 'AGENT_CHECKLIST.md')
    if [ -n "$file_path" ]; then
      echo "      Location: $file_path"
    fi
  done
  errors=$((errors + ${#missing_catalog[@]}))
fi

# Check agent count in README.md (parent directory)
parent_readme="$SCRIPT_DIR/../README.md"
if [ -f "$parent_readme" ]; then
  # Extract agent count from badge or text
  badge_count=$(grep -oP 'agents-\K\d+' "$parent_readme" | head -1)

  if [ -n "$badge_count" ] && [ "$badge_count" != "$actual_count" ]; then
    echo ""
    echo "⚠️  Agent count mismatch in README.md:"
    echo "   Badge shows: $badge_count agents"
    echo "   Actual count: $actual_count agents"
    echo "   Update README.md badge to reflect correct count"
  fi
fi

# Summary
echo ""
if [ $errors -eq 0 ]; then
  echo "✅ Catalog verification passed"
  echo "   All $actual_count agent files are properly cataloged"
  exit 0
else
  echo "❌ Catalog verification failed with $errors error(s)"
  echo ""
  echo "To fix:"
  echo "1. Add missing agents to agents/README.md Active Agents table"
  echo "2. Remove stale entries from agents/README.md"
  echo "3. Update configs/agent-metadata.json if needed"
  echo "4. Update AGENTS.md top-level catalog"
  exit 1
fi
