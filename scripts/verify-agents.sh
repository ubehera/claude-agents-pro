#!/bin/bash
set -e

AGENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../agents && pwd)"

pass=0; warn=0; fail=0

echo "Verifying agents in: $AGENTS_DIR"

while IFS= read -r -d '' file; do
  base=$(basename "$file")
  case "$base" in
    README.md|TESTING.md|AGENT_CHECKLIST.md)
      continue
      ;;
  esac

  name=$(sed -n '1,20p' "$file" | awk -F: '/^name:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')
  desc_ok=$(sed -n '1,40p' "$file" | grep -q '^description:' && echo ok || echo no)
  category=$(sed -n '1,40p' "$file" | awk -F: '/^category:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')
  complexity=$(sed -n '1,40p' "$file" | awk -F: '/^complexity:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')
  model=$(sed -n '1,40p' "$file" | awk -F: '/^model:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')
  tools=$(sed -n '1,40p' "$file" | awk -F: '/^tools:/ {sub(/^ /, "", $2); print $2; exit}')

  errs=()
  [[ $(head -n1 "$file") == "---" ]] || errs+=("missing frontmatter start ---")
  [[ -n "$name" ]] || errs+=("missing name")
  [[ "$desc_ok" == ok ]] || errs+=("missing description")
  [[ -n "$category" ]] || errs+=("missing category")
  [[ -n "$complexity" ]] || errs+=("missing complexity")
  [[ -n "$model" ]] || errs+=("missing model")

  # name should match filename (without .md)
  expected="${base%.md}"
  if [[ -n "$name" && "$name" != "$expected" ]]; then
    errs+=("name '$name' != filename '$expected'")
  fi

  # validate category is one of the allowed values
  valid_categories="orchestrator foundation development specialist expert integration quality finance"
  if [[ -n "$category" ]] && ! echo "$valid_categories" | grep -qw "$category"; then
    errs+=("invalid category '$category' (must be one of: $valid_categories)")
  fi

  # validate complexity is one of the allowed values
  valid_complexity="simple moderate complex expert"
  if [[ -n "$complexity" ]] && ! echo "$valid_complexity" | grep -qw "$complexity"; then
    errs+=("invalid complexity '$complexity' (must be one of: $valid_complexity)")
  fi

  # warn if tools field is present (this repo uses tool inheritance)
  if [[ -n "$tools" ]]; then
    echo "[WARN] $base: has explicit 'tools' field (this repo uses tool inheritance - consider removing)"
    ((warn++))
  fi

  # warn if both WebSearch and WebFetch are in tools (only if tools field exists)
  if echo "$tools" | grep -q "WebSearch" && echo "$tools" | grep -q "WebFetch"; then
    echo "[WARN] $base: tools include both WebSearch and WebFetch"
    ((warn++))
  fi

  if ((${#errs[@]})); then
    echo "[FAIL] $base: ${errs[*]}"
    ((fail++))
  else
    rel_path=${file#"$AGENTS_DIR/"}
    printf '[OK]   %s (%s)\n' "$base" "$rel_path"
    ((pass++))
  fi
done < <(find "$AGENTS_DIR" -mindepth 1 -type f -name '*.md' -print0 | sort -z)

echo "\nSummary: $pass ok, $warn warnings, $fail failures"
exit $fail
