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
  valid_categories="orchestrator foundation development specialist expert integration quality finance platform security"
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
    warn=$((warn + 1))
  fi

  # warn if both WebSearch and WebFetch are in tools (only if tools field exists)
  if echo "$tools" | grep -q "WebSearch" && echo "$tools" | grep -q "WebFetch"; then
    echo "[WARN] $base: tools include both WebSearch and WebFetch"
    warn=$((warn + 1))
  fi

  # SECURITY CHECKS
  # Check for potential hardcoded secrets/API keys
  if grep -qiE '(api[_-]?key|secret[_-]?key|password|token)[[:space:]]*[:=][[:space:]]*["\x27][^"\x27]{8,}' "$file"; then
    echo "[SEC]  $base: potential hardcoded secret detected"
    warn=$((warn + 1))
  fi

  # Check for dangerous command patterns
  if grep -qE 'rm\s+-rf\s+[/$~]|sudo\s+rm|chmod\s+777|>\s*/dev/sda' "$file"; then
    echo "[SEC]  $base: potentially dangerous command pattern detected"
    warn=$((warn + 1))
  fi

  # Check for secrets file references that shouldn't be in prompts
  if grep -qiE '\.(env|pem|key|credentials|secrets)\b' "$file" && grep -qE '(read|cat|source|load)' "$file"; then
    echo "[SEC]  $base: references to secret files with read operations"
    warn=$((warn + 1))
  fi

  # Check for prompt injection susceptibility (overly permissive instructions)
  if grep -qiE 'ignore\s+(previous|above|all)\s+instructions|do\s+whatever|you\s+are\s+now' "$file"; then
    echo "[SEC]  $base: potential prompt injection vector detected"
    warn=$((warn + 1))
  fi

  # Check for eval/exec patterns in code examples
  if grep -qE 'eval\s*\(|exec\s*\(' "$file" && ! grep -qE '#.*eval|#.*exec|sandboxed|validated' "$file"; then
    echo "[SEC]  $base: eval/exec without safety context"
    warn=$((warn + 1))
  fi

  if ((${#errs[@]})); then
    echo "[FAIL] $base: ${errs[*]}"
    fail=$((fail + 1))
  else
    rel_path=${file#"$AGENTS_DIR/"}
    printf '[OK]   %s (%s)\n' "$base" "$rel_path"
    pass=$((pass + 1))
  fi
done < <(find "$AGENTS_DIR" -mindepth 1 -type f -name '*.md' -print0 | sort -z)

printf '\nSummary: %d ok, %d warnings, %d failures\n' "$pass" "$warn" "$fail"
exit $fail
