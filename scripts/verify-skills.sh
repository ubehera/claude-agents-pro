#!/bin/bash
set -e

SKILLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../skills && pwd)"

pass=0; warn=0; fail=0

echo "Verifying skills in: $SKILLS_DIR"
echo ""

while IFS= read -r -d '' file; do
  base=$(basename "$file")

  # Skip README files
  case "$base" in
    README.md|TESTING.md|*.example.md)
      continue
      ;;
  esac

  # Parse frontmatter
  name=$(sed -n '1,20p' "$file" | awk -F: '/^name:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')
  desc_ok=$(sed -n '1,20p' "$file" | grep -q '^description:' && echo ok || echo no)
  triggers=$(sed -n '1,20p' "$file" | grep -q 'trigger_keywords:' && echo ok || echo no)

  errs=()
  warns_arr=()

  # Check frontmatter start
  [[ $(head -n1 "$file") == "---" ]] || errs+=("missing frontmatter start ---")

  # Required fields
  [[ -n "$name" ]] || errs+=("missing name")
  [[ "$desc_ok" == ok ]] || errs+=("missing description")
  [[ "$triggers" == ok ]] || errs+=("missing trigger_keywords")

  # Name should match filename (without .md)
  expected="${base%.md}"
  if [[ -n "$name" && "$name" != "$expected" ]]; then
    errs+=("name '$name' != filename '$expected'")
  fi

  # Check for Core Concepts section
  if ! grep -q '## Core Concepts' "$file"; then
    warns_arr+=("missing '## Core Concepts' section")
  fi

  # Check for Implementation Patterns section
  if ! grep -q '## Implementation Patterns\|## Implementation\|## Patterns' "$file"; then
    warns_arr+=("missing implementation patterns section")
  fi

  # Check for code examples
  if ! grep -q '```' "$file"; then
    warns_arr+=("no code examples found")
  fi

  # SECURITY CHECKS
  # Check for potential hardcoded secrets/API keys
  if grep -qiE '(api[_-]?key|secret[_-]?key|password|token)[[:space:]]*[:=][[:space:]]*["'\''"][^"'\'']{8,}' "$file"; then
    if ! grep -qiE '(example|placeholder|your[_-]|<.*>|\$\{)' "$file"; then
      echo "[SEC]  $base: potential hardcoded secret detected"
      warn=$((warn + 1))
    fi
  fi

  # Check for dangerous command patterns
  if grep -qE 'rm\s+-rf\s+[/$~]|sudo\s+rm|chmod\s+777|>\s*/dev/sda' "$file"; then
    if ! grep -qiE '(warning|caution|never|avoid|bad practice)' "$file"; then
      echo "[SEC]  $base: potentially dangerous command pattern detected"
      warn=$((warn + 1))
    fi
  fi

  # Check for prompt injection susceptibility
  if grep -qiE 'ignore\s+(previous|above|all)\s+instructions|do\s+whatever|you\s+are\s+now' "$file"; then
    echo "[SEC]  $base: potential prompt injection vector detected"
    warn=$((warn + 1))
  fi

  # Check for eval/exec patterns in code examples
  if grep -qE 'eval\s*\(|exec\s*\(' "$file" && ! grep -qE '#.*eval|#.*exec|sandboxed|validated|warning|caution' "$file"; then
    echo "[SEC]  $base: eval/exec without safety context"
    warn=$((warn + 1))
  fi

  # Print warnings
  for w in "${warns_arr[@]}"; do
    echo "[WARN] $base: $w"
    warn=$((warn + 1))
  done

  # Print results
  if ((${#errs[@]})); then
    echo "[FAIL] $base: ${errs[*]}"
    fail=$((fail + 1))
  else
    rel_path=${file#"$SKILLS_DIR/"}
    printf '[OK]   %s (%s)\n' "$base" "$rel_path"
    pass=$((pass + 1))
  fi
done < <(find "$SKILLS_DIR" -mindepth 1 -type f -name '*.md' -print0 | sort -z)

echo ""
echo "Summary: $pass ok, $warn warnings, $fail failures"
exit $fail
