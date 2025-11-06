#!/usr/bin/env python3
import re
from pathlib import Path

agents_dir = Path("/Users/umank/Code/agent-repos/ubehera/agents")
agent_files = list(agents_dir.glob("**/*.md"))

for agent_file in agent_files:
    content = agent_file.read_text()

    # Check if has model but missing model_rationale
    has_model = re.search(r'^model: claude-sonnet', content, re.MULTILINE)
    has_rationale = re.search(r'^model_rationale:', content, re.MULTILINE)

    if has_model and not has_rationale:
        # Add model_rationale after model line
        updated = re.sub(
            r'(^model: claude-sonnet-4-5-20250929$)',
            r'\1\nmodel_rationale: Balanced performance for complex analysis requiring deep technical reasoning',
            content,
            flags=re.MULTILINE
        )
        agent_file.write_text(updated)
        print(f"Fixed: {agent_file.name}")

print("\nDone!")
