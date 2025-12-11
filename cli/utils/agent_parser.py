"""Agent file parser utility."""

import yaml
from pathlib import Path
from typing import Dict, Tuple


class AgentParser:
    """Parse and extract metadata from agent markdown files."""

    def parse_agent_file(self, agent_file: Path) -> Dict:
        """
        Parse an agent markdown file and extract frontmatter metadata.

        Args:
            agent_file: Path to agent markdown file

        Returns:
            Dictionary containing agent metadata

        Raises:
            ValueError: If file cannot be parsed
        """
        if not agent_file.exists():
            raise ValueError(f"Agent file not found: {agent_file}")

        with open(agent_file, 'r', encoding='utf-8') as f:
            content = f.read()

        frontmatter, body = self._split_frontmatter(content)

        if not frontmatter:
            raise ValueError(f"No frontmatter found in {agent_file}")

        try:
            metadata = yaml.safe_load(frontmatter)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML frontmatter in {agent_file}: {e}")

        # Add body content
        metadata['_body'] = body
        metadata['_file'] = str(agent_file)

        return metadata

    def _split_frontmatter(self, content: str) -> Tuple[str, str]:
        """
        Split markdown content into frontmatter and body.

        Args:
            content: Full markdown file content

        Returns:
            Tuple of (frontmatter, body)
        """
        if not content.startswith('---'):
            return '', content

        parts = content.split('---', 2)

        if len(parts) < 3:
            return '', content

        frontmatter = parts[1].strip()
        body = parts[2].strip()

        return frontmatter, body

    def extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract markdown sections from content.

        Args:
            content: Markdown content

        Returns:
            Dictionary mapping section titles to content
        """
        sections = {}
        current_section = None
        current_content = []

        for line in content.split('\n'):
            if line.startswith('## '):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()

                # Start new section
                current_section = line[3:].strip()
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def validate_metadata(self, metadata: Dict) -> Tuple[bool, list]:
        """
        Validate agent metadata against required schema.

        Args:
            metadata: Agent metadata dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        required_fields = ['name', 'description', 'category', 'complexity', 'model']

        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
            elif not metadata[field]:
                errors.append(f"Empty required field: {field}")

        # Validate category
        valid_categories = ['meta', 'foundation', 'development', 'specialists', 'experts', 'integration', 'quality', 'finance']
        if 'category' in metadata and metadata['category'] not in valid_categories:
            errors.append(f"Invalid category: {metadata['category']}")

        # Validate complexity
        valid_complexity = ['simple', 'moderate', 'complex', 'expert']
        if 'complexity' in metadata and metadata['complexity'] not in valid_complexity:
            errors.append(f"Invalid complexity: {metadata['complexity']}")

        # Validate model
        valid_models = [
            'claude-sonnet-4-5-20250929',
            'claude-opus-4-5-20251101',
            'claude-haiku-4-5-20250929'
        ]
        if 'model' in metadata and metadata['model'] not in valid_models:
            errors.append(f"Invalid model: {metadata['model']}")

        return len(errors) == 0, errors
