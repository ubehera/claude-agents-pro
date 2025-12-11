"""Configuration management for CLI."""

import json
from pathlib import Path
from typing import Dict, Optional


class Config:
    """Manage CLI configuration and marketplace registry."""

    DEFAULT_CONFIG = {
        'version': '1.0.0',
        'default_scope': 'user',
        'marketplace_url': 'https://github.com/ubehera/claude-agents-pro',
        'quality_threshold': 0.7,
        'auto_update': False
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            config_path: Custom config file path (default: ~/.claude/cli-config.json)
        """
        if config_path is None:
            config_path = Path.home() / '.claude' / 'cli-config.json'

        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    return {**self.DEFAULT_CONFIG, **config}
            except (json.JSONDecodeError, OSError):
                pass

        return self.DEFAULT_CONFIG.copy()

    def save(self):
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value):
        """Set configuration value."""
        self._config[key] = value

    @property
    def marketplace_registry(self) -> Path:
        """Get path to marketplace registry."""
        cli_root = Path(__file__).parent.parent.parent
        return cli_root / 'configs' / 'marketplace.json'

    def load_marketplace_registry(self) -> Dict:
        """Load marketplace registry data."""
        registry_path = self.marketplace_registry

        if not registry_path.exists():
            return {'agents': [], 'tiers': [], 'categories': []}

        with open(registry_path, 'r') as f:
            return json.load(f)

    def get_user_agents_dir(self) -> Path:
        """Get user agents directory."""
        return Path.home() / '.claude' / 'agents'

    def get_project_agents_dir(self) -> Path:
        """Get project agents directory."""
        return Path.cwd() / '.claude' / 'agents'

    def get_repo_agents_dir(self) -> Path:
        """Get repository agents directory."""
        cli_root = Path(__file__).parent.parent.parent
        return cli_root / 'agents'
