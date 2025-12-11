#!/usr/bin/env python3
"""
Setup script for claude-agents-cli.

This provides backward compatibility with older build systems.
Use pyproject.toml for modern installations.
"""

from setuptools import setup, find_packages

setup(
    name="claude-agents-cli",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "rich>=13.0.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "claude-agents=cli.__main__:main",
        ],
    },
    python_requires=">=3.10",
    author="Claude Agents Pro",
    description="CLI distribution tool for managing Claude Code agents",
    license="Apache-2.0",
    keywords="claude ai agents cli code-assistant",
    url="https://github.com/ubehera/claude-agents-pro",
)
