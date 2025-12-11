# Makefile for Claude Agents CLI
# Provides convenient shortcuts for common development tasks

.PHONY: help install install-dev test clean lint format build publish demo

help:
	@echo "Claude Agents CLI - Development Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install CLI in development mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  test          - Run quick test suite"
	@echo "  clean         - Remove build artifacts and virtual environment"
	@echo "  lint          - Run code quality checks (ruff)"
	@echo "  format        - Format code (black)"
	@echo "  build         - Build distribution packages"
	@echo "  publish       - Publish to PyPI (requires credentials)"
	@echo "  demo          - Run demo script"
	@echo ""

install:
	@echo "Installing CLI in development mode..."
	pip install -e .
	@echo "✓ Installation complete"
	@echo "Run 'claude-agents --help' to get started"

install-dev:
	@echo "Installing CLI with development dependencies..."
	pip install -e ".[dev]"
	@echo "✓ Development installation complete"

test:
	@echo "Running CLI test suite..."
	@chmod +x test-cli.sh
	@./test-cli.sh

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf commands/__pycache__/
	rm -rf utils/__pycache__/
	rm -rf venv/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"

lint:
	@echo "Running code quality checks..."
	ruff check . || true
	@echo "✓ Lint complete"

format:
	@echo "Formatting code..."
	black . --line-length 100
	@echo "✓ Format complete"

build:
	@echo "Building distribution packages..."
	pip install --upgrade build
	python -m build
	@echo "✓ Build complete"
	@echo "Packages created in dist/"

publish: build
	@echo "Publishing to PyPI..."
	pip install --upgrade twine
	twine upload dist/*
	@echo "✓ Published to PyPI"

demo:
	@echo "Running demo script..."
	python3 demo.py

# Virtual environment setup
venv:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "✓ Virtual environment created"
	@echo "Activate with: source venv/bin/activate"

# Quick start for new users
quickstart: venv
	@echo "Setting up development environment..."
	@source venv/bin/activate && pip install -e ".[dev]"
	@echo ""
	@echo "✓ Quickstart complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate virtual environment: source venv/bin/activate"
	@echo "  2. Run tests: make test"
	@echo "  3. Use CLI: claude-agents --help"
