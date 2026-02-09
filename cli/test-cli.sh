#!/bin/bash
# Quick CLI test script
# Tests the claude-agents CLI without full installation

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLI_DIR="$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Claude Agents CLI - Quick Test Suite"
echo "=========================================="
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python3 --version
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "$CLI_DIR/venv" ]; then
    echo "📦 Creating virtual environment..."
    cd "$CLI_DIR"
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source "$CLI_DIR/venv/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q click rich PyYAML
echo "✓ Dependencies installed"
echo ""

# Install CLI in development mode
echo "🔧 Installing CLI in development mode..."
cd "$CLI_DIR"
pip install -q -e .
echo "✓ CLI installed"
echo ""

# Run tests
echo "=========================================="
echo "🧪 Running CLI Tests"
echo "=========================================="
echo ""

# Test 1: Help command
echo "Test 1: Help command"
echo "$ claude-agents --help"
claude-agents --help
echo ""
echo "✓ Test 1 passed"
echo ""

# Test 2: List command (dry run on repo)
echo "Test 2: List agents from repository"
echo "$ claude-agents list --format simple | head -10"
claude-agents list --format simple | head -10
echo ""
echo "✓ Test 2 passed"
echo ""

# Test 3: Search command
echo "Test 3: Search for 'API design'"
echo "$ claude-agents search 'API design' --limit 3"
claude-agents search 'API design' --limit 3
echo ""
echo "✓ Test 3 passed"
echo ""

# Test 4: Info command
echo "Test 4: Agent info"
echo "$ claude-agents info api-platform-engineer | head -20"
claude-agents info api-platform-engineer | head -20
echo ""
echo "✓ Test 4 passed"
echo ""

# Test 5: Validate command
echo "Test 5: Validate agents (repo)"
echo "$ claude-agents validate --agents-dir $REPO_ROOT/agents"
claude-agents validate --agents-dir "$REPO_ROOT/agents"
echo ""
echo "✓ Test 5 passed"
echo ""

# Test 6: Install with dry-run
echo "Test 6: Install with dry-run"
echo "$ claude-agents install --scope user --dry-run --tier 01-foundation"
claude-agents install --scope user --dry-run --tier 01-foundation
echo ""
echo "✓ Test 6 passed"
echo ""

# Summary
echo "=========================================="
echo "✅ All CLI Tests Passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review test output above"
echo "  2. Run individual commands manually"
echo "  3. Test actual installation (remove --dry-run)"
echo ""
echo "To use CLI interactively:"
echo "  $ source $CLI_DIR/venv/bin/activate"
echo "  $ claude-agents --help"
echo ""
