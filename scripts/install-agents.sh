#!/bin/bash

# Claude Code Agent Installation Script
# Installs ubehera agents to user or project Claude Code configuration
# Usage: ./install-agents.sh [--user|--project] [--select agent1,agent2] [--dry-run]

set -eo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$(dirname "$SCRIPT_DIR")/agents"
USER_CLAUDE_DIR="$HOME/.claude/agents"
PROJECT_CLAUDE_DIR=".claude/agents"

# Default values
INSTALL_MODE="user"
DRY_RUN=false
SELECTED_AGENTS=""
BACKUP=true
VERBOSE=false
NON_INTERACTIVE=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Claude Code Agent Installation Script

Usage: $0 [OPTIONS]

OPTIONS:
    -u, --user              Install agents for current user (default)
    -p, --project           Install agents for current project only
    -s, --select AGENTS     Install specific agents (comma-separated)
    -l, --list              List available agents
    -d, --dry-run           Show what would be installed without doing it
    -b, --no-backup         Skip backup of existing agents
    -v, --verbose           Enable verbose output
    -n, --non-interactive   Do not prompt when project checks fail (continue)
    -h, --help              Show this help message

EXAMPLES:
    # Install all agents for current user
    $0

    # Install specific agents for project
    $0 --project --select api-platform-engineer,security-architect

    # List available agents
    $0 --list

    # Dry run to see what would be installed
    $0 --dry-run

EOF
}

# Function to list available agents
list_agents() {
    print_info "Available agents in $AGENTS_DIR:"
    echo ""
    while IFS= read -r -d '' agent; do
        basename=$(basename "$agent")
        name="${basename%.md}"
        rel_path=${agent#"$AGENTS_DIR/"}

        # Extract description from frontmatter
        description=$(grep -m 1 "^description:" "$agent" 2>/dev/null | sed 's/description: //' | cut -c1-60)

        if [[ -n "$description" ]]; then
            printf "  ${GREEN}%-35s${NC} (%s) %s...\n" "$name" "$rel_path" "$description"
        else
            printf "  ${GREEN}%-35s${NC} (%s)\n" "$name" "$rel_path"
        fi
    done < <(find "$AGENTS_DIR" -mindepth 1 -type f -name '*.md' \
        ! -name 'README.md' ! -name 'TESTING.md' ! -name 'AGENT_CHECKLIST.md' -print0 | sort -z)
    echo ""
}

# Resolve an agent identifier (name or relative path) to a file path
resolve_agent_path() {
    local identifier="$1"

    # Accept relative paths (with or without .md)
    if [[ -f "$AGENTS_DIR/${identifier}" ]]; then
        printf '%s\n' "$AGENTS_DIR/${identifier}"
        return 0
    fi
    if [[ -f "$AGENTS_DIR/${identifier}.md" ]]; then
        printf '%s\n' "$AGENTS_DIR/${identifier}.md"
        return 0
    fi

    # Fallback: search by file name within tiered directories
    local match
    match=$(find "$AGENTS_DIR" -mindepth 1 -maxdepth 3 -type f -name "${identifier}.md" | head -n 1)
    if [[ -z "$match" ]]; then
        return 1
    fi

    printf '%s\n' "$match"
    return 0
}

# Function to validate agent exists
validate_agent() {
    local agent="$1"
    if ! resolve_agent_path "$agent" > /dev/null; then
        print_error "Agent '$agent' not found in $AGENTS_DIR"
        return 1
    fi
    return 0
}

# License/tier checking removed - all agents are now freely available for personal use

# Function to backup existing agents
backup_agents() {
    local target_dir="$1"
    
    if [[ -d "$target_dir" ]] && [[ "$BACKUP" == true ]]; then
        local backup_dir="${target_dir}.backup.$(date +%Y%m%d_%H%M%S)"
        print_info "Backing up existing agents to $backup_dir"
        
        if [[ "$DRY_RUN" == false ]]; then
            cp -r "$target_dir" "$backup_dir"
            print_success "Backup created at $backup_dir"
        else
            print_info "[DRY RUN] Would create backup at $backup_dir"
        fi
    fi
}

# Function to install agents
install_agents() {
    local target_dir="$1"
    local agents_to_install=()
    
    # Determine which agents to install
    if [[ -n "$SELECTED_AGENTS" ]]; then
        IFS=',' read -ra agents_to_install <<< "$SELECTED_AGENTS"
        local resolved_agents=()
        for agent in "${agents_to_install[@]}"; do
            agent=$(echo "$agent" | xargs)
            if resolved_path=$(resolve_agent_path "$agent"); then
                resolved_agents+=("$resolved_path")
            else
                return 1
            fi
        done
        agents_to_install=("${resolved_agents[@]}")
    else
        # Install all valid agent files (with YAML frontmatter + name)
        while IFS= read -r -d '' file; do
            [[ -f "$file" ]] || continue
            basename=$(basename "$file")
            case "$basename" in
                README.md|TESTING.md|AGENT_CHECKLIST.md)
                    continue
                    ;;
            esac
            if head -n 1 "$file" | grep -q '^---$' && head -n 20 "$file" | grep -q '^name:'; then
                agents_to_install+=("$file")
            fi
        done < <(find "$AGENTS_DIR" -type f -name '*.md' -print0)
    fi

    # Validate all agents before installation
    print_info "Validating agents..."
    for source_file in "${agents_to_install[@]}"; do
        local agent_base=$(basename "${source_file%.md}")
        if ! resolve_agent_path "$agent_base" > /dev/null; then
            print_error "Agent '$agent_base' not found in $AGENTS_DIR"
            return 1
        fi
    done
    
    # Create target directory if needed
    if [[ ! -d "$target_dir" ]]; then
        print_info "Creating directory $target_dir"
        if [[ "$DRY_RUN" == false ]]; then
            mkdir -p "$target_dir"
        fi
    fi
    
    # Backup existing agents
    backup_agents "$target_dir"
    
    # Install agents
    print_info "Installing ${#agents_to_install[@]} agents to $target_dir"
    echo ""

    local installed_count=0
    local skipped_count=0

    for source_file in "${agents_to_install[@]}"; do
        local agent=$(basename "${source_file%.md}")
        local target_file="$target_dir/${agent}.md"

        if [[ "$VERBOSE" == true ]]; then
            print_info "Processing $agent..."
        fi

        if [[ -f "$target_file" ]]; then
            # Check if files are different
            if ! cmp -s "$source_file" "$target_file"; then
                if [[ "$DRY_RUN" == false ]]; then
                    cp "$source_file" "$target_file"
                    print_success "  ✓ Updated: $agent"
                    ((installed_count++))
                else
                    print_info "  [DRY RUN] Would update: $agent"
                fi
            else
                if [[ "$VERBOSE" == true ]]; then
                    print_info "  - Skipped: $agent (already up to date)"
                fi
                ((skipped_count++))
            fi
        else
            if [[ "$DRY_RUN" == false ]]; then
                cp "$source_file" "$target_file"
                print_success "  ✓ Installed: $agent"
                ((installed_count++))
            else
                print_info "  [DRY RUN] Would install: $agent"
            fi
        fi
    done

    echo ""
    if [[ "$DRY_RUN" == false ]]; then
        print_success "Installation complete!"
        print_info "  Installed/Updated: $installed_count agents"
        if [[ "$skipped_count" -gt 0 ]]; then
            print_info "  Already up to date: $skipped_count agents"
        fi
    else
        print_info "[DRY RUN] Would install/update $installed_count agents"
    fi
}

# Function to verify installation
verify_installation() {
    local target_dir="$1"
    
    print_info "Verifying installation..."
    
    local verified_count=0
    for agent in "$target_dir"/*.md; do
        if [[ -f "$agent" ]]; then
            # Check if agent has valid frontmatter
            if grep -q "^name:" "$agent" && grep -q "^description:" "$agent"; then
                ((verified_count++))
            else
                print_warning "Invalid agent format: $(basename "$agent")"
            fi
        fi
    done
    
    print_success "Verified $verified_count agents in $target_dir"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--user)
            INSTALL_MODE="user"
            shift
            ;;
        -p|--project)
            INSTALL_MODE="project"
            shift
            ;;
        -s|--select)
            SELECTED_AGENTS="$2"
            shift 2
            ;;
        -l|--list)
            list_agents
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -b|--no-backup)
            BACKUP=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -n|--non-interactive)
            NON_INTERACTIVE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "Claude Code Agent Installation Script"
    print_info "=====================================\n"
    
    # Check if agents directory exists
    if [[ ! -d "$AGENTS_DIR" ]]; then
        print_error "Agents directory not found: $AGENTS_DIR"
        exit 1
    fi
    
    # Determine target directory
    local target_dir=""
    if [[ "$INSTALL_MODE" == "user" ]]; then
        target_dir="$USER_CLAUDE_DIR"
        print_info "Installation mode: User (global)"
    else
        target_dir="$PROJECT_CLAUDE_DIR"
        print_info "Installation mode: Project (local)"
        
        # Check if we're in a git repository or project directory
        if [[ ! -d ".git" ]] && [[ ! -f "package.json" ]] && [[ ! -f "Cargo.toml" ]] && [[ ! -f "go.mod" ]]; then
            print_warning "Current directory doesn't appear to be a project root"
            if [[ "$NON_INTERACTIVE" == true ]]; then
                print_info "Non-interactive mode: continuing despite project check"
            else
                read -p "Continue anyway? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    print_info "Installation cancelled"
                    exit 0
                fi
            fi
        fi
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        print_warning "DRY RUN MODE - No changes will be made\n"
    fi
    
    # Install agents
    install_agents "$target_dir"
    
    # Verify installation
    if [[ "$DRY_RUN" == false ]]; then
        echo ""
        verify_installation "$target_dir"
    fi
    
    # Post-installation instructions
    echo ""
    print_info "Next steps:"
    echo "  1. Restart Claude Code to load the new agents"
    echo "  2. Test agent invocation with relevant prompts"
    echo "  3. Check ~/.claude/agents/ or .claude/agents/ for installed agents"
    
    if [[ "$BACKUP" == true ]] && [[ "$DRY_RUN" == false ]]; then
        echo ""
        print_info "Tip: To restore from backup, use:"
        echo "  rm -rf $target_dir && mv ${target_dir}.backup.* $target_dir"
    fi
}

# Run main function
main
