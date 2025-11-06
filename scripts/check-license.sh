#!/usr/bin/env bash
# License validation for Claude Agents Pro
# Checks for valid Pro or Enterprise license before installing premium agents

set -euo pipefail

LICENSE_FILE="${HOME}/.claude-agents-pro-license"
TIER="${1:-free}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

check_license() {
    local required_tier="$1"

    # Free tier always valid
    if [[ "$required_tier" == "free" ]]; then
        return 0
    fi

    # Check if license file exists
    if [[ ! -f "$LICENSE_FILE" ]]; then
        echo -e "${RED}❌ No license found${NC}"
        echo ""
        echo -e "${YELLOW}Premium agents require a license key.${NC}"
        echo ""
        echo "🔐 Get your license:"
        echo "   Pro ($12/month):        https://gumroad.com/l/claude-agents-pro"
        echo "   Enterprise ($45/month): https://gumroad.com/l/claude-agents-enterprise"
        echo ""
        echo "After purchase, run:"
        echo "   echo 'YOUR-LICENSE-KEY' > ~/.claude-agents-pro-license"
        exit 1
    fi

    # Read license from file
    LICENSE_KEY=$(cat "$LICENSE_FILE" | tr -d '[:space:]')

    # Basic validation (format check only for now)
    if [[ ! "$LICENSE_KEY" =~ ^CAP-(PRO|ENTERPRISE)-[A-Z0-9]{12}$ ]]; then
        echo -e "${RED}❌ Invalid license format${NC}"
        echo ""
        echo "Expected format: CAP-PRO-XXXXXXXXXXXX or CAP-ENTERPRISE-XXXXXXXXXXXX"
        echo ""
        echo "If you purchased a license, check your Gumroad receipt for the correct key."
        exit 1
    fi

    # Extract tier from license
    if [[ "$LICENSE_KEY" =~ ^CAP-PRO ]]; then
        LICENSE_TIER="pro"
    elif [[ "$LICENSE_KEY" =~ ^CAP-ENTERPRISE ]]; then
        LICENSE_TIER="enterprise"
    else
        echo -e "${RED}❌ Unknown license tier${NC}"
        exit 1
    fi

    # Check if license tier matches or exceeds required tier
    if [[ "$required_tier" == "pro" && ("$LICENSE_TIER" == "pro" || "$LICENSE_TIER" == "enterprise") ]]; then
        echo -e "${GREEN}✓ Pro license valid${NC}"
        return 0
    elif [[ "$required_tier" == "enterprise" && "$LICENSE_TIER" == "enterprise" ]]; then
        echo -e "${GREEN}✓ Enterprise license valid${NC}"
        return 0
    else
        echo -e "${RED}❌ Insufficient license tier${NC}"
        echo ""
        echo "This agent requires: $required_tier"
        echo "Your license tier: $LICENSE_TIER"
        echo ""
        if [[ "$required_tier" == "enterprise" && "$LICENSE_TIER" == "pro" ]]; then
            echo "Upgrade to Enterprise: https://gumroad.com/l/claude-agents-enterprise"
        fi
        exit 1
    fi
}

# Show license status
show_status() {
    echo -e "${BLUE}Claude Agents Pro - License Status${NC}"
    echo ""

    if [[ -f "$LICENSE_FILE" ]]; then
        LICENSE_KEY=$(cat "$LICENSE_FILE" | tr -d '[:space:]')
        if [[ "$LICENSE_KEY" =~ ^CAP-PRO ]]; then
            echo -e "${GREEN}✓ Pro License Active${NC}"
            echo "  Access to 23 agents (15 free + 8 pro)"
        elif [[ "$LICENSE_KEY" =~ ^CAP-ENTERPRISE ]]; then
            echo -e "${GREEN}✓ Enterprise License Active${NC}"
            echo "  Access to all 33 agents"
        else
            echo -e "${RED}❌ Invalid License${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Free Tier${NC}"
        echo "  Access to 15 free agents"
        echo ""
        echo "Upgrade for premium agents:"
        echo "  Pro ($12/mo): https://gumroad.com/l/claude-agents-pro"
        echo "  Enterprise ($45/mo): https://gumroad.com/l/claude-agents-enterprise"
    fi
    echo ""
}

# Main logic
case "${1:-status}" in
    status)
        show_status
        ;;
    free|pro|enterprise)
        check_license "$1"
        ;;
    *)
        echo "Usage: $0 {status|free|pro|enterprise}"
        exit 1
        ;;
esac
