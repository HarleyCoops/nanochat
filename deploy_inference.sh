#!/bin/bash
#
# Quick deployment script for NanoChat inference on HuggingFace Spaces
#
# Usage:
#   ./deploy_inference.sh                    # Deploy with default settings
#   ./deploy_inference.sh my-space-name      # Deploy with custom name
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   NanoChat Inference Deployment Script${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if huggingface_hub is installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip install -q huggingface_hub
    echo -e "${GREEN}✓ Installed huggingface_hub${NC}"
else
    echo -e "${GREEN}✓ huggingface_hub is installed${NC}"
fi

# Check if user is logged in
echo ""
echo -e "${YELLOW}Checking HuggingFace login...${NC}"
if ! python3 -c "from huggingface_hub import whoami; whoami()" 2>/dev/null; then
    echo -e "${RED}✗ Not logged into HuggingFace${NC}"
    echo ""
    echo -e "${YELLOW}Please login to HuggingFace:${NC}"
    echo -e "1. Get your token from: ${BLUE}https://huggingface.co/settings/tokens${NC}"
    echo -e "2. Run: ${BLUE}huggingface-cli login${NC}"
    echo ""
    exit 1
fi
echo -e "${GREEN}✓ Logged into HuggingFace${NC}"

# Get space name from argument or use default
SPACE_NAME="${1:-nanochat-inference}"

echo ""
echo -e "${YELLOW}Deployment Configuration:${NC}"
echo -e "  Space Name: ${GREEN}${SPACE_NAME}${NC}"
echo -e "  Model: ${GREEN}HarleyCooper/nanochat561${NC}"
echo -e "  Hardware: ${GREEN}cpu-basic (free)${NC}"
echo ""

read -p "Continue with deployment? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

# Run the deployment script
echo ""
echo -e "${BLUE}Starting deployment...${NC}"
python3 scripts/deploy_hf_space.py --space-name "${SPACE_NAME}"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}   Deployment Successful!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo -e "1. Wait 5-10 minutes for the Space to build"
    echo -e "2. Check logs at: ${BLUE}https://huggingface.co/spaces/YOUR-USERNAME/${SPACE_NAME}/logs${NC}"
    echo -e "3. Once built, chat at: ${BLUE}https://huggingface.co/spaces/YOUR-USERNAME/${SPACE_NAME}${NC}"
    echo ""
    echo -e "${YELLOW}Optional: Upgrade hardware for faster inference${NC}"
    echo -e "  Go to Space Settings and select GPU (t4-small recommended)"
    echo ""
else
    echo ""
    echo -e "${RED}Deployment failed. Check errors above.${NC}"
    exit 1
fi
