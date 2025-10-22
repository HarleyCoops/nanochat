#!/bin/bash
#
# Manual Deployment Guide for NanoChat HuggingFace Space
#
# Run this script on your LOCAL MACHINE (not in Claude Code environment)
# This script will guide you through deploying NanoChat to HuggingFace Spaces
#

set -e

cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   NanoChat HuggingFace Space - Manual Deployment Guide        ║
║                                                                ║
╔════════════════════════════════════════════════════════════════╗

This guide will help you deploy NanoChat to HuggingFace Spaces manually.

PREREQUISITES:
--------------
1. HuggingFace Account (sign up at https://huggingface.co)
2. Python 3.10+ installed
3. Git installed

EOF

read -p "Press Enter to continue..."
echo ""

# Step 1: Check Python
echo "Step 1/6: Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.10+ from https://python.org"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "✅ Found: $PYTHON_VERSION"
echo ""

# Step 2: Install dependencies
echo "Step 2/6: Installing HuggingFace Hub..."
pip install --quiet huggingface_hub
echo "✅ HuggingFace Hub installed"
echo ""

# Step 3: Login instructions
echo "Step 3/6: HuggingFace Authentication"
echo "----------------------------------------"
echo ""
echo "You need to authenticate with HuggingFace."
echo ""
echo "IMPORTANT: Create a token with WRITE permissions"
echo ""
echo "1. Go to: https://huggingface.co/settings/tokens"
echo "2. Click: 'Create new token'"
echo "3. Name: 'nanochat-deployment'"
echo "4. Type: Select 'Write' (NOT 'Read' or 'Fine-grained')"
echo "5. Click: 'Generate token'"
echo "6. Copy your token (starts with 'hf_')"
echo ""
read -p "Have you created a token with WRITE permissions? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please create a token and run this script again."
    exit 0
fi
echo ""

# Step 4: Login
echo "Step 4/6: Logging into HuggingFace..."
echo "----------------------------------------"
echo ""
echo "Running: huggingface-cli login"
echo ""
huggingface-cli login

if [ $? -ne 0 ]; then
    echo "❌ Login failed"
    exit 1
fi
echo "✅ Successfully logged in"
echo ""

# Step 5: Get Space name
echo "Step 5/6: Space Configuration"
echo "----------------------------------------"
echo ""
read -p "Enter your Space name (e.g., 'nanochat-demo'): " SPACE_NAME

if [ -z "$SPACE_NAME" ]; then
    SPACE_NAME="nanochat-inference"
    echo "Using default name: $SPACE_NAME"
fi
echo ""

# Step 6: Deploy
echo "Step 6/6: Deploying to HuggingFace Spaces..."
echo "----------------------------------------"
echo ""
echo "Space Name: $SPACE_NAME"
echo "Model: HarleyCooper/nanochat561"
echo "Hardware: cpu-basic (free)"
echo ""

python3 scripts/deploy_hf_space.py --space-name "$SPACE_NAME"

if [ $? -eq 0 ]; then
    cat << 'EOF'

╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║                   🎉 DEPLOYMENT SUCCESSFUL! 🎉                 ║
║                                                                ║
╔════════════════════════════════════════════════════════════════╗

WHAT'S NEXT:
------------

1. 🕐 WAIT 5-10 MINUTES
   Your Space is now building. This takes a few minutes.

2. 🔍 CHECK BUILD STATUS
   Visit: https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME/logs

3. 💬 START CHATTING
   Once built: https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME

4. ⚡ UPGRADE (OPTIONAL)
   For faster responses:
   - Go to Space Settings
   - Select 'Hardware'
   - Choose 't4-small' GPU (~$0.60/hr)

SHARING YOUR SPACE:
-------------------

Your Space is public by default. Share it with:
- Direct link: https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME
- Embed in website (see docs)
- API access (see INFERENCE_DEPLOYMENT.md)

TROUBLESHOOTING:
----------------

If the Space fails to build:
1. Check logs (add /logs to your Space URL)
2. See INFERENCE_DEPLOYMENT.md for solutions
3. Report issues at: https://github.com/HarleyCoops/nanochat561/issues

THANK YOU FOR USING NANOCHAT! 🚀

EOF
else
    echo ""
    echo "❌ Deployment failed"
    echo "Check the error messages above"
    echo "See INFERENCE_DEPLOYMENT.md for troubleshooting"
    exit 1
fi
