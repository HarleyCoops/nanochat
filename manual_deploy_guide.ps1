# NanoChat HuggingFace Space - Windows PowerShell Deployment Script
#
# Run this script on Windows to deploy NanoChat to HuggingFace Spaces
#
# Usage: .\manual_deploy_guide.ps1

$ErrorActionPreference = "Stop"

# Display banner
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "   NanoChat HuggingFace Space - Deployment Guide" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This guide will help you deploy NanoChat to HuggingFace Spaces." -ForegroundColor White
Write-Host ""
Write-Host "PREREQUISITES:" -ForegroundColor Yellow
Write-Host "--------------"
Write-Host "1. HuggingFace Account (sign up at https://huggingface.co)"
Write-Host "2. Python 3.10+ installed"
Write-Host "3. Git installed"
Write-Host ""
$null = Read-Host "Press Enter to continue"
Write-Host ""

# Step 1: Check Python
Write-Host "Step 1/6: Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python 3 is not installed" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://python.org" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Step 2: Install dependencies
Write-Host "Step 2/6: Installing HuggingFace Hub..." -ForegroundColor Cyan
try {
    python -m pip install --quiet huggingface_hub
    Write-Host "✓ HuggingFace Hub installed" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to install huggingface_hub" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Login instructions
Write-Host "Step 3/6: HuggingFace Authentication" -ForegroundColor Cyan
Write-Host "----------------------------------------"
Write-Host ""
Write-Host "You need to authenticate with HuggingFace." -ForegroundColor White
Write-Host ""
Write-Host "IMPORTANT: Create a token with WRITE permissions" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Go to: https://huggingface.co/settings/tokens" -ForegroundColor White
Write-Host "2. Click: 'Create new token'" -ForegroundColor White
Write-Host "3. Name: 'nanochat-deployment'" -ForegroundColor White
Write-Host "4. Type: Select 'Write' (NOT 'Read' or 'Fine-grained')" -ForegroundColor Yellow
Write-Host "5. Click: 'Generate token'" -ForegroundColor White
Write-Host "6. Copy your token (starts with 'hf_')" -ForegroundColor White
Write-Host ""
$response = Read-Host "Have you created a token with WRITE permissions? (y/n)"
if ($response -notmatch "^[Yy]") {
    Write-Host "Please create a token and run this script again." -ForegroundColor Yellow
    exit 0
}
Write-Host ""

# Step 4: Login
Write-Host "Step 4/6: Logging into HuggingFace..." -ForegroundColor Cyan
Write-Host "----------------------------------------"
Write-Host ""
Write-Host "Running: huggingface-cli login" -ForegroundColor White
Write-Host ""
try {
    huggingface-cli login
    if ($LASTEXITCODE -ne 0) {
        throw "Login failed"
    }
    Write-Host "✓ Successfully logged in" -ForegroundColor Green
} catch {
    Write-Host "✗ Login failed" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Get Space name
Write-Host "Step 5/6: Space Configuration" -ForegroundColor Cyan
Write-Host "----------------------------------------"
Write-Host ""
$spaceName = Read-Host "Enter your Space name (e.g., 'nanochat-demo')"

if ([string]::IsNullOrWhiteSpace($spaceName)) {
    $spaceName = "nanochat-inference"
    Write-Host "Using default name: $spaceName" -ForegroundColor Yellow
}
Write-Host ""

# Step 6: Deploy
Write-Host "Step 6/6: Deploying to HuggingFace Spaces..." -ForegroundColor Cyan
Write-Host "----------------------------------------"
Write-Host ""
Write-Host "Space Name: $spaceName" -ForegroundColor White
Write-Host "Model: HarleyCooper/nanochat561" -ForegroundColor White
Write-Host "Hardware: cpu-basic (free)" -ForegroundColor White
Write-Host ""

try {
    python scripts/deploy_hf_space.py --space-name $spaceName

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "================================================================" -ForegroundColor Green
        Write-Host "                🎉 DEPLOYMENT SUCCESSFUL! 🎉" -ForegroundColor Green
        Write-Host "================================================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "WHAT'S NEXT:" -ForegroundColor Cyan
        Write-Host "------------"
        Write-Host ""
        Write-Host "1. 🕐 WAIT 5-10 MINUTES" -ForegroundColor Yellow
        Write-Host "   Your Space is now building. This takes a few minutes."
        Write-Host ""
        Write-Host "2. 🔍 CHECK BUILD STATUS" -ForegroundColor Yellow
        Write-Host "   Visit: https://huggingface.co/spaces/YOUR-USERNAME/$spaceName/logs"
        Write-Host ""
        Write-Host "3. 💬 START CHATTING" -ForegroundColor Yellow
        Write-Host "   Once built: https://huggingface.co/spaces/YOUR-USERNAME/$spaceName"
        Write-Host ""
        Write-Host "4. ⚡ UPGRADE (OPTIONAL)" -ForegroundColor Yellow
        Write-Host "   For faster responses:"
        Write-Host "   - Go to Space Settings"
        Write-Host "   - Select 'Hardware'"
        Write-Host "   - Choose 't4-small' GPU (~`$0.60/hr)"
        Write-Host ""
        Write-Host "SHARING YOUR SPACE:" -ForegroundColor Cyan
        Write-Host "-------------------"
        Write-Host ""
        Write-Host "Your Space is public by default. Share it with:"
        Write-Host "- Direct link: https://huggingface.co/spaces/YOUR-USERNAME/$spaceName"
        Write-Host "- Embed in website (see docs)"
        Write-Host "- API access (see INFERENCE_DEPLOYMENT.md)"
        Write-Host ""
        Write-Host "THANK YOU FOR USING NANOCHAT! 🚀" -ForegroundColor Green
        Write-Host ""
    } else {
        throw "Deployment command returned non-zero exit code"
    }
} catch {
    Write-Host ""
    Write-Host "✗ Deployment failed" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Yellow
    Write-Host "See INFERENCE_DEPLOYMENT.md for troubleshooting" -ForegroundColor Yellow
    exit 1
}
