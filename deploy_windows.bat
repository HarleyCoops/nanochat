@echo off
REM NanoChat HuggingFace Space - Quick Windows Deployment
REM Usage: deploy_windows.bat [space-name]

echo.
echo ================================================================
echo    NanoChat HuggingFace Space - Quick Deploy
echo ================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Checking Python... OK
echo.

REM Install dependencies
echo [2/4] Installing HuggingFace Hub...
python -m pip install -q huggingface_hub
if errorlevel 1 (
    echo [ERROR] Failed to install huggingface_hub
    pause
    exit /b 1
)
echo       Installed successfully
echo.

REM Login instructions
echo [3/4] HuggingFace Authentication
echo ----------------------------------------------------------------
echo.
echo IMPORTANT: You need a HuggingFace token with WRITE permissions
echo.
echo To create a token:
echo   1. Go to: https://huggingface.co/settings/tokens
echo   2. Click: 'Create new token'
echo   3. Type: Select 'Write' (NOT 'Read')
echo   4. Copy your token
echo.
echo Now running: huggingface-cli login
echo Paste your token when prompted...
echo.

huggingface-cli login
if errorlevel 1 (
    echo [ERROR] Login failed
    pause
    exit /b 1
)

echo.
echo Login successful!
echo.

REM Deploy
echo [4/4] Deploying Space...
echo ----------------------------------------------------------------
echo.

set SPACE_NAME=%1
if "%SPACE_NAME%"=="" set SPACE_NAME=nanochat-inference

echo Space Name: %SPACE_NAME%
echo Model: HarleyCooper/nanochat561
echo Hardware: cpu-basic (free)
echo.

python scripts/deploy_hf_space.py --space-name %SPACE_NAME%

if errorlevel 1 (
    echo.
    echo [ERROR] Deployment failed
    echo See INFERENCE_DEPLOYMENT.md for troubleshooting
    pause
    exit /b 1
)

echo.
echo ================================================================
echo                 DEPLOYMENT SUCCESSFUL!
echo ================================================================
echo.
echo Your Space is building now (takes 5-10 minutes)
echo.
echo View your Space at:
echo https://huggingface.co/spaces/YOUR-USERNAME/%SPACE_NAME%
echo.
echo Check build logs at:
echo https://huggingface.co/spaces/YOUR-USERNAME/%SPACE_NAME%/logs
echo.
echo ================================================================
echo.
pause
