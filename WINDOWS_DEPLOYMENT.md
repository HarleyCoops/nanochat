# NanoChat Deployment on Windows

Quick guide for deploying NanoChat to HuggingFace Spaces on Windows.

## 🚀 Quick Start (3 Methods)

### Method 1: PowerShell Script (Recommended)

Open PowerShell in the nanochat directory:

```powershell
# Run the deployment script
.\manual_deploy_guide.ps1
```

If you get an execution policy error:
```powershell
# Allow script execution (run PowerShell as Administrator)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then run the script
.\manual_deploy_guide.ps1
```

### Method 2: Batch File (Simplest)

Open Command Prompt in the nanochat directory:

```cmd
deploy_windows.bat
```

Or with a custom space name:
```cmd
deploy_windows.bat my-custom-name
```

### Method 3: Manual Python Commands

Open Command Prompt or PowerShell:

```cmd
# 1. Install HuggingFace Hub
pip install huggingface_hub

# 2. Login to HuggingFace
huggingface-cli login
REM Paste your token when prompted

# 3. Deploy
python scripts/deploy_hf_space.py --space-name nanochat-demo
```

## 🔑 Getting Your HuggingFace Token

1. **Go to**: https://huggingface.co/settings/tokens
2. **Click**: "Create new token"
3. **Name**: `nanochat-deployment`
4. **Type**: Select **"Write"** (NOT "Read" or "Fine-grained")
5. **Click**: "Generate token"
6. **Copy**: Your token (starts with `hf_`)

**Important**: The token MUST have "Write" permissions to create Spaces.

## ⚠️ Common Windows Issues

### Issue 1: PowerShell Execution Policy Error

**Error**: `cannot be loaded because running scripts is disabled`

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue 2: Python Not Found

**Error**: `'python' is not recognized`

**Solution**:
1. Install Python from https://python.org
2. During installation, check "Add Python to PATH"
3. Restart Command Prompt/PowerShell

### Issue 3: Git Not Found

**Error**: `'git' is not recognized`

**Solution**:
1. Install Git from https://git-scm.com/download/win
2. Restart Command Prompt/PowerShell

### Issue 4: huggingface-cli Not Found

**Error**: `'huggingface-cli' is not recognized`

**Solution**:
```cmd
# Reinstall with Scripts directory in PATH
pip install --user huggingface_hub

# Or use full path
python -m huggingface_hub.commands.huggingface_cli login
```

### Issue 5: Display Error (Chrome/Electron)

**Error**: `ERROR:device_event_log_impl.cc...`

**Solution**: This is a harmless warning from Chrome/Electron. You can ignore it.

## 📁 Windows File Paths

When working with paths on Windows:

```powershell
# PowerShell - use forward slashes or escape backslashes
cd C:/Users/chris/nanochat
# or
cd C:\Users\chris\nanochat

# To run scripts
.\deploy_windows.bat
.\manual_deploy_guide.ps1
python scripts\deploy_hf_space.py --space-name demo
```

## 🎯 Step-by-Step Deployment

### 1. Open PowerShell

- Press `Win + X`
- Select "Windows PowerShell" or "Terminal"
- Navigate to your nanochat directory:
  ```powershell
  cd C:\Users\YourUsername\nanochat
  ```

### 2. Run Deployment Script

**Option A** - PowerShell (Interactive):
```powershell
.\manual_deploy_guide.ps1
```

**Option B** - Batch (Quick):
```cmd
deploy_windows.bat
```

**Option C** - Python (Advanced):
```powershell
pip install huggingface_hub
huggingface-cli login
python scripts/deploy_hf_space.py --space-name my-demo
```

### 3. Follow Prompts

The script will:
1. ✅ Check Python installation
2. ✅ Install dependencies
3. ✅ Prompt for HuggingFace login
4. ✅ Ask for Space name
5. ✅ Deploy your Space
6. ✅ Provide the Space URL

### 4. Wait for Build

- ⏱️ Building takes 5-10 minutes
- 🔍 Check logs: `https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME/logs`
- 💬 Chat: `https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME`

## 🔧 Advanced Options

### Deploy with GPU

```cmd
python scripts/deploy_hf_space.py --space-name my-demo --hardware t4-small
```

### Deploy to Organization

```cmd
python scripts/deploy_hf_space.py --space-name demo --org my-org
```

### Private Space

```cmd
python scripts/deploy_hf_space.py --space-name demo --private
```

### Use Different Model

```cmd
python scripts/deploy_hf_space.py --space-name demo --model-id your-username/your-model
```

## 📊 What You'll Get

After deployment:

- **Chat Interface**: Beautiful Gradio UI
- **Public URL**: `https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME`
- **Free Hosting**: CPU-based (or upgrade to GPU)
- **API Access**: Use your Space as an API
- **Embeddable**: Add to your website

## 💰 Costs

| Hardware | Speed | Cost |
|----------|-------|------|
| CPU Basic | 5-10s | Free |
| CPU Upgrade | 2-5s | $0.03/hr |
| T4 Small (GPU) | <1s | $0.60/hr |

## 📚 Additional Resources

- **Full Guide**: See `INFERENCE_DEPLOYMENT.md`
- **Troubleshooting**: See `DEPLOYMENT_STATUS.md`
- **HuggingFace Docs**: https://huggingface.co/docs/hub/spaces

## 🆘 Getting Help

If you encounter issues:

1. **Check this guide** for common Windows issues
2. **See logs**: Add `/logs` to your Space URL
3. **Read**: `INFERENCE_DEPLOYMENT.md` for detailed troubleshooting
4. **Report**: https://github.com/HarleyCoops/nanochat561/issues

## ✅ Quick Checklist

Before deploying:
- [ ] Python 3.10+ installed
- [ ] Git installed (optional, but recommended)
- [ ] HuggingFace account created
- [ ] HuggingFace token with **Write** permissions
- [ ] In the nanochat directory

To deploy:
- [ ] Run `.\manual_deploy_guide.ps1` or `deploy_windows.bat`
- [ ] Paste your token when prompted
- [ ] Choose a Space name
- [ ] Wait 5-10 minutes for build
- [ ] Visit your Space URL!

---

## Summary

**Easiest method**:
```powershell
.\manual_deploy_guide.ps1
```

**Quickest method**:
```cmd
deploy_windows.bat
```

**Most control**:
```cmd
python scripts/deploy_hf_space.py --space-name demo --hardware t4-small
```

All methods will give you a live chat interface in ~10 minutes! 🚀
