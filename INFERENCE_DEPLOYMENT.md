# NanoChat Inference Deployment Guide

This guide covers deploying NanoChat for inference on HuggingFace Spaces.

## Overview

Your model is already on HuggingFace at **HarleyCooper/nanochat561**. This guide shows you how to set up a web interface for inference.

## Quick Deployment (Recommended)

### Prerequisites

1. **HuggingFace Account**: Sign up at https://huggingface.co
2. **HuggingFace Token**: Get one at https://huggingface.co/settings/tokens
3. **Python 3.10+**: Installed on your system

### One-Command Deployment

```bash
# 1. Install HuggingFace CLI (if not already installed)
pip install huggingface_hub

# 2. Login to HuggingFace
huggingface-cli login
# Paste your token when prompted

# 3. Deploy your Space
./deploy_inference.sh

# Or with a custom name
./deploy_inference.sh my-custom-space-name
```

That's it! Your inference Space will be live in 5-10 minutes at:
```
https://huggingface.co/spaces/YOUR-USERNAME/nanochat-inference
```

## What Gets Deployed

The deployment script creates a complete inference environment:

- **Gradio Web Interface**: Interactive chat UI
- **Model Integration**: Automatically loads HarleyCooper/nanochat561
- **Generation Parameters**: Adjustable temperature, top-k, etc.
- **Streaming Responses**: Real-time token-by-token generation
- **Mobile-Friendly**: Works on all devices

## Advanced Deployment Options

### Deploy to Organization

```bash
python scripts/deploy_hf_space.py \
  --space-name nanochat-demo \
  --org my-organization
```

### Private Space

```bash
python scripts/deploy_hf_space.py \
  --space-name nanochat-demo \
  --private
```

### With GPU Hardware

```bash
python scripts/deploy_hf_space.py \
  --space-name nanochat-demo \
  --hardware t4-small
```

Note: GPU hardware requires a HuggingFace Pro account ($9/month).

### Use Different Model

If you've trained your own model:

```bash
python scripts/deploy_hf_space.py \
  --space-name my-nanochat \
  --model-id your-username/your-nanochat-model
```

## Hardware Options & Pricing

| Hardware | Cost | Best For | Speed |
|----------|------|----------|-------|
| cpu-basic | Free | Testing, demos | Slow (5-10 sec/response) |
| cpu-upgrade | $0.03/hr | Light usage | Medium (2-5 sec/response) |
| t4-small | $0.60/hr | Production | Fast (<1 sec/response) |
| t4-medium | $1.20/hr | High traffic | Fast (<1 sec/response) |
| a10g-small | $3.15/hr | Heavy workloads | Very Fast |

**Tip**: Start with `cpu-basic` (free) for testing, then upgrade to GPU for production.

## Post-Deployment

### Access Your Space

Once deployed, your Space will be at:
```
https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME
```

### Monitor Build Progress

Check build logs:
```
https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME/logs
```

### Upgrade Hardware

1. Go to Space Settings: `https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME/settings`
2. Scroll to "Hardware"
3. Select desired tier (requires Pro account for GPU)
4. Click "Save"
5. Space will restart with new hardware

### Pause/Resume Space

To save costs when not in use:

1. Go to Space Settings
2. Click "Pause Space"
3. Click "Resume" when needed

## Embedding Your Space

Add your Space to any website:

```html
<iframe
  src="https://YOUR-USERNAME-SPACE-NAME.hf.space"
  frameborder="0"
  width="850"
  height="600"
></iframe>
```

## API Access

Use your Space as an API endpoint:

```python
from gradio_client import Client

client = Client("YOUR-USERNAME/SPACE-NAME")
response = client.predict(
    "Hello, how are you?",  # message
    [],  # history
    256,  # max_new_tokens
    0.7,  # temperature
    0.95,  # top_p
    50,  # top_k
    1.05,  # repetition_penalty
    api_name="/chat"
)
print(response)
```

## Alternative: Direct Model Usage

You can also use the model directly without deploying a Space:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "HarleyCooper/nanochat561",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "HarleyCooper/nanochat561",
    trust_remote_code=True
)

# Generate
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Local Inference Server

Run inference locally with the web UI:

```bash
# Using your local checkpoint
python scripts/chat_web.py --source sft --port 8000

# Then open http://localhost:8000
```

## Troubleshooting

### Space Build Fails

**Check the logs** at `https://huggingface.co/spaces/YOUR-USERNAME/SPACE-NAME/logs`

Common issues:

1. **Out of memory**: Upgrade to larger CPU or use GPU
2. **Import errors**: Ensure all files are uploaded
3. **Model not found**: Check model ID in `app.py`

### Slow Responses

1. **Upgrade to GPU**: T4 Small recommended
2. **Reduce max tokens**: Lower the default in generation settings
3. **Use cached results**: For common queries

### Space Doesn't Start

1. Check build logs for errors
2. Verify all requirements are in `requirements.txt`
3. Test locally first with `cd deploy/hf_space && python app.py`

## Deployment Script Reference

### Quick Script (`deploy_inference.sh`)

Simple wrapper that:
- Checks dependencies
- Verifies HF login
- Deploys with sensible defaults

```bash
./deploy_inference.sh [space-name]
```

### Full Script (`scripts/deploy_hf_space.py`)

Full-featured Python script with all options:

```bash
python scripts/deploy_hf_space.py --help
```

**Options:**
- `--space-name NAME` (required): Space name
- `--model-id MODEL`: Model to use (default: HarleyCooper/nanochat561)
- `--org ORG`: Deploy to organization
- `--private`: Make Space private
- `--hardware TYPE`: Hardware tier

## Cost Estimates

### Free Tier (CPU Basic)
- **Cost**: $0
- **Limits**: Rate limited
- **Best for**: Personal demos, testing

### Paid Tier (T4 Small GPU)
- **Cost**: ~$0.60/hour
- **Monthly**: ~$432 if running 24/7
- **With pausing**: ~$50-100/month (paused when not in use)
- **Best for**: Production deployments

**Cost-saving tips:**
1. Pause Space when not in use
2. Use CPU for low-traffic periods
3. Implement caching for common queries
4. Use HuggingFace Inference Endpoints (serverless, pay-per-use)

## Next Steps

1. ✅ Deploy your Space: `./deploy_inference.sh`
2. ⏱️ Wait 5-10 minutes for build
3. 🎉 Start chatting with your model
4. 📊 Monitor usage in Space Settings
5. 🚀 Upgrade to GPU if needed

## Support & Resources

- **HuggingFace Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Documentation**: https://gradio.app/docs
- **Model Repository**: https://huggingface.co/HarleyCooper/nanochat561
- **Issues**: https://github.com/HarleyCoops/nanochat561/issues

## Files Overview

```
deploy/hf_space/
├── app.py                      # Gradio application
├── requirements.txt            # Python dependencies
├── README.md                   # Space description (shows on HF)
├── DEPLOYMENT.md              # This guide
├── configuration_nanochat.py  # Model configuration
└── modeling_nanochat.py       # Model implementation

scripts/
└── deploy_hf_space.py         # Automated deployment script

deploy_inference.sh            # Quick deployment wrapper
```

## License

MIT - See LICENSE file for details.

---

**Ready to deploy?** Run: `./deploy_inference.sh`
