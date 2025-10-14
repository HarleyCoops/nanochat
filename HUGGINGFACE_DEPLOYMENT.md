# HuggingFace Deployment Guide

This guide explains how to convert your nanochat model to HuggingFace format and deploy it on HuggingFace Inference Endpoints.

## Why Deploy on HuggingFace?

After training your model on Hyperbolic Labs, you can deploy it on HuggingFace Inference Endpoints for:

- **Scalable Inference**: Auto-scaling serverless inference
- **Cost Efficiency**: Pay only for what you use
- **Easy Integration**: REST API compatible with OpenAI format
- **Built-in UI**: Test your model in HF model page
- **Model Versioning**: Track and manage different versions

## Overview

The conversion process:
1. Train model on Hyperbolic Labs (or locally)
2. Export to HuggingFace format
3. Upload to HuggingFace Hub
4. Deploy on Inference Endpoints

> **Need a Space instead?** A ready-to-run Gradio starter now lives in
> `deploy/hf_space/`. Drop `app.py` and `requirements.txt` into a new Hugging
> Face Space (with the model assets) to spin up a hosted chat UI.

## Step 1: Export Model to HuggingFace Format

After training completes, export your model:

```bash
# Basic export
python scripts/export_to_huggingface.py \
  --source sft \
  --output-dir ./hf_model

# With specific model tag
python scripts/export_to_huggingface.py \
  --source sft \
  --model-tag d20 \
  --output-dir ./hf_model
```

This creates a directory with HuggingFace-compatible files:

```
hf_model/
├── config.json                    # Model configuration
├── pytorch_model.bin              # Model weights
├── vocab.json                     # Tokenizer vocabulary
├── tokenizer_config.json          # Tokenizer settings
├── special_tokens_map.json        # Special tokens
├── modeling_nanochat_gpt.py       # Custom model class
└── README.md                      # Model card
```

## Step 2: Install HuggingFace CLI

```bash
pip install huggingface_hub

# Login to your HuggingFace account
huggingface-cli login
# Paste your HF token when prompted (get it from https://huggingface.co/settings/tokens)
```

## Step 3: Upload to HuggingFace Hub

```bash
# Create a new model repository and upload
huggingface-cli upload your-username/nanochat-d20 ./hf_model

# Or upload to existing repository
huggingface-cli upload your-username/nanochat-d20 ./hf_model --repo-type model
```

Replace `your-username` with your HuggingFace username and `nanochat-d20` with your desired model name.

## Step 4: Deploy on HuggingFace Inference Endpoints

### Option A: Via Web UI (Recommended)

1. Go to your model page: `https://huggingface.co/your-username/nanochat-d20`
2. Click **"Deploy"** → **"Inference Endpoints"**
3. Configure your endpoint:
   - **Endpoint name**: `nanochat-production`
   - **Cloud**: AWS, Azure, or GCP
   - **Region**: Choose closest to your users
   - **Instance type**: CPU (for testing) or GPU (for production)
   - **Min/Max replicas**: 1/3 (auto-scaling)
   - **Framework**: PyTorch
   - **Task**: Text Generation

4. Click **"Create Endpoint"**
5. Wait 5-10 minutes for deployment

### Option B: Via API

```python
from huggingface_hub import create_inference_endpoint

endpoint = create_inference_endpoint(
    name="nanochat-production",
    repository="your-username/nanochat-d20",
    framework="pytorch",
    task="text-generation",
    accelerator="cpu",  # or "gpu"
    instance_size="small",  # small, medium, large, xlarge
    instance_type="c6i",  # AWS instance type
    region="us-east-1",
    vendor="aws",
    min_replica=1,
    max_replica=3,
    type="protected",  # or "public"
)

print(f"Endpoint URL: {endpoint.url}")
```

## Step 5: Test Your Deployed Model

### Using Python

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="your-username/nanochat-d20")

# Generate text
response = client.text_generation(
    "Hello, how are you?",
    max_new_tokens=100,
    temperature=0.7,
    top_k=50
)
print(response)
```

### Using cURL

```bash
curl https://api-inference.huggingface.co/models/your-username/nanochat-d20 \
  -X POST \
  -H "Authorization: Bearer YOUR_HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "Hello, how are you?",
    "parameters": {
      "max_new_tokens": 100,
      "temperature": 0.7,
      "top_k": 50
    }
  }'
```

### Using JavaScript

```javascript
async function query(data) {
  const response = await fetch(
    "https://api-inference.huggingface.co/models/your-username/nanochat-d20",
    {
      headers: { Authorization: "Bearer YOUR_HF_TOKEN" },
      method: "POST",
      body: JSON.stringify(data),
    }
  );
  return await response.json();
}

query({
  inputs: "Hello, how are you?",
  parameters: { max_new_tokens: 100, temperature: 0.7 }
}).then((response) => {
  console.log(response);
});
```

## Complete Workflow Example

### Training to Deployment Pipeline

```bash
# 1. Train on Hyperbolic Labs
python hyperbolic/deploy.py --model-size medium --auto-launch

# 2. SSH into instance when training completes
ssh user@hyperbolic-instance

# 3. Export model to HF format
python scripts/export_to_huggingface.py \
  --source sft \
  --model-tag d20 \
  --output-dir ./hf_model

# 4. Download exported model to local
# From local machine:
scp -r user@hyperbolic-instance:/path/to/hf_model ./

# 5. Terminate training instance
python hyperbolic/deploy.py --terminate <instance-id>

# 6. Upload to HuggingFace
huggingface-cli login
huggingface-cli upload your-username/nanochat-d20 ./hf_model

# 7. Deploy via HF web UI or API
# Visit: https://huggingface.co/your-username/nanochat-d20
```

## Cost Comparison

### Training (Hyperbolic Labs)
- **Medium model**: 4 GPUs × $2-5/hr × 5 hours = **$40-100**
- **One-time cost**

### Inference (HuggingFace)
| Instance Type | vCPU | RAM | GPU | Cost/Hour | Best For |
|---------------|------|-----|-----|-----------|----------|
| CPU small | 1 | 2GB | - | $0.06 | Testing |
| CPU medium | 4 | 8GB | - | $0.24 | Low traffic |
| GPU small | 4 | 16GB | T4 | $0.60 | Moderate traffic |
| GPU medium | 8 | 32GB | A10G | $1.30 | High traffic |

**Auto-scaling**: Pay only for actual usage
- Minimum: 0 replicas (scale to zero when idle)
- Maximum: Set based on expected load

## Monitoring and Management

### View Endpoint Status

```python
from huggingface_hub import get_inference_endpoint

endpoint = get_inference_endpoint("nanochat-production", namespace="your-username")

print(f"Status: {endpoint.status}")
print(f"URL: {endpoint.url}")
print(f"Replicas: {endpoint.replicas}")
```

### Update Endpoint

```python
endpoint.update(
    min_replica=0,  # Scale to zero when idle
    max_replica=5,  # Allow up to 5 replicas under load
    accelerator="gpu",  # Upgrade to GPU
)
```

### Pause/Resume Endpoint

```python
# Pause (stop billing)
endpoint.pause()

# Resume
endpoint.resume()
```

### Delete Endpoint

```python
endpoint.delete()
```

## Troubleshooting

### Model Won't Load

**Issue**: `trust_remote_code` error

**Solution**: The model uses custom code. Users must set `trust_remote_code=True`:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-username/nanochat-d20",
    trust_remote_code=True  # Required!
)
```

### Generation is Slow

**Solutions**:
1. Upgrade to GPU instance
2. Enable auto-scaling for multiple replicas
3. Use batch inference for multiple requests

### Out of Memory

**Solutions**:
1. Reduce `max_new_tokens`
2. Upgrade to larger instance
3. Use CPU for very small models

## Alternative: ONNX Export (Advanced)

For even faster CPU inference, convert to ONNX:

```bash
# Install optimum
pip install optimum[exporters]

# Export to ONNX
python -m optimum.exporters.onnx \
  --model ./hf_model \
  --task text-generation \
  ./hf_model_onnx
```

## Best Practices

1. **Version Control**: Use semantic versioning for model updates
2. **Testing**: Always test on CPU small instance first
3. **Auto-scaling**: Start with min=0, max=3 for cost efficiency  
4. **Monitoring**: Set up alerts for high latency/errors
5. **Model Cards**: Keep README.md updated with model capabilities
6. **Rate Limiting**: Implement on client side to control costs
7. **Caching**: Cache common responses to reduce API calls

## Security Considerations

- **Private Models**: Set repository to private if needed
- **API Tokens**: Use read-only tokens for inference
- **Endpoint Protection**: Use "protected" type to require authentication
- **Rate Limiting**: Implement rate limits to prevent abuse
- **Input Validation**: Validate inputs before sending to model

## Cost Optimization Tips

1. **Scale to Zero**: Set min_replica=0 for dev/staging
2. **Batch Requests**: Group multiple requests when possible
3. **Cache Results**: Cache common queries
4. **Use CPU**: For low-traffic applications
5. **Monitor Usage**: Track and optimize based on actual patterns
6. **Auto-pause**: Pause endpoints during low-usage periods

## Support Resources

- **HuggingFace Docs**: https://huggingface.co/docs/inference-endpoints
- **Pricing**: https://huggingface.co/pricing#inference-endpoints
- **Status**: https://status.huggingface.co
- **Community**: https://discuss.huggingface.co

## Summary

Complete deployment flow:

1. ✅ Train on Hyperbolic Labs (~$40-100, 4-8 hours)
2. ✅ Export to HF format (5 minutes)
3. ✅ Upload to HF Hub (5-10 minutes)
4. ✅ Deploy on Inference Endpoints (10 minutes)
5. ✅ Start serving requests (~$0.06+/hour, auto-scaling)

**Total time to production**: ~1 day (mostly training)
**Ongoing cost**: Pay-per-use, scales with traffic

Your model is now production-ready! 🚀
