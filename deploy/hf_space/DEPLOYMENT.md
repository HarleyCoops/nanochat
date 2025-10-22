# HuggingFace Space Deployment Instructions

This directory contains all the files needed to deploy NanoChat as a HuggingFace Space.

## Quick Start

### Option 1: Deploy via HuggingFace Web UI (Recommended)

1. **Create a new Space**:
   - Go to https://huggingface.co/new-space
   - Choose a name (e.g., `your-username/nanochat-demo`)
   - Select **Gradio** as the SDK
   - Choose **Public** or **Private**
   - Click **Create Space**

2. **Upload files**:
   - Click **Files** tab in your new Space
   - Click **Add file** > **Upload files**
   - Upload all files from this directory:
     - `app.py`
     - `requirements.txt`
     - `README.md`
     - `configuration_nanochat.py`
     - `modeling_nanochat.py`

3. **Wait for build**:
   - The Space will automatically start building
   - Building takes 5-10 minutes
   - Check the **Logs** tab for progress

4. **Test your Space**:
   - Once building is complete, the **App** tab will show the chat interface
   - Try asking it some questions!

### Option 2: Deploy via Git

1. **Clone your Space repository**:
   ```bash
   git clone https://huggingface.co/spaces/your-username/nanochat-demo
   cd nanochat-demo
   ```

2. **Copy files**:
   ```bash
   cp /path/to/nanochat561/deploy/hf_space/* .
   ```

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Initial commit: NanoChat Space"
   git push
   ```

4. **Monitor deployment**:
   - Visit https://huggingface.co/spaces/your-username/nanochat-demo
   - Check the **Logs** tab for build progress

### Option 3: Deploy via HuggingFace CLI

1. **Install HuggingFace CLI**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

2. **Create and upload**:
   ```bash
   huggingface-cli repo create nanochat-demo --type space --space_sdk gradio
   cd deploy/hf_space
   huggingface-cli upload your-username/nanochat-demo . --repo-type space
   ```

## Files Overview

- **app.py**: Main Gradio application code
- **requirements.txt**: Python dependencies
- **README.md**: Space description and metadata (appears on Space page)
- **configuration_nanochat.py**: Model configuration for transformers
- **modeling_nanochat.py**: Custom model implementation

## Configuration Options

### Hardware Settings

By default, Spaces run on CPU. For better performance:

1. Go to your Space **Settings**
2. Under **Hardware**, select:
   - **CPU basic** (free, slower)
   - **CPU upgrade** (small fee, faster)
   - **T4 small** (GPU, fastest but costs more)
3. Click **Save**

### Environment Variables (Optional)

You can add these in Space Settings > Variables:

- `HF_TOKEN`: If you want to use private models (not needed for this public model)

## Customization

### Change Model

Edit `app.py` line 10:
```python
MODEL_ID = "HarleyCooper/nanochat561"  # Change to your model
```

### Adjust UI

Edit `app.py` lines 80-95 to modify:
- Title and description
- Default parameter values
- Parameter ranges

### Update Dependencies

Edit `requirements.txt` to add/remove packages.

## Troubleshooting

### Space Fails to Build

**Check the Logs tab** for error messages. Common issues:

1. **Missing dependencies**: Add to `requirements.txt`
2. **Import errors**: Ensure `configuration_nanochat.py` and `modeling_nanochat.py` are uploaded
3. **Model not found**: Verify `MODEL_ID` in `app.py` is correct

### Space Runs but Chat Doesn't Work

1. **Check model loading**: Look for "Loading nanochat model" in logs
2. **Memory issues**: Upgrade to larger CPU or use GPU hardware
3. **Trust remote code**: Model loads with `trust_remote_code=True` by default

### Slow Responses

1. **Upgrade hardware**: Use GPU for faster inference
2. **Reduce max_new_tokens**: Lower the default value in `app.py`
3. **Use smaller model**: If available

## Space URL

Once deployed, your Space will be available at:
```
https://huggingface.co/spaces/your-username/nanochat-demo
```

You can embed it anywhere with an iframe:
```html
<iframe
  src="https://your-username-nanochat-demo.hf.space"
  width="850"
  height="600"
></iframe>
```

## Cost Estimates

- **CPU basic**: Free (with rate limits)
- **CPU upgrade**: ~$0.03/hour
- **T4 small GPU**: ~$0.60/hour
- **A10G GPU**: ~$3.15/hour

Spaces can be paused when not in use to save costs.

## Support

- HuggingFace Spaces Docs: https://huggingface.co/docs/hub/spaces
- Model Repository: https://huggingface.co/HarleyCooper/nanochat561
- Issues: https://github.com/HarleyCoops/nanochat561/issues

## License

MIT - See LICENSE file in the main repository.
