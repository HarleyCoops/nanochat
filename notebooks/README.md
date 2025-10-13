# NanoChat Notebooks

This directory contains Jupyter notebooks for training nanochat models on different platforms.

## Available Notebooks

### train_on_colab.ipynb
**Train nanochat on Google Colab's free or paid GPUs**

Perfect for users without local GPU access. Simply upload to Google Colab and run!

**GPU Options:**
- **Free Tier**: T4 (16GB) - Small models (~2-3 hours)
- **Colab Pro**: V100 - Medium models (~6-8 hours)
- **Colab Pro+**: A100 (40GB) - Large models

**Features:**
- Step-by-step guided training
- Automatic checkpoint saving to Google Drive
- Download trained models directly
- Configurable for different GPU sizes
- Troubleshooting guides included

**Quick Start:**
1. Upload notebook to Google Colab
2. Runtime > Change runtime type > GPU
3. Run all cells
4. Download your trained model

**Direct Link:** [Open in Colab](https://colab.research.google.com/github/HarleyCoops/nanochat/blob/master/notebooks/train_on_colab.ipynb)

## Cost Comparison

| Platform | GPU | Cost | Best For |
|----------|-----|------|----------|
| **Google Colab (Free)** | T4 | $0 | Learning, small models |
| **Colab Pro** | V100/A100 | $10/month | Medium models |
| **Colab Pro+** | A100 | $50/month | Large models, priority access |
| **Hyperbolic Labs** | H100/A100 | $1.50-3.20/hr | Production training |
| **Local GPU** | Your GPU | Electricity | If you have 24GB+ VRAM |

## Training Times Estimate

| Model Size | Colab T4 (Free) | Colab Pro (A100) | Hyperbolic H100 |
|------------|-----------------|------------------|-----------------|
| Small (60M) | ~2-3 hours | ~1 hour | ~30 mins |
| Medium (160M) | ~6-8 hours | ~2-3 hours | ~1 hour |
| Large (410M) | Not recommended | ~6-8 hours | ~2-3 hours |

## Tips for Colab Training

1. **Save to Google Drive** - Checkpoints survive disconnections
2. **Keep tab active** - Colab may disconnect inactive sessions
3. **Monitor memory** - Reduce batch size if you hit OOM errors
4. **Start small** - Test with small model first
5. **Use Colab Pro** - Worth it for faster GPUs and longer runtime

## After Training

Once training completes, you have several options:

### Option 1: Use Native Inference (Recommended)
```bash
# Download model from Colab/Drive
python -m scripts.chat_cli -i mid
python -m scripts.chat_web -i mid
```

### Option 2: Deploy to HuggingFace
```bash
# Export to HF format
python scripts/export_to_huggingface.py --source mid -o ./hf_model

# Upload and deploy
huggingface-cli upload your-username/model-name ./hf_model
```

### Option 3: Continue Training on Hyperbolic Labs
If you want to scale up, you can transfer your Colab-trained checkpoint to Hyperbolic Labs for further training on more powerful GPUs.

## Troubleshooting

**Disconnected during training?**
- Checkpoints are saved automatically every few hundred steps
- Download latest checkpoint from Google Drive
- Can resume training manually (or restart from checkpoint)

**Out of Memory (OOM)?**
- Reduce `device_batch_size` (16 → 8 → 4)
- Reduce `max_seq_len` (1024 → 512)
- Reduce model `depth` (12 → 10 → 8)

**Slow training?**
- Normal on T4, consider upgrading to Colab Pro
- Verify GPU is actually enabled (check nvidia-smi output)

**Can't install Rust?**
- Try restarting runtime and running Rust installation cell again
- Ensure PATH is updated correctly

## Contributing

Found a better way to train on Colab? Have optimization tips? Open a PR!
