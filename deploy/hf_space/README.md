---
title: NanoChat 561M
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
suggested_hardware: cpu-basic
models:
  - HarleyCooper/nanochat561
---


# NanoChat 561M - Chat Interface

A Gradio-based chat interface for the NanoChat 561M parameter language model.

## About NanoChat

NanoChat is a full-stack implementation of a ChatGPT-like language model trained from scratch. This 561M parameter model demonstrates that capable conversational AI can be trained with modest computational budgets (~$100 in compute costs).

### Model Details

- **Parameters**: 561M (560,988,160 learnable parameters)
- **Architecture**: Decoder-only Transformer (GPT-style)
- **Depth**: 20 layers
- **Context Length**: 2048 tokens
- **Vocabulary**: 65,536 tokens (BPE)
- **Training Cost**: ~$100 on 8x H100 GPUs

### Features

- Modern Transformer architecture with RoPE, RMSNorm, and Multi-Query Attention
- Efficient inference with KV caching
- Interactive chat interface
- Adjustable generation parameters (temperature, top-k, etc.)

## Performance Metrics

Based on the final SFT checkpoint:

- **ARC-Easy**: 43.35%
- **ARC-Challenge**: 32.51%
- **MMLU**: 32.35%
- **GSM8K**: 5.53%
- **HumanEval**: 6.10%

## Usage

Simply type your message in the chat box and adjust the generation parameters as needed:

- **Max new tokens**: Controls the maximum length of the response
- **Temperature**: Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.5) make it more focused
- **Top-p**: Nucleus sampling parameter
- **Top-k**: Limits sampling to the k most likely tokens
- **Repetition penalty**: Reduces repetitive text

## Model Repository

This Space uses the model from: [HarleyCooper/nanochat561](https://huggingface.co/HarleyCooper/nanochat561)

## Source Code

The full training and inference code is available at: [nanochat561 repository](https://github.com/HarleyCoops/nanochat561)

Based on the original work by Andrej Karpathy: [karpathy/nanochat](https://github.com/karpathy/nanochat)

## License

MIT License - See the model repository for details.
