# Multimodal LLM from Scratch

A from-scratch implementation of **PaliGemma**, a vision-language model (VLM) that combines a **SigLIP** vision encoder with a **Gemma 2B** language model to understand images and generate text.

## Architecture

```
Image ──► SigLIP Vision Encoder ──► Linear Projection ──► Gemma Language Model ──► Text Output
                                                    ▲
                                          Text Prompt ┘
```

- **SigLIP Vision Encoder** (`modeling_siglip.py`): Splits images into patches, embeds them, and processes through transformer layers with multi-head attention
- **Linear Projection** (`modeling_gemma.py`): Projects vision embeddings to match the language model's hidden dimensions
- **Gemma 2B Decoder** (`modeling_gemma.py`): Decoder-only transformer with grouped query attention, rotary position embeddings, and KV caching
- **Input Processor** (`processing.py`): Handles image preprocessing (resize, normalize) and text tokenization with image placeholder tokens

## Documentation

See [docs/PaliGemma-VLM.md](docs/PaliGemma-VLM.md) for detailed explanations of every architectural concept including contrastive learning, SigLIP, attention mechanisms, KV cache, RoPE, and normalization techniques.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**STILL WORK IN PROGRESS**