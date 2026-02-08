# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational from-scratch implementation of **PaliGemma**, a multimodal vision-language model (VLM). The code implements the full architecture: SigLIP vision encoder + linear projection + Gemma 2B language model decoder. This is a learning-focused codebase with extensive inline comments explaining each component.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Python 3.11 (see `.python-version`)
- Key dependencies: PyTorch 2.5.1, Pillow, NumPy

## Architecture (3 source files)

The model follows the PaliGemma architecture: Image -> Vision Encoder -> Linear Projection -> Language Model

### `modeling_siglip.py` - Vision Encoder (SigLIP)
Contrastive vision encoder that converts images to patch embeddings:
- `SiglipVisionConfig` -> `SiglipVisionModel` -> `SiglipVisionTransformer` -> `SiglipEncoder` (stacked `SiglipEncoderLayer`)
- Each encoder layer: LayerNorm -> Multi-Head Attention -> Residual -> LayerNorm -> MLP -> Residual
- `SiglipVisionEmbeddings`: Conv2d patch extraction + learned position embeddings
- Uses standard Multi-Head Attention (all heads have same count for Q, K, V)

### `modeling_gemma.py` - Language Model (Gemma) + Full PaliGemma Model
Contains both the Gemma decoder and the top-level PaliGemma model:
- **Config classes**: `GemmaConfig`, `PaliGemmaConfig` (wraps both vision and text configs)
- **Gemma components**: `GemmaRMSNorm`, `GemmaMLP` (gated with GELU), `GemmaAttention` (Grouped Query Attention with RoPE), `GemmaDecoderLayer`, `GemmaModel`, `GemmaForCausalLM`
- **PaliGemma top-level**: `PaliGemmaMultiModalProjector` (linear projection), `PaliGemmaForConditonalGeneration` (note: typo in class name "Conditonal")
- `KVCache`: Per-layer key/value cache for autoregressive inference
- `repeat_kv()`: Utility to expand KV heads to match query head count in GQA
- **NOT YET IMPLEMENTED**: `GemmaRotaryEmbedding` and `apply_rotary_pos_emb` are referenced but not defined in any file

### `processing.py` - Input Processor
Handles image and text preprocessing for the model:
- `PaliGemmaProcessor`: Main processor class that combines image + text processing
- Image pipeline: resize (BICUBIC) -> rescale (1/255) -> normalize (ImageNet mean/std 0.5) -> transpose to CHW
- Text pipeline: prepends `<image>` placeholder tokens + BOS token before user prompt
- `_merge_input_ids_with_image_features()` in the main model handles replacing placeholder tokens with actual vision embeddings

## Key Design Decisions

- **Grouped Query Attention (GQA)** in Gemma: fewer KV heads than query heads, KV heads are repeated via `repeat_kv()` to match query count
- **No padding support**: The model asserts `attention_mask == 1` everywhere; padding logic is not implemented
- **Weight tying**: `GemmaForCausalLM.tie_weights()` shares embedding and output projection weights
- **Embedding scaling**: Both text embeddings (by `sqrt(hidden_size)` in `GemmaModel.forward`) and image features (by `1/sqrt(hidden_size)` in `_merge_input_ids_with_image_features`) are scaled

## Reference Documentation

- `docs/PaliGemma-VLM.md`: Comprehensive explanation of every architectural concept (contrastive learning, SigLIP, attention mechanisms, KV cache, RoPE, normalization, etc.)
- `docs/resources/`: Diagrams referenced throughout the code comments (vision transformer, encoder, attention steps, architecture overviews)
