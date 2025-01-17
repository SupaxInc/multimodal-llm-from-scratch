from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
            self,
            # Size of the embedding vector of the vision transformer
            hidden_size=768,
            # Size of the intermediate (feed-forward network) linear layers
            intermediate_size=3072,
            # Number of attention heads for multi-head attention
            num_attention_heads=12,
            # Number of transformer layers in the encoder
            num_hidden_layers=12,
            # Number of input image channels (3 for RGB)
            num_channels=3,
            # Size of input images (224x224 pixels), PaliGemma comes in different resolutions
            image_size=224,
            # Size of image patches that get embedded (16x16 pixels)
            patch_size=16,
            # Small epsilon for layer normalization numerical stability
            layer_norm_eps=1e-6,
            # Dropout probability in attention layers
            attention_dropout=0.0,
            # Optional override for number of image tokens
            # Default is (image_size/patch_size)^2 + 1 for cls token
            num_image_tokens: int = None,
            **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values) -> Tuple:
        """Vision model processes batches of images through a Vision Transformer.

        Args:
            pixel_values: Batch of images in shape (batch_size, channels=3, height, width)

        Returns:
            Image embeddings in shape (batch_size, num_patches + 1, hidden_size)
            where +1 is for the [CLS] token that summarizes the whole image

        Processing steps:
            1. Splits each image into 16x16 patches
            2. Projects patches to embeddings and adds position encoding 
            3. Processes through transformer layers
        """
        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Num_Patches, Embed_Dim)
        return self.vision_model(pixel_values=pixel_values)

