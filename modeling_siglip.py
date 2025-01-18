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
    """Vision Transformer that processes images through patch embedding, position encoding, and transformer layers.
    
    This implements the architecture shown in vision-transformer-diagram.png, with processing flowing bottom to top:
    1. Image → Patches via self.embeddings (IMAGE → EMBEDDINGS OF PATCHES in diagram)
    2. Add position encodings via self.embeddings (POS. ENC. in diagram) 
    3. Process through transformer encoder (TRANSFORMER box in diagram)
    4. Output contextualized embeddings (CONTEXTUALIZED EMBEDDINGS in diagram)
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Handles patch extraction, flattening patches to embeddings, and adding position encodings
        # Maps from raw image pixels to sequence of patch embeddings with position information
        self.embeddings = SiglipVisionEmbeddings(config)
        
        # Transformer encoder that allows patches to interact via self-attention
        # Each patch embedding can attend to all other patches to build global context
        self.encoder = SiglipEncoder(config)
        
        # Final layer norm for numerical stability of output embeddings
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Process a batch of images through the Vision Transformer.

        Args:
            pixel_values: Batch of images in shape (batch_size, channels=3, height, width)
                        Each image is divided into patches of size patch_size x patch_size

        Returns:
            torch.Tensor: Contextualized patch embeddings in shape (batch_size, num_patches, embed_dim)
                         Each patch embedding contains information from other patches via self-attention

        Processing steps (matching diagram):
            1. Convert images to sequence of patch embeddings via convolution
            2. Add learned position encodings to maintain spatial information
            3. Process through transformer layers allowing patches to interact
            4. Apply final layer normalization
        """
        # Extract patches and create embeddings with position encoding
        # (B, C, H, W) -> (B, num_patches, embed_dim) 
        hidden_states = self.embeddings(pixel_values)

        # Process through transformer encoder - patches attend to each other
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        # Final layer normalization
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

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

