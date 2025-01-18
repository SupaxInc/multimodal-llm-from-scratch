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

class SiglipVisionEmbeddings(nn.Module):
    """Handles the initial processing of images into patch embeddings with position encoding.
    
    This implements the bottom two layers of the vision_transformer-diagram.png:
    1. IMAGE → EMBEDDINGS OF PATCHES: Converts image into fixed-size patches and projects them to embeddings
       using a convolutional layer (patch_embedding)
    2. POS. ENC. (LEARNED): Adds learned position embeddings to maintain spatial information
    
    The process:
    1. Input image (e.g. 224x224) is divided into patches (e.g. 16x16)
    2. Each patch is projected to an embedding vector via convolution
    3. Position embeddings are added to help the model understand spatial relationships
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # * From diagram: IMAGE → EMBEDDINGS OF PATCHES
        # The bottom layer showing image being split into numbered patches (1-16)
        # Convolutional layer that serves as patch embedder to extract embeddings:
        # - Input: Raw image of shape (batch_size, channels=3, height=224, width=224)
        # - The conv2d operation implicitly splits image into patches and projects each patch
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,  # Usually 3 for RGB images
            out_channels=self.embed_dim,      # Project each patch to embed_dim dimensions
            kernel_size=self.patch_size,      # Size of each patch (e.g., 16x16)
            stride=self.patch_size,           # Non-overlapping strides as its same size as kernel size
            padding="valid",                  # No padding, only complete patches
        )

        # * From diagram: The 4x4 grid showing 16 total patches
        # Calculate total number of patches:
        # For 224x224 image with 16x16 patches:
        # - Creates a 14x14 grid of patches (224/16 = 14) because:
        #   - The 224 pixel width is divided into 16-pixel wide patches: 224/16 = 14 patches horizontally
        #   - The 224 pixel height is divided into 16-pixel tall patches: 224/16 = 14 patches vertically
        #   - This creates a square grid of 14x14 = 196 total non-overlapping patches
        # - The (image_size // patch_size) gives number of patches in one dimension
        # - Square it to get total patches in 2D grid
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches # Number of position encodings

        # * From diagram: POS. ENC. row showing position numbers 1-16
        # Create learnable position embeddings:
        # - One embedding vector for each patch position
        # - Each position embedding has same dimension as patch embeddings
        # - These help model understand spatial relationships between patches when trained
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # Create fixed position IDs for each patch:
        # - torch.arange(self.num_positions): creates tensor [0, 1, ..., num_positions-1]
        # - expand((1, -1)): adds batch dimension, making shape (1, num_positions)
        # - These IDs index into position_embedding to get embeddings for each position
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)), 
            persistent=False,  # Don't save in state_dict as these are deterministic
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Process input images into patch embeddings with positional encoding.
        
        Args:
            pixel_values: Input images [batch_size, channels, height, width]
            
        Returns:
            embeddings: Patch embeddings with position encoding [batch_size, num_patches, embed_dim]
            
        Processing steps:
            1. Extract patches via convolution
            2. Reshape patches into sequence
            3. Add positional embeddings
        """
        # Input shape: [B, C, H, W]
        _, _, height, width = pixel_values.shape

        # * From diagram: IMAGE → EMBEDDINGS OF PATCHES section (convolution part)
        # Extract patch embeddings via convolution
        # Convolve the patch_size kernel over the image with no overlapping patches
        # [B, C, H, W] -> [B, embed_dim, num_patches_h, num_patches_w]
        # where num_patches_h = Height // patch_size, num_patches_w = Width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)

        # * From diagram: IMAGE → EMBEDDINGS OF PATCHES section (flatten part)
        # Flatten spatial dimensions (num_patches_h, num_patches_w) into sequence of patches
        # [B, embed_dim, num_patches_h, num_patches_w] -> [B, embed_dim, num_patches]
        # where num_patches = num_patches_h * num_patches_w
        embeddings = patch_embeds.flatten(2)
        
        # Transpose to get patches as sequence dimension
        # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        embeddings = embeddings.transpose(1, 2)

        # * From diagram: POS. ENC. (LEARNED) section + EMBEDDINGS OF PATCHES section
        # Add positional embeddings to provide spatial information
        # position_ids shape: [1, num_patches]
        # position_embedding output: [num_patches, embed_dim]
        # Final shape remains: [B, num_patches, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        # * From diagram: Output feeds into TRANSFORMER section
        # [B, num_patches, embed_dim]
        return embeddings

class SiglipVisionTransformer(nn.Module):
    """Vision Transformer that processes images through patch embedding, position encoding, and transformer layers.
    
    This implements the complete architecture shown in the diagram, processing from bottom to top:
    1. IMAGE → EMBEDDINGS OF PATCHES: Convert image into patch embeddings via self.embeddings.patch_embedding
    2. POS. ENC. (LEARNED): Add position encodings via self.embeddings.position_embedding
    3. TRANSFORMER: Process through transformer encoder allowing patches to interact via self-attention
    4. CONTEXTUALIZED EMBEDDINGS: Output final embeddings where each patch has information from all other patches
    
    The key insight is that this architecture:
    - Breaks down images into manageable patches
    - Converts patches into embeddings that can be processed like text tokens
    - Uses position encoding to maintain spatial relationships
    - Allows patches to share information through transformer layers
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
    """Main vision model that wraps the Vision Transformer.
    
    This is the top-level class that:
    1. Takes in raw images (IMAGE in diagram)
    2. Processes them through the Vision Transformer
    3. Outputs contextualized embeddings (CONTEXTUALIZED EMBEDDINGS in diagram)
       that can be used for downstream tasks
    """
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

