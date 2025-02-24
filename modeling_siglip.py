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

class SiglipAttention(nn.Module):
    """Multi-headed attention mechanism that processes input sequences in parallel through multiple attention heads.
    
    This implements the complete multi-head attention process as described in resources/PaliGemma-VLM.md:
    
    Step 1: Transform input sequence X into Q, K, V matrices
    - Input sequence is transformed through learned parameter matrices Wq, Wk, Wv
    - Each transformation splits the embedding dimension across multiple heads
    
    Step 2: Treat each head independently
    - Reshape and transpose matrices to enable parallel processing
    - Each head processes its own subset of the embedding dimensions
    
    Step 3: Calculate attention for each head
    - Compute attention scores through Q × K^T / √d_head
    - Apply softmax to get attention probabilities
    - Optional masking for causal attention in language models
    
    Step 4: Multiply by V sequence
    - Multiply attention probabilities with value vectors
    - Creates weighted combinations of value vectors
    
    Steps 5-7: Combine and mix head outputs
    - Transpose back to original sequence ordering
    - Concatenate all head outputs
    - Mix information through Wo parameter matrix
    """
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # Scale factor to prevent dot products from growing too large
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # Parameter matrices for transforming input sequence:
        # Each projects from embed_dim to embed_dim, but output is split across heads
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Key projection (Wk)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Value projection (Wv)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Query projection (Wq)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Output projection (Wo)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Process input sequence through multi-head attention.
        
        Args:
            hidden_states: Input sequence [batch_size, seq_len, embed_dim]
                         For our example: [B, 4, 1024] from resources/PaliGemma-VLM.md
            
        Returns:
            attn_output: Processed sequence with same shape as input
                        [batch_size, seq_len, embed_dim]
            attn_weights: Optional attention weights for visualization
            
        Shape transformations through each step:
        
        Step 1: X → Q, K, V
        - Input:     [B, 4, 1024]
        - After Wq/Wk/Wv: [B, 4, 1024] (but conceptually split for 8 heads)
        
        Step 2: Reshape for parallel processing
        - Reshape:   [B, 4, 8, 128]  (split embed_dim into num_heads × head_dim)
        - Transpose: [B, 8, 4, 128]  (move heads dim for parallel processing)
        
        Step 3: Calculate attention
        - Q × K^T:   [B, 8, 4, 4]    (attention scores for each head)
        - Softmax:   [B, 8, 4, 4]    (convert to probabilities)
        
        Step 4: Multiply by V
        - Attention × V: [B, 8, 4, 128]  (weighted combination of values)
        
        Steps 5-7: Combine heads
        - Transpose:    [B, 4, 8, 128]  (prepare for concatenation)
        - Concatenate:  [B, 4, 1024]    (join all heads)
        - Multiply Wo:  [B, 4, 1024]    (mix head information)
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Step 1: Transform input sequence through Wq, Wk, Wv
        # [B, 4, 1024] -> [B, 4, 1024]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Step 2: Reshape and transpose for parallel processing
        # First split embed_dim into num_heads × head_dim
        # [B, 4, 1024] -> [B, 4, 8, 128]
        # Then transpose to get heads dimension for parallel processing
        # [B, 4, 8, 128] -> [B, 8, 4, 128]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Calculate attention scores and apply scaling
        # [B, 8, 4, 128] × [B, 8, 128, 4] -> [B, 8, 4, 4]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)} but is"
                f" {attn_weights.size()}"
            )

        # Convert attention scores to probabilities with softmax, scores between 0 to 1 so that it sums up to 1
        # [B, 8, 4, 4] -> [B, 8, 4, 4] (same shape, now probabilities)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Step 4: Multiply attention weights with values
        # [B, 8, 4, 4] × [B, 8, 4, 128] -> [B, 8, 4, 128]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Steps 5-7: Combine and mix head outputs
        # Step 5: Transpose back, contiguous means we want the tensor to represent the information in memory
        # [B, 8, 4, 128] -> [B, 4, 8, 128]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Step 6: Concatenate heads, contiguous helps not make any computation in the memory for the reshape
        # [B, 4, 8, 128] -> [B, 4, 1024]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # Step 7: Mix information through Wo
        # [B, 4, 1024] -> [B, 4, 1024]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class SiglipMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) component used in the SigLIP encoder layer.
    
    This implements the feed-forward network (FFN) part of each transformer layer, which:
    1. Projects embeddings to a higher dimension (fc1)
    2. Applies non-linearity (GELU)
    3. Projects back to original dimension (fc2)
    
    The MLP serves several crucial purposes:
    - Increases model capacity through higher-dimensional projections
    - Introduces non-linearity to process complex patterns
    - Maintains dimensionality through the skip connection in encoder layer
    
    Looking at the siglip-encoder.png:
    - This is the "MLP" box in each encoder layer
    - Works together with attention to process patch embeddings
    - Part of the standard transformer architecture pattern:
      Attention → LayerNorm → MLP → LayerNorm
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # Project from hidden_size to higher dimension
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Project back to hidden_size for residual connection
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process embeddings through the MLP.
        
        Args:
            hidden_states: Input embeddings [batch_size, num_patches, embed_dim]
            
        Returns:
            torch.Tensor: Processed embeddings [batch_size, num_patches, embed_dim]
            
        The process:
        1. Project to higher dimension (fc1)
        2. Apply GELU activation
        3. Project back to original dimension (fc2)
        """
        # [B, num_patches, embed_dim] -> [B, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # [B, num_patches, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [B, num_patches, intermediate_size] -> [B, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipEncoderLayer(nn.Module):
    """Single transformer encoder layer in the SigLIP vision encoder.
    
    This implements one complete transformer block as shown in siglip-encoder.png:
    1. Layer Norm → Self-Attention → Residual Connection
    2. Layer Norm → MLP → Residual Connection
    
    Key components:
    - self_attn: Multi-head self-attention allowing patches to interact
    - layer_norm1: Pre-normalization before attention
    - mlp: Feed-forward network for additional processing
    - layer_norm2: Pre-normalization before MLP
    
    Looking at siglip-encoder.png:
    - Each encoder layer processes patch embeddings through attention and MLP
    - The "+" symbols represent residual connections
    - Layer norms (yellow boxes) stabilize activations
    - Multiple such layers are stacked to build the full encoder
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process embeddings through one encoder layer.
        
        Args:
            hidden_states: Input embeddings [batch_size, num_patches, embed_dim]
            
        Returns:
            torch.Tensor: Processed embeddings [batch_size, num_patches, embed_dim]
            
        The process (following siglip-encoder.png):
        1. Store residual for first skip connection
        2. Layer norm → Self-attention
        3. Add first residual connection
        4. Store residual for second skip connection
        5. Layer norm → MLP
        6. Add second residual connection
        """
        # [B, num_patches, embed_dim]
        residual = hidden_states
        # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)

        # First residual connection (+) in Siglip Encoder diagram
        # [B, num_patches, embed_dim]
        hidden_states = residual + hidden_states

        # Prepare second residual connection
        # [B, num_patches, embed_dim]
        residual = hidden_states
        # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [B, num_patches, embed_dim] -> [B, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = residual + hidden_states

        return hidden_states

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

