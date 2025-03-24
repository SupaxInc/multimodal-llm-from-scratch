import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

# * Language Model (Gemma) config *
class GemmaConfig():
    """Configuration class for the Gemma language model.
    
    This defines all the parameters needed for the Gemma language model,
    which forms the right side (decoder) component of the PaliGemma architecture
    as shown in vision-language-model-architecture.png.
    
    Gemma is a decoder-only transformer that processes tokens sequentially
    using self-attention mechanisms with rotary positional encodings.
    
    Args:
        vocab_size: How many tokens in vocabulary
        hidden_size: Size of embeddings for each token
        intermediate_size: Size of the feed-forward network
        num_hidden_layers: Number of hidden layers (transformer blocks) 
        num_attention_heads: Number of attention heads for queries
        num_key_value_heads: Number of attention heads for keys and values
        head_dim: Dimensions each head will work with for multi-head attention (default: 256)
        max_position_embeddings: Maximum sequence length supported (default: 8192)
        rms_norm_eps: Epsilon for RMS normalization (default: 1e-6)
        rope_theta: Base value for rotary position embeddings (default: 10000.0)
        attention_bias: Whether to use bias in attention layers (default: False)
        attention_dropout: Dropout probability for attention (default: 0.0)
        pad_token_id: Token ID used for padding (default: None)
    """
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6, 
        rope_theta=10000.0, 
        attention_bias=False, 
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_values_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

# * Entire architecture including Gemma config *
class PaliGemmaConfig():
    """Configuration class for the entire PaliGemma multimodal model.
    
    This class defines parameters for the entire architecture as shown in
    vision-language-model-architecture.png, including both the vision encoder (SigLIP)
    and language model (Gemma) components, along with how they're connected.
    
    Args:
        vision_config: Configuration dictionary for the vision encoder (SigLIP)
        text_config: Configuration dictionary for the text decoder (Gemma)
        ignore_index: Value to ignore for labels during training (default: -100)
        image_token_index: Token ID corresponding to the <image> placeholder (default: 256000)
        vocab_size: Total vocabulary size of the model (default: 257152)
        projection_dim: Dimension that image features are projected to before entering language model (default: 2048)
        hidden_size: Embedding size of the language model (default: 2048)
        pad_token_id: Token ID used for padding (default: None)
    """
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None, 
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        # Configuration of the text language model (Gemma) 
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # Calculate number of image tokens based on image size and patch size
        # This corresponds to how many image tokens are in the unified representation (middle part of vision-language-model-architecture.png)
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x):
        # Calculating the RMSNorm(x) equation
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # 1 / sqrt(...)

    def forward(self, x):
        output = self._norm(x.float())

        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

# * Transformer layers for Gemma model consisting of attention, FFN, and RMS normalization *
class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Grab reference of initial input for skip connection
        residual = hidden_states
        
        # Apply initial layer normalization to input
        # [B, seq_len, hidden_size]
        hidden_states = self.input_layernorm(hidden_states)
        # Apply the output of layer norm to self attention to mask the positional encodings
        # [B, seq_len, hidden_size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # Sum up the outputs of self attention with the skip connection
        # [B, seq_len, hidden_size]
        hidden_states = residual + hidden_states

        # Grab reference of outputs for skip connection
        # [B, seq_len, hidden_size]
        residual = hidden_states

        # Apply last layer normalization to the outputs
        # [B, seq_len, hidden_size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Pass the outputs thru the feed forward network
        # [B, seq_len, hidden_size]
        hidden_states = self.mlp(hidden_states)
        # Sum up the outputs with the skip connection
        # [B, seq_len, hidden_size]
        hidden_states = residual + hidden_states

        return hidden_states
    
# * Gemma Model (language model), embedding layer + list of transformer layers*
class GemmaModel(nn.Module):
    """Core Gemma transformer model that processes token embeddings through transformer layers.
    
    As shown in transformer-architecture-outputs.png, this implements the main transformer 
    architecture including embedding layer, multiple decoder layers with attention and feed-forward 
    networks, and final normalization. This corresponds to the middle part of the diagram 
    excluding the final linear layer and softmax.
    
    Args:
        config: Gemma configuration dictionary containing model parameters
    """
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embedding layer - converts token IDs to vectors of size hidden_size, padding is the position of embedding tokens in vocab
        # This is the initial embedding before any transformer processing
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Stack of transformer decoder layers 
        # In transformer-architecture-outputs.png, this is the "GemmaDecoderLayer × N" box
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Final RMS normalization before output
        # This is the "Final RMS Norm" box in transformer-architecture-outputs.png
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """Process token embeddings through the transformer stack.
        
        Args:
            attention_mask: Mask to avoid attending to padding tokens
            position_ids: Positions of tokens for rotary positional encodings
            inputs_embeds: Token embeddings of shape [B, seq_len, hidden_size]
            kv_cache: Optional key-value cache for faster inference
            
        Returns:
            torch.FloatTensor: Contextualized token embeddings of shape [B, seq_len, hidden_size]
        """
        # Use provided embeddings directly
        # [B, seq_len, hidden_size]
        hidden_states = inputs_embeds
        
        # Scale embeddings by sqrt(hidden_size) as per transformer architecture best practices
        # [B, seq_len, hidden_size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # Process through each transformer decoder layer sequentially
        # Each layer creates increasingly contextualized embeddings
        for decoder_layer in self.layers:
            # [B, seq_len, hidden_size] -> [B, seq_len, hidden_size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )
        
        # Apply final layer normalization 
        # [B, seq_len, hidden_size]
        hidden_states = self.norm(hidden_states)

        # Return contextualized embeddings
        # [B, seq_len, hidden_size]
        return hidden_states



# * Gemma Model (transformer) + Linear Layer *
class GemmaForCausalLM(nn.Module):
    """Gemma model with language modeling head for next token prediction.
    
    This extends the GemmaModel with a linear projection layer (lm_head) that converts
    hidden states to vocabulary logits. As shown in transformer-architecture-outputs.png,
    this implements both the transformer stack and the "Linear Layer" that follows it.
    
    Args:
        config: Configuration object containing model parameters
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config) # Transformer model (language model)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        """Tie the weights between input embeddings and output linear layer.
        
        This is a common technique in language models to share parameters and
        improve performance by making the embedding and output projection layers use the same weights.
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """Process embeddings through transformer and project to vocabulary logits.
        
        Args:
            attention_mask: Mask to avoid attending to padding tokens
            position_ids: Positions of tokens for rotary positional encodings
            inputs_embeds: Token embeddings of shape [B, seq_len, hidden_size]
            kv_cache: Optional key-value cache for faster inference
            
        Returns:
            Dictionary containing:
                - logits: Output token probabilities of shape [B, seq_len, vocab_size]
                - kv_cache: Updated KV cache if provided
        """
        # Outputs will be a series of embeddings
        # [B, seq_len, hidden_size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache
        )

        # Convert embeddings to logits
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    """Projects vision encoder outputs to dimensions compatible with the language model.
    
    This implements the "Linear projection" component shown in vision-language-model-architecture.png,
    which converts image embeddings from the vision encoder to dimensions expected by the language model.
    
    Args:
        config: PaliGemma configuration object
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        # Linear layer that converts the hidden size of vision model into projection dimensions 
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        """Project image features to the right dimension for the language model.
        
        Args:
            image_features: Features from vision encoder of shape [B, num_patches, embed_dim]
            
        Returns:
            torch.Tensor: Projected features of shape [B, num_patches, projection_dim]
        """
        # [B, num_patches, embed_dim] -> [B, num_patches, projection_dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditonalGeneration(nn.Module):
    """Complete PaliGemma model for conditional text generation based on images and prompts.
    
    This implements the entire architecture shown in vision-language-model-architecture.png:
    1. Vision encoder (SigLIP) processes images
    2. Linear projection adapts image features to the language model's dimensions
    3. Image and text embeddings are merged
    4. Language model (Gemma) generates text based on the combined representation
    
    Args:
        config: Configuration object containing parameters for all model components
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config

        # Vision encoder for processing images - left side of vision-language-model-architecture.png
        # "SigLIP: 400M Vision Model" and "Contrastive Vision Encoder" component
        self.vision_tower = SiglipVisionModel(config.vision_config)
        
        # Projection layer to match vision output dimensions to language model dimensions
        # "Linear projection" component in vision-language-model-architecture.png
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

        self.vocab_size = config.vocab_size

        # Language model for text generation - right side of vision-language-model-architecture.png
        # "Gemma: 2B Language Model" and "Transformer Decoder" component
        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        """Tie input and output embedding weights in the language model."""
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        """Merge image features with text token embeddings to create a unified representation.
        
        This implements the middle part of vision-language-model-architecture.png where
        image tokens and text tokens are combined into a single sequence.
        
        Args:
            image_features: Processed image features from vision model and projector
            inputs_embeds: Text token embeddings from language model's embedding layer
            input_ids: Token IDs for the text part
            attention_mask: Attention mask for the sequence
            kv_cache: Optional key-value cache for inference
            
        Returns:
            Tuple containing:
                - final_embedding: Combined image and text embeddings
                - causal_mask: Attention mask for the transformer
                - position_ids: Position IDs for rotary positional encoding
        """
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        # Embedding of each token after they have been extracted from embedding layer of language model
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        
        # Scale image features similar to how text embeddings are scaled
        # [B, num_patches, hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens
        # [B, seq_len, embed_dim]
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        # Create masks that will be useful for understanding which is the placeholder <image>, text, padding tokens
        # E.g. Input ids: [567, 567, 567, 567, 567, 1, 65, 78, 99, 21, 11, 2]
            # 567                -> image placeholder tokens
            # 1                  -> beginning of sequence token
            # 65, 78, 99, 21, 11 -> user input prompt tokens
            # 2                  -> \n token
        # [B, seq_len]: True for text tokens
            # Converts input_ids above: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # [B, seq_len]: True for image tokens
            # Converts input_ids above: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        image_mask = input_ids == self.config.image_token_index
        # [B, seq_len]: True for padding tokens (won't be used for this implementation)
            # Converts input_ids above: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pad_mask = input_ids ==self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings to final embeddings
            # If text_mask_expanded true: Copy inputs_embeds -> else -> Copy final_embedding
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings by copying scaled image features where image_mask_expanded is true
            # Can't use torch.where since the seq len of scaled_image_features is not equal to seq len of final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens (not implemented)
            # If pad_mask_expanded true: Copy tensor made up of zeros other wise keep final embedding as it is
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # Create appropriate attention mask based on KV Cache state
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase: no need to mask tokens for autoregressive generation yet
            # NOTE: Only works when we have no padding
            # Create causal mask with shape [B, q_len, q_len]
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Generation phase: processing a single new token
            # Each token should attend to all previous tokens
            assert q_len == 1
            
            kv_len = kv_cache.num_items() + q_len
            # Do not need to mask anything, each query should be able to attend to all previous tokens
            # NOTE: Only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add head dimension for multi-head attention
        # [B, q_len, kv_len] -> [B, num_heads_q, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(1)

        # Generate position IDs for rotary positional encodings based on KV Cache state
        if kv_cache is not None and kv_cache.num_items() > 0:
            # For generation phase, use the position of the last token
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # For prefill phase, create position IDs for all tokens
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None, # Image extracted from PaliGemma processing.py using ImageNet mean/dev
            attention_mask: Optional[torch.Tensor] = None, # Provided by the tokenizer from PaliGemma processing.py
            kv_cache: Optional[KVCache] = None, 
    ) -> Tuple:
        """Process image and text inputs to generate conditional outputs.
        
        This implements the full forward pass through all components shown in 
        vision-language-model-architecture.png.
        
        Args:
            input_ids: Token IDs including image placeholders
            pixel_values: Preprocessed image input
            attention_mask: Attention mask for the sequence
            kv_cache: Optional key-value cache for faster inference
            
        Returns:
            Model outputs including logits and updated cache
        """
        # Code was not implemented to include padding
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extract the input embeddings of the <image> placeholder and text tokens
        # [B, seq_len, hidden_size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images using Siglip Vision Model
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds).dtype)
        # Re-size image embeddings into the same size of the language model hidden size
        # [B, num_patches, embed_dim] -> [B, num_patches, hidden_size]
        image_features = self.multi_modal_projector(selected_image_feature)
        
        # Merge the embeddings of the text tokens and image tokens that were re-sized by the linear projection
            # The input embeddings (inputs_embeds) is the middle part of the diagram that contains both text and image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, 
            inputs_embeds,
            input_ids, 
            attention_mask,
            kv_cache
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs