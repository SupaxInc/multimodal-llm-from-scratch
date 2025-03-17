import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

# * Language Model (Gemma) config *
class GemmaConfig():
    def __init__(
        self,
        vocab_size, # How many tokens in vocabulary
        hidden_size, # Size of embeddings for each token
        intermediate_size, # Feed forward size
        num_hidden_layers, # Number of hidden layers for transformer
        num_attention_heads, # Refers to number of heads for queries
        num_key_value_heads,
        head_dim=256, # Dimensions each head will work with for multi head attention
        max_position_embeddings=8192, # How many dimensions each head will watch
        # From config file of PaliGemma in HuggingFace
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
    def __init__(
        self,
        vision_config=None, # Config of vision encoder
        text_config=None, # Config of text decoder (Gemma)
        ignore_index=-100, # Used for labels during training (we are only doing inference)
        image_token_index=256000, # Token corresponding to the <image> token
        vocab_size=257152, # Vocabulary size of model
        projection_dim=2048, # Final dimension that the image features should be resized to before fed to language model
        hidden_size=2048, # Embedding size of the language model (tokens are embeddings which have dimensions)
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

        # How many patches for each image which corresponds to how many image tokens are in the unified representation (middle diagram)
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class PaliGemmaForConditonalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config

        # Contrastive Vision Encoder component
        self.vision_tower = SiglipVisionModel(config.vision_config)
        # Linear Projection component
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        # Embedding dimension from image features that have been resized from Linear Projection
        _, _, embed_dim = image_features.shape
        # How many tokens we have, input ids is the number of the position from the vocabulary
        batch_size, sequence_length = input_ids.shape
        # Embedding of each token after they have been extracted from embedding layer of language model
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # [B, seq_len, hidden_size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens
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

        # Creation of the attention mask is based on KV Cache
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0: # Prefill phase
            # Prefill phase, do not mask any tokens
            # NOTE: Only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we're generating tokens, the query must be one single token
            assert q_len == 1
            
            kv_len = kv_cache.num_items() + q_len
            # Do not need to mask anything, each query should be able to attend to all previous tokens
            # NOTE: Only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [B, q_len, kv_len] -> [B, num_heads_q, q_len, kv_len]
        causal_mask = causal_mask.unsqueeze(1)

        # Generate the positions of the tokens that will be used by rotary positional encodings
        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids on the size of the attention_mask
            # For masked tokens, use the number 1 as position
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None, # Image extracted from PaliGemma processing.py using ImageNet mean/dev
            attention_mask: Optional[torch.Tensor] = None, # Provided by the tokenizer from PaliGemma processing.py
            kv_cache: Optional[KVCache] = None, 
    ) -> Tuple:
        # Code was not implemented to include padding
        assert torch.al(attention_mask == 1), "The input cannot be padded"

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