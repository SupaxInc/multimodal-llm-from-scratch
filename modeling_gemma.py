import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

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