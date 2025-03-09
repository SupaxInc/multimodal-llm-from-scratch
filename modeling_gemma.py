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
        self.vision_tower = SiglipVisionConfig(config.vision_config)
        # Linear Projection component
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        return self.language_model.tie_weights()