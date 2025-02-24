from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

class PaliGemmaProcessor:
    # The tokenizer will only insert tokens for the text but we need a way to separate the image tokens
        # So we insert placeholder tokens (<image>) that will be replaced by the embeddings extracted by the vision encoder
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_images_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_images_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        # https://huggingface.co/blog/paligemma
        # PaliGemma uses the tokenizer of the Gemma model but it was not created for special tokens with images
        tokens_to_add = { "additional_special_tokens": [self.IMAGE_TOKEN] }
        tokenizer.add_special_tokens(tokens_to_add)
        
        # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS = [
            f"<loc{i:04d}" for i in range(1024) # These are positions in the image that will be in the output showing where boxes are
        ]
        # These tokens are used for object segmentation 
        EXTRA_TOKENS += [
            f"<loc{i:03d}" for i in range(128)
        ]

        tokenizer.add_tokens(EXTRA_TOKENS)
        # Insert image token place holder to be replaced by embeddings from vision encoder
        self.image_token_id = tokenizer.convert_token_to_ids(self.IMAGE_TOKEN)

        # Add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer
    
    def __call__(
            self,
            text: List[str],
            images: List[Image.Image],
            padding: str = "longest",
            truncation: bool = True,
    ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # Convert the list of numpy arrays to a single numpy array with shape [B, C, H, W]
        pixel_values = np.stack(pixel_values, axis=0)
        # Convert the numpy array to a tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_strings = [
            # Creates tokens for the text and placeholder image tokens
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = { "pixel_values": pixel_values, **inputs }

        return return_data