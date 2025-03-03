from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# Standard normalization values for ImageNet-based models
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5] # Values for RGB
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    Prepares the input prompt for PaliGemma VLM by adding image tokens and beginning-of-sequence token.
    
    For multimodal models like PaliGemma, the input needs to be structured in a specific format:
    1. First, a sequence of image tokens is added (256 tokens for PaliGemma 3B)
    2. Then, the beginning-of-sequence (BOS) token is added
    3. Finally, the text prompt follows
    
    During processing:
    - The image tokens are placeholders that will be replaced by actual image embeddings
    - The vision encoder extracts features from the image
    - These features are projected to the text embedding space and replace the image tokens
    
    Args:
        prefix_prompt (str): The text prompt to be processed alongside the image
        bos_token (str): The beginning-of-sequence token used by the tokenizer
        image_seq_len (int): Number of image tokens to use (256 for PaliGemma 3B)
        image_token (str): The placeholder token for images (typically "<image>")
        
    Returns:
        str: Formatted prompt with image tokens, BOS token, and the text prompt
    """
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process)

    # TODO: The PaliGemma paper states that \n should be tokenizer separately but HF implementation is not doing it. Could be an issue.
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(
        image: Image,
        size: Tuple[int, int],
        resample: Image.Resampling = None,
        reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )

    return resized_image

def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    For vision models, we typically rescale pixel values from [0, 255] to [0, 1]
    by using a scale factor of 1/255.0.
    
    Args:
        image (np.ndarray): Image array to rescale
        scale (float): Scaling factor (typically 1/255.0 for RGB images)
        dtype (np.dtype): Data type for the output array
        
    Returns:
        np.ndarray: Rescaled image array
    """
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)

    return rescaled_image

def normalize(image: np.ndarray, mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]]) -> np.ndarray:
    """
    Normalize an image using mean and standard deviation.
    
    Vision models are typically trained on normalized images where each channel
    has been standardized to have a specific mean and standard deviation.
    For PaliGemma/SigLIP, we use mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5].
    
    Args:
        image (np.ndarray): Image array to normalize
        mean (float or list): Mean value(s) for normalization
        std (float or list): Standard deviation value(s) for normalization
        
    Returns:
        np.ndarray: Normalized image array
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(
        images: List[Image.Image],
        size: Dict[str, int] = None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
) -> List[np.ndarray]:
    """
    Process a list of images for input to the vision encoder.
    
    This function performs the complete image preprocessing pipeline:
    1. Resize images to the required dimensions
    2. Convert to numpy arrays
    3. Rescale pixel values (typically from [0, 255] to [0, 1])
    4. Normalize using mean and standard deviation
    5. Transpose to [C, H, W] format expected by PyTorch models
    
    Args:
        images (List[Image.Image]): List of PIL images to process
        size (Dict[str, int]): Target size as (height, width)
        resample (Image.Resampling): Resampling method for resizing
        rescale_factor (float): Factor to rescale pixel values (typically 1/255.0)
        image_mean (float or list): Mean values for normalization
        image_std (float or list): Standard deviation values for normalization
        
    Returns:
        List[np.ndarray]: List of processed image arrays ready for the model
    """
    height, width = size[0], size[1]

    # Resize the images based on the resample
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]

    return images

class PaliGemmaProcessor:
    """
    Processor for preparing inputs for the PaliGemma vision-language model.
    
    This class handles the complete processing pipeline for both image and text inputs:
    1. Image processing: resize, normalize, and prepare for the vision encoder
    2. Text processing: tokenize and format with image placeholders
    3. Combined processing: create the unified input format for the model
    
    The key innovation is how image and text are combined:
    - Image tokens are added as placeholders
    - These placeholders will be replaced by actual image embeddings
    - The resulting sequence contains both modalities in a unified format
    
    The input format follows the structure shown in the PaliGemma paper:
    [img1, img2, ..., imgN][bos]prefix prompt[sep]
    """
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
        # Add the <image> token to the tokenizer's vocabulary
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

        # Add all the special tokens to the tokenizer
        tokenizer.add_tokens(EXTRA_TOKENS)
        # Get the ID for the image token to use later and insert image token place holder to be replaced by embeddings from vision encoder
        self.image_token_id = tokenizer.convert_token_to_ids(self.IMAGE_TOKEN)

        # Configure the tokenizer behavior
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
        """
        Process images and text for input to the PaliGemma model.
        
        Args:
            text (List[str]): List of text prompts
            images (List[Image.Image]): List of images
            padding (str): Padding strategy for tokenization
            truncation (bool): Whether to truncate sequences that are too long
            
        Returns:
            dict: Dictionary containing processed inputs:
                - pixel_values: Processed image tensors
                - input_ids: Token IDs for the combined sequence
                - attention_mask: Attention mask for the sequence
        """
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # Process the images through the complete pipeline
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

        # Tokenize the combined input strings
        # This converts the text to token IDs and creates attention masks
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        # Combine the processed images and tokenized inputs into a single dictionary
        return_data = { "pixel_values": pixel_values, **inputs }

        return return_data