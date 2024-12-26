![vision-language-model](vision-language-model-architecture.png)

# Components

## Contrastive Vision Encoder

### What is contrastive learning?
![contrastive-encoder](contrastive-encoder.png)
Above is an example of **CLIP** (Contrastive-Language-Image Pre-training) architecture.

#### Text Encoder
The text encoder in CLIP typically uses a Transformer-based architecture (similar to GPT or BERT):
1. Input Processing
   - Text is first tokenized into sub-words
   - Example: "Pepper the aussie pup" → ["Pepper", "the", "aussie", "pup"]

2. Token Embeddings (e.g. T1, T2, T3 ... Tn)
   - Each token is converted into an embedding vector
   - Positional encodings are added to maintain sequence order

3. Transformer Processing
    ```
    [CLS] (special classification token that aggregates sequence info) + tokens → Transformer Layers →
    ↓
    Self-attention processes relationships between words
    ↓
    Feed-forward networks process token representations
    ↓
    Final text embedding
    ```

---

#### Image Encoder
The image encoder typically uses a Vision Transformer (ViT) or CNN architecture:
1. Image Preprocessing
    ```
    Original Image (in this case puppy photo) → Resize → Normalize
    ↓
    Split into patches (for ViT) or process through conv layers (for CNN)
    ```

2. For Vision Transformer (ViT)
    ```
    Image patches → Linear projection + position embeddings
    ↓
    Transformer encoder layers process patch relationships
    ↓
    Image embeddings (I₁, I₂, I₃, ..., Iₙ)
    ↓
    Paired with text embeddings to form similarity matrix
    ```

3. For CNN-based
    ```
    Image → Convolutional layers
    ↓
    Feature maps at different scales
    ↓
    Global pooling for final representation
    ```

PaliGemma is **ViT-based** rather than CNN-based.

---

#### Similarity Matrix
The grid in the center represents how well each image feature matches with each text feature. Essentially
the corresponding text embedding paired with the image embedding's dot product should have a high value (the blue squares). 
(e.g. I1 * T1)

A longer overview: 
1. Matrix Structure
   - Rows: Image features (I₁, I₂, I₃, ..., Iₙ)
   - Columns: Text features (T₁, T₂, T₃, ..., Tₙ)
   - Each cell (Iᵢ·Tⱼ): Similarity score between an image feature and text feature

2. How it Works
   - Higher scores (brighter cells) = stronger matches (e.g. I₁ * T₁)
   - Lower scores (lighter cells) = weaker matches (e.g. I₁ * T₂)
   - During training, the model learns to:
     - Maximize scores for matching image-text pairs
     - Minimize scores for non-matching pairs

3. Training Objective
   - The model wants matching pairs (like a dog photo with "aussie pup" text) to have high similarity
   - Non-matching pairs (like a dog photo with "red car" text) should have low similarity
   - This pushes the model to understand the relationship between visual and textual content

This contrastive approach helps the model learn meaningful connections between images and text without needing explicit labels for every concept.

---

<br>

**Problem:** How do we tell the model we want one item in each row/column (brighter cells) to be maximized while minimizing all the others?

**Answer:** The use of loss functions!

#### Loss Functions

To ensure that all matching dot products (brighter cells) have a maximized score as opposed to the non-matched dot products (lighter cells), we use something called loss functions. In our case, we could use the loss function **cross-entropy loss**.

To understand why cross-entropy loss is used, we need to first know how language models are trained. When training LLM's it is done using **next token prediction task**.

Lets say we have a sentence "I am very into", the LLM will produce a series of embeddings which are then converted into logits. logits are a vector that tells what is the score that the language model has assigned to what the next token should be amongst the tokens in the library. In the library, if there are words such as "fashion", "games", or "sports". If we want the sentence "I am very into fashion" then we would then use cross entropy loss to ensure that the highest score would be "fashion" so that it could be the next token in the sentence. To do this, cross entropy loss converts the vectors into a distribution with the softmax function then we compare with the label and we force the output to be equal to the label (where fashion is classified as a 1 and other tokens are 0). 

This is very similar to what is happening in the similarity matrix above. We can force each column and rows in the similarity matrix where the brighter cells (matched dot products) should be the highest value and all the other numbers should have a low value. 

*show example pseudo code here from the CLIP paper on how to implement the CLIP training with contrastive loss and comments of how it works based on what we have talked about already*

