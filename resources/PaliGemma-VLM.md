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

**Problem:** How do we train the model to maximize similarity scores for matching image-text pairs while minimizing scores for non-matching pairs?

**Answer:** Through carefully designed loss functions!

#### Training with Loss Functions

To train CLIP effectively, we use **cross-entropy loss**. To understand why this works well, let's first look at how language models are typically trained:

1. **Language Model Training Example**
   - Given a sentence: "I am very into ___"
   - Using Next Token Prediction Task
   - The model produces embeddings → converts to logits (raw scores before softmax)
   
   Example logits output might look like:
   ```python
   # Raw logit scores for possible next words
   logits = {
       "fashion": 5.2,   # Highest score
       "games": 3.1,
       "sports": 2.8,
       "coding": 1.9,
       "music": 1.7
       # ... (thousands more words with scores)
   }
   
   # After softmax conversion to probabilities
   probabilities = {
       "fashion": 0.65,  # 65% confidence
       "games": 0.15,    # 15% confidence
       "sports": 0.12,   # 12% confidence
       "coding": 0.05,   # 5% confidence
       "music": 0.03     # 3% confidence
       # ... (all probabilities sum to 1)
   }
   ```
   
   - Logits are the raw scores before probability conversion
   - Cross-entropy loss helps by:
     - Converting these raw logits into probabilities using softmax
     - Pushing the probability of correct word ("fashion") towards 1
     - Pushing other probabilities towards 0

2. **Applying This to CLIP**
   - Instead of predicting next words like above, we're matching images and text instead
   - Each row/column in our similarity matrix needs one high value (matching pair dot products)
   - All other values should be low (non-matching pairs)
   - Cross-entropy loss helps achieve this pattern

#### CLIP Training Implementation

Here's an example of how the contrastive learning is implemented:

```python
# Based on official CLIP paper implementation
def clip_training(image_encoder, text_encoder, images, texts, temperature=0.07):
    # Step 1: Encode images and text through their respective encoders (purple and green sections in diagram)
    image_features = image_encoder(images)  # [batch_size, feature_dim] -> Creates I₁, I₂, I₃, ..., Iₙ
    text_features = text_encoder(texts)     # [batch_size, feature_dim] -> Creates T₁, T₂, T₃, ..., Tₙ
    
    # Step 2: Normalize features to unit length (helps with dot product similarity)
    # This ensures all similarity scores are between -1 and 1
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    # Step 3: Create the similarity matrix shown in the center of the diagram
    # image_features @ text_features.t() computes all possible I·T dot products
    # This creates the grid where each cell is Iᵢ·Tⱼ
    logits = (image_features @ text_features.t()) / temperature
    
    # Step 4: Set up the ground truth - we want the diagonal to be highest
    # In the diagram, this means:
    # - The blue squares should have high values (matching pairs)
    # - All other squares should have low values
    labels = torch.arange(len(images))
    
    # Step 5: Compute bidirectional loss
    # For each row (image): want I₁·T₁ to be highest in row 1, I₂·T₂ in row 2, etc.
    image_loss = cross_entropy_loss(logits, labels)
    # For each column (text): want I₁·T₁ to be highest in column 1, I₂·T₂ in column 2, etc.
    text_loss = cross_entropy_loss(logits.t(), labels)
    
    # Step 6: Combine losses symmetrically
    # This ensures both modalities are trained equally
    loss = (image_loss + text_loss) / 2
    
    return loss

# The training process aims to:
# 1. Maximize values in the blue squares (like I₁·T₁, I₂·T₂, etc.)
#    - These are the matching pairs shown in the diagram
# 2. Minimize values everywhere else
#    - The non-blue squares represent incorrect pairings
# 
# Temperature controls contrast in similarity scores:
# - Lower temperature (e.g., 0.07) = sharper contrast between matches/non-matches
# - Higher temperature = softer, more gradual distinctions
```

This implementation:
1. Processes batches of image-text pairs
2. Creates normalized embeddings in the same space
3. Computes similarity scores between all possible pairs
4. Uses cross-entropy loss to push matching pairs together
5. While pushing non-matching pairs apart in the embedding space

The temperature parameter helps control how "strict" the model is in its matching - lower values make it more certain about its choices, while higher values make it more flexible.

