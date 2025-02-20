# Table of Contents
- [Table of Contents](#table-of-contents)
- [Components](#components)
  - [Contrastive Vision Encoder](#contrastive-vision-encoder)
    - [What is contrastive learning?](#what-is-contrastive-learning)
      - [Text Encoder](#text-encoder)
      - [Image Encoder](#image-encoder)
      - [Similarity Matrix](#similarity-matrix)
      - [Training with Loss Functions](#training-with-loss-functions)
      - [CLIP Training Implementation](#clip-training-implementation)
    - [What is the problem with CLIP?](#what-is-the-problem-with-clip)
      - [1. The Softmax Function](#1-the-softmax-function)
        - [In CLIP's Context](#in-clips-context)
        - [The Solution: Log-Space Calculations](#the-solution-log-space-calculations)
      - [2. Computational Challenges in CLIP](#2-computational-challenges-in-clip)
        - [Asymmetric Computation Requirements](#asymmetric-computation-requirements)
        - [Why This Is Expensive](#why-this-is-expensive)
        - [**Solution** to Computational Challenges: Sigmoid Loss](#solution-to-computational-challenges-sigmoid-loss)
          - [Why Replace Cross-Entropy Loss?](#why-replace-cross-entropy-loss)
          - [How Sigmoid Loss Works](#how-sigmoid-loss-works)
        - [SigLIP's Sigmoid-based Solution](#siglips-sigmoid-based-solution)
          - [How SigLIP Works](#how-siglip-works)
        - [Example of SigLIP Processing](#example-of-siglip-processing)
    - [Vision Transformers (general)](#vision-transformers-general)
      - [Actual SigLip Encoder Diagram](#actual-siglip-encoder-diagram)
- [Random Teachings](#random-teachings)
  - [Normalization](#normalization)
    - [Linear Layer and Layer Normalization Example](#linear-layer-and-layer-normalization-example)
    - [Why Layer Normalization?](#why-layer-normalization)
    - [The Problem of Covariate Shift](#the-problem-of-covariate-shift)
      - [Batch Normalization: A Solution to Covariate Shifts](#batch-normalization-a-solution-to-covariate-shifts)
      - [Layer Normalization: A Better Solution](#layer-normalization-a-better-solution)
      - [Key Difference: Normalization Dimensions](#key-difference-normalization-dimensions)
  - [Multi-Head Attention](#multi-head-attention)
    - [Vision Transformers: Contextualizing Image Patches](#vision-transformers-contextualizing-image-patches)
    - [Language Models: Causal Self-Attention](#language-models-causal-self-attention)
    - [Parallel Training: A Powerful Feature](#parallel-training-a-powerful-feature)
    - [How it Works](#how-it-works)
      - [Step 1: From X to Q, K, V Transformations](#step-1-from-x-to-q-k-v-transformations)
        - [Matrix Multiplication Process](#matrix-multiplication-process)
        - [Visualizing the Transformation](#visualizing-the-transformation)
        - [Importance of Matrix Multiplication for Multi-Head Attention](#importance-of-matrix-multiplication-for-multi-head-attention)
      - [Step 2: Treat Each Head Independently!](#step-2-treat-each-head-independently)
        - [Initial Matrix Structure (Left Side)](#initial-matrix-structure-left-side)
        - [Transposition Process (Arrow in Diagram)](#transposition-process-arrow-in-diagram)
        - [Resulting Structure (Right Side)](#resulting-structure-right-side)
        - [Why This Transformation Matters](#why-this-transformation-matters)
      - [Step 3: Calculate the Attention for Each Head in Parallel](#step-3-calculate-the-attention-for-each-head-in-parallel)
        - [Matrix Setup and Multiplication](#matrix-setup-and-multiplication)
        - [Attention Score Computation](#attention-score-computation)
        - [Scaling Factor (√d\_head)](#scaling-factor-d_head)
        - [Attention Mask for Language Models](#attention-mask-for-language-models)
        - [Softmax Application](#softmax-application)
      - [Step 4: Multiply by the V Sequence](#step-4-multiply-by-the-v-sequence)
        - [Matrix Components](#matrix-components)
        - [Computing Weighted Sums](#computing-weighted-sums)
        - [Understanding the Weighted Combinations](#understanding-the-weighted-combinations)
      - [Step 5, 6, 7: Transpose Back, Concatenate all the Heads and Multiply by Wo](#step-5-6-7-transpose-back-concatenate-all-the-heads-and-multiply-by-wo)
        - [Step 5: Transpose Back](#step-5-transpose-back)
        - [Step 6: Concatenate all the Heads](#step-6-concatenate-all-the-heads)
        - [Step 7: Multiply by Wo](#step-7-multiply-by-wo)
        - [Why This Process Matters](#why-this-process-matters)

# Components

![vision-language-model](vision-language-model-architecture.png)

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

<br>

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

<br>

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

<br>

---

**Problem:** How do we train the model to maximize similarity scores for matching image-text pairs (brighter cells) while minimizing scores for non-matching pairs (lighter cells)?

**Answer:** We use cross-entropy loss!

<br>

---

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
   
   # After softmax conversion to probabilities distribution
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

   Let's look at a small example with 3 images and 3 text pairs:

   **Step 1: Raw Similarity Scores**
   - When I₁ (dog photo) is compared with:
     - T₁ (text "aussie pup"): score 0.9 ✓
     - T₂ (text "red car"): score 0.3 ✗
     - T₃ (text "blue sky"): score 0.2 ✗

   - When I₂ (car photo) is compared with:
     - T₁ (text "aussie pup"): score 0.2 ✗
     - T₂ (text "red car"): score 0.8 ✓
     - T₃ (text "blue sky"): score 0.3 ✗

   - When I₃ (sky photo) is compared with:
     - T₁ (text "aussie pup"): score 0.1 ✗
     - T₂ (text "red car"): score 0.2 ✗
     - T₃ (text "blue sky"): score 0.95 ✓

   **Step 2: After Softmax (Image → Text Direction)**
   - For I₁ (dog photo):
     - T₁: 65% probability ✓ (want this to be 100%)
     - T₂: 20% probability ✗ (want this to be 0%)
     - T₃: 15% probability ✗ (want this to be 0%)

   **Step 3: After Softmax (Text → Image Direction)**
   - For T₁ (text "aussie pup"):
     - I₁: 70% probability ✓ (want this to be 100%)
     - I₂: 20% probability ✗ (want this to be 0%)
     - I₃: 10% probability ✗ (want this to be 0%)

   Cross-entropy loss then:
   - Makes correct pairs (marked with ✓) have higher probability
   - Makes incorrect pairs (marked with ✗) have lower probability
   - Does this in both directions (image→text and text→image)
   - Combines both directions for balanced training

The actual code implementation of this process is shown in the "CLIP Training Implementation" section below.

<br>

---

#### CLIP Training Implementation

Here's an example of how the contrastive learning is implemented based on the CLIP diagram above:

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
# - Control how "strict" the model is in its matching
# - Lower temperature (e.g., 0.07) = sharper contrast between matches/non-matches
# - Higher temperature = softer, more gradual distinctions
```

This implementation:
1. Processes batches of image-text pairs
2. Creates normalized embeddings in the same space
3. Computes similarity scores between all possible pairs
4. Uses cross-entropy loss to push matching pairs together
5. While pushing non-matching pairs apart in the embedding space

<br>

---

<br>

### What is the problem with CLIP?

Using cross-entropy loss is a problem with CLIP due to numerical stability issues with the softmax function. Let's break this down:

<br>

#### 1. The Softmax Function

The softmax function converts raw logits into a probability distribution. For input vector x:

$$ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}} $$

Where:
- $x_i$ is the current logit (similarity score)
- $e^x$ is the exponential function (e ≈ 2.71828)
- $\sum_{j=1}^n e^{x_j}$ is the sum of all exponentials in the sequence

<br>

**Problem: Exponential Growth**
- When x is large, exp(x) grows extremely fast
- For example:
  - exp(10) ≈ 22,026
  - exp(50) ≈ 5.18 × 10²¹
  - exp(100) ≈ 2.69 × 10⁴³
- A 32-bit float can only represent numbers up to ~3.4 × 10³⁸
- This means exp(89) is already too large to represent!

<br>

---

##### In CLIP's Context

When computing similarity scores:
1. Large dot products between embeddings can produce large numbers
2. These large numbers get exponentially larger through softmax
3. This can lead to:
   - Numerical overflow (numbers too large to represent)
   - Loss of precision
   - NaN (Not a Number) errors

<br>

---

##### The Solution: Log-Space Calculations

Looking at the similarity matrix in the image, we need to compute stable similarity scores between image features (I₁, I₂, I₃, ..., Iₙ) and text features (T₁, T₂, T₃, ..., Tₙ). To prevent numerical instability:

$$ \text{log\_softmax}(x_i) = x_i - \log(\sum_{j=1}^n e^{x_j}) $$

This first equation:
- Takes the original similarity score ($x_i$)
- Subtracts the log of the sum of exponentials
- Helps avoid overflow by working in log space

We can make this even more stable by rewriting it as:

$$ \text{log\_softmax}(x_i) = x_i - (\max(x) + \log(\sum_{j=1}^n e^{x_j - \max(x)})) $$

This improved version:
- First finds the maximum value in the sequence ($\max(x)$)
- Subtracts this maximum from each score before exponentiating
- Makes all values ≤ 0 before exp(), preventing overflow
- Adds back the maximum at the end to maintain correctness

For example, in our similarity matrix:
- If I₂·T₂ = 100 (maximum score in column T₂)
- Then we subtract 100 from all scores in that column
- Now exp(score - 100) will be small and manageable
- The relative relationships between scores are preserved

This stabilization is crucial because:
1. Each row in the matrix needs one high value (matching pair)
2. Each column needs one high value (matching pair)
3. All other values should be relatively small
4. We need to compute this stably for both image→text and text→image directions

<br>

#### 2. Computational Challenges in CLIP

Looking at the similarity matrix in the image, CLIP faces significant computational overhead due to its bidirectional nature:

##### Asymmetric Computation Requirements

Based on the SigLip paper, due to the asymmetry of the softmax loss, normalization is performed twice: across images and across texts:

1. **Row-wise Softmax (Image → Text)**
   - For each image embedding (I₁, I₂, ..., Iₙ):
     - Must compute softmax across all text embeddings
     - Example: For I₃ row, compute softmax over [I₃·T₁, I₃·T₂, I₃·T₃, ..., I₃·Tₙ]
   - Total: N softmax computations (one per row)

2. **Column-wise Softmax (Text → Image)**
   - For each text embedding (T₁, T₂, ..., Tₙ):
     - Must compute softmax across all image embeddings
     - Example: For T₂ column (purple dotted box), compute softmax over [I₁·T₂, I₂·T₂, I₃·T₂, ..., Iₙ·T₂]
   - Total: N softmax computations (one per column)

<br>

---

##### Why This Is Expensive

1. **Scaling Issues**
   - With batch size N, need 2N softmax computations
   - Each softmax requires:
     - N exponential operations
     - N additions for the denominator
     - N divisions for normalization

2. **Memory Requirements**
   - Must store full N×N similarity matrix
   - Keeps all intermediate values for gradient computation
   - Memory grows quadratically with batch size

3. **Computational Complexity**
   - Total operations: O(N²) for matrix creation
   - Plus O(N²) for bidirectional softmax
   - Makes large batch training challenging

<br>

---

##### **Solution** to Computational Challenges: Sigmoid Loss

![sigmoid-loss](sigmoid-loss.png)

One solution to CLIP's computational overhead is replacing cross-entropy loss with sigmoid loss:

###### Why Replace Cross-Entropy Loss?

1. **Problem with Cross-Entropy**
   - Requires softmax computation first
   - As we saw, needs both row-wise and column-wise softmax
   - Computationally expensive at O(N²)
   - Memory intensive for large batches

2. **Sigmoid Loss Alternative**
   - Operates on individual similarity scores
   - No need for softmax normalization
   - Can process each I·T pair independently
   - Reduces memory and computation requirements

###### How Sigmoid Loss Works

Instead of normalizing across rows and columns:
1. Each similarity score I·T is passed through sigmoid function:
   $$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

2. Binary cross-entropy is applied to each score:
   - Matching pairs (diagonal) should output 1
   - Non-matching pairs should output 0

Benefits:
- Processes each element independently
- No need for expensive normalization
- Memory efficient: can process in smaller chunks
- Still maintains contrastive learning objective

Comparison of Complexity:
```
Cross-Entropy + Softmax:
- Must process entire N×N matrix at once
- O(N²) memory and computation

Sigmoid Loss:
- Can process elements independently
- O(1) per element
- Easily parallelizable
```

This approach trades off some model accuracy for significant computational efficiency, making it more practical for large-scale training. Below is a larger overview of how sigmoid loss works.

<br>

---

##### SigLIP's Sigmoid-based Solution

Looking at the similarity matrix in the image, SigLIP (Sigmoid Loss for Image-Text Pairs) proposes a more efficient approach:

###### How SigLIP Works

1. **Direct Similarity Processing**
   - For each cell in the similarity matrix (I·T pairs):
     - I₁·T₁, I₁·T₂, I₁·T₃, ... (first row)
     - I₂·T₁, I₂·T₂, I₂·T₃, ... (second row)
     - And so on...
   - Each similarity score is processed independently through sigmoid

2. **Binary Labels**
   - Matching pairs (blue squares in diagram):
     - I₁·T₁ should be 1
     - I₂·T₂ should be 1
     - I₃·T₃ should be 1
   - Non-matching pairs (all other cells):
     - Should be 0
     - Example: I₁·T₂, I₁·T₃, I₂·T₁, etc.

3. **Loss Computation**
   $$ L_{SigLIP} = -\frac{1}{N}\sum_{i=1}^N [\log(\sigma(s_{ii})) + \sum_{j\neq i}\log(1-\sigma(s_{ij}))] $$
   
   Where:
   - $s_{ij}$ is the similarity score between Iᵢ and Tⱼ
   - $\sigma$ is the sigmoid function
   - $s_{ii}$ represents matching pairs (diagonal)
   - $s_{ij}$ (i≠j) represents non-matching pairs

4. **Key Benefits**
   - No need for row/column normalization
   - Can process each cell independently
   - Easily parallelizable
   - Memory efficient: can process subsets of matrix

5. **Comparison to Original CLIP**
   - Original CLIP (looking at T₂ column in purple):
     - Needs entire column to compute softmax
     - Must normalize across all image pairs
   - SigLIP:
     - Can process I₁·T₂, I₂·T₂, I₃·T₂ independently
     - No need to wait for full column computation

This approach maintains the contrastive learning objective (matching correct image-text pairs) while being computationally more efficient and numerically stable.

<br>

---

##### Example of SigLIP Processing

Let's look at a small 3×3 example from our similarity matrix:

1. **Raw Similarity Scores**
   ```
   Similarity Matrix (s_ij):
   [Dog Image I₁]    [0.8  0.2  0.1]  → [T₁: "aussie pup"]
   [Car Image I₂]    [0.1  0.9  0.2]  → [T₂: "red car"]
   [Sky Image I₃]    [0.2  0.1  0.7]  → [T₃: "blue sky"]
   ```

2. **Apply Sigmoid to Each Score**
   $$ \sigma(x) = \frac{1}{1 + e^{-x}} $$
   
   ```
   After Sigmoid:
   [Dog Image I₁]    [0.69  0.45  0.42]  → [T₁: "aussie pup"]
   [Car Image I₂]    [0.42  0.71  0.45]  → [T₂: "red car"]
   [Sky Image I₃]    [0.45  0.42  0.67]  → [T₃: "blue sky"]
   ```

3. **Binary Label Classification**
   ```
   Target Matrix (1 for matches, 0 for non-matches):
   [Dog Image I₁]    [1  0  0]  → [T₁: "aussie pup"]
   [Car Image I₂]    [0  1  0]  → [T₂: "red car"]
   [Sky Image I₃]    [0  0  1]  → [T₃: "blue sky"]
   ```

4. **Loss Computation**
   For diagonal elements (matches):
   - I₁·T₁: -log(0.69)
   - I₂·T₂: -log(0.71)
   - I₃·T₃: -log(0.67)

   For off-diagonal elements (non-matches):
   - I₁·T₂: -log(1 - 0.45)
   - I₁·T₃: -log(1 - 0.42)
   etc.

   Final loss is the average of all these terms.

Key Advantages Shown in Example:
1. Each cell processed independently
2. No need to normalize rows or columns
3. Can compute loss for any subset of pairs
4. Numerically stable (all values between 0 and 1)

<br>

---

<br>

### Vision Transformers (general)

Using the article "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (https://arxiv.org/abs/2010.11929), we will dive deep on how it works for the 
Contrastive Vision Encoder. The transformer model is a sequence-to-sequence model that is fed with a sequence of embeddings that outputs a sequence of contextualized embeddings.

**NOTE:** See "Actual SigLip Encoder Diagram" below to see the actual diagram for the encoder that contains the connections.
![vision-transformer-diagram](vision-transformer-diagram.png)

What the transformer looks like: <br>
![transformer_architecture](<transformer_architecture.png>)

Looking at the vision transformer diagram, we can see the process flows from bottom to top:
1. Start with an input IMAGE (4x4 grid shown in diagram)
2. Transform it into EMBEDDINGS OF PATCHES
3. Add positional encodings (POS. ENC.)
4. Process through TRANSFORMER
5. Output CONTEXTUALIZED EMBEDDINGS

Let's dive deeper into each step:

1. **Image Patching & Convolution**
   - Starting with the IMAGE in the diagram (4x4 grid at bottom)
   - The image is divided into non-overlapping patches
   - In the diagram, we see a 4x4 grid creating 16 patches (numbered 1-16)
   ```python
   # Example of patch extraction using convolution
   class PatchExtractor(nn.Module):
       def __init__(self, patch_size):
           super().__init__()
           self.conv = nn.Conv2d(
               in_channels=3,  # RGB image
               out_channels=768,  # Embedding dimension
               kernel_size=patch_size,
               stride=patch_size
           )
           
       def forward(self, x):  # x: (B, 3, H, W)
           # Convert image to patches using convolution
           patches = self.conv(x)  # (B, 768, H/patch_size, W/patch_size)
           # Reshape to sequence of patches
           patches = patches.flatten(2).transpose(1, 2)  # (B, N, 768)
           return patches
   ```

2. **Embedding Creation (EMBEDDINGS OF PATCHES in diagram)**
   - Each patch is processed through convolution + flatten operation
   - This flattens each 2D patch (e.g. 16x16 pixels with 3 RGB channels = 768 values) into a single row vector by concatenating all pixel values in sequence
   - Results in sequence of embeddings (shown as 1-16 in bottom row)
   ```python
   # Converting patches to embeddings
   patch_dim = patch_size * patch_size * 3  # 16x16x3 = 768 for 16x16 patches
   embedding_dim = 768
   patch_to_embedding = nn.Linear(patch_dim, embedding_dim)
   ```

3. **Position Encoding Addition (POS. ENC. in diagram)**
   - The diagram shows "POS. ENC." row with numbers 1-16
   - Position encodings are vectors that encode the spatial location of each patch
   - When added to patch embeddings, they help the transformer understand where each patch was located in the original image
   - Without position encodings, the transformer would lose all spatial information since patches are processed as a flat sequence
   - The addition is element-wise: each position encoding vector is added to its corresponding patch embedding vector
   ```python
   class PositionalEncoding(nn.Module):
       def __init__(self, d_model, max_patches):
           super().__init__()
           # Create learnable position embeddings
           self.pos_embedding = nn.Parameter(
               torch.randn(1, max_patches, d_model)
           )
           
       def forward(self, x):
           # x: patch embeddings (B, N, D)
           return x + self.pos_embedding  # Add positional information
   ```

4. **Transformer Processing**
   - The large "TRANSFORMER" box in diagram processes the sequence
   - Each patch embedding can interact with and incorporate information from all other patch embeddings through self-attention:
     - For example, if patch 1 shows part of a dog's ear and patch 8 shows part of the tail:
       - The self-attention mechanism allows patch 1 to look at patch 8 and all other patches
       - This helps patch 1's embedding understand it's part of a larger dog shape
       - Similarly, patch 8 can look back at patch 1 and other patches
     - This all-to-all interaction between patches is what the large "TRANSFORMER" box represents
   - Uses standard transformer encoder architecture with modifications:
   ```python
   class TransformerEncoder(nn.Module):
       def __init__(self, dim, depth, heads):
           super().__init__()
           self.layers = nn.ModuleList([])
           for _ in range(depth):
               self.layers.append(nn.ModuleList([
                   PreNorm(dim, SelfAttention(dim, heads)),
                   PreNorm(dim, FeedForward(dim))
               ]))
           
       def forward(self, x):
           # x: sequence of patch embeddings + positions
           for attn, ff in self.layers:
               # Self-attention allows patches to interact
               x = attn(x) + x  # Residual connection
               x = ff(x) + x    # MLP processing
           return x
   ```

5. **Final Contextualized Embeddings**
   - Output shown at top of diagram as "CONTEXTUALIZED EMBEDDINGS"
   - Each position (1-16) now contains information from other patches
   - The embeddings maintain spatial relationships but are enriched with context
   - These embeddings can be used for downstream tasks
   - Unlike language models that process tokens sequentially (word1 → word2 → word3),
     Vision Transformers process all patches simultaneously:
     ```
     Language Model (Sequential):
     "The" → "cat" → "sits" → "on" → "mat"
     (Each word depends on previous words)
     
     Vision Transformer (Parallel):
     Patch1 ↔ Patch2 ↔ Patch3 ↔ ... ↔ Patch16
     (All patches attend to each other simultaneously)
     ```
   - Example with a light source in an image:
     - Consider patches showing a lamp illuminating a room
     - In the diagram's 4x4 grid, if patch 6 contains a bright lamp:
       1. Initial patch embedding just sees local brightness
       2. Through transformer's self-attention (big "TRANSFORMER" box):
          - Patch 6 (lamp) influences all other patches
          - Surrounding patches (5,7,2,10) learn light falloff
          - Distant patches learn shadow patterns
       3. Final contextualized embeddings (top row) capture:
          - Local features (brightness, color)
          - Global context (lighting effects, shadows)
          - Spatial relationships (light direction, distance)
   - This parallel processing of all patches enables the model to:
     - Capture complex spatial relationships
     - Model long-range dependencies across the image
     - Learn global visual patterns

The diagram effectively shows the transformation from:
- 2D spatial image (bottom 4x4 grid)
- To 1D sequence of patch embeddings (EMBEDDINGS OF PATCHES)
- Enhanced with position information (POS. ENC.)
- Processed through transformer for context (TRANSFORMER)
- Resulting in final contextualized representation (top row)

This process allows the model to:
1. Break down spatial image data into processable sequences
2. Maintain spatial relationships through position encodings
3. Enable global reasoning through transformer's self-attention
4. Create rich, context-aware representations of image patches

<br>

---

#### Actual SigLip Encoder Diagram

![siglip-encoder](siglip-encoder.png)

The SigLIP encoder diagram shows the detailed architecture of the vision encoder component. Let's break down its key components and their roles:

1. **Input: Patch Embeddings with Positional Encodings**
   - Bottom of diagram shows input embeddings
   - Each embedding represents a patch of the image
   - Position encodings are added to maintain spatial information

2. **Layer Architecture (repeated Nx times)**
   Each encoder layer contains:
   
   a) **First Branch**
   - Layer Normalization (yellow box)
   - Self-Attention mechanism
   - Residual connection (+)
   
   b) **Second Branch**
   - Layer Normalization (yellow box)
   - MLP (Multi-Layer Perceptron)
   - Residual connection (+)

3. **Layer Normalization**
   - Shown as yellow boxes in the diagram
   - Applied before attention and MLP (pre-norm design)
   - Helps stabilize training by normalizing activations
   - Prevents internal covariate shift

4. **Self-Attention**
   - Allows each patch to attend to all other patches
   - Computes attention scores between patches
   - Helps build global understanding of image
   - Key component for capturing long-range dependencies

5. **MLP (Multi-Layer Perceptron)**
   - Two-layer feed-forward network
   - Projects to higher dimension then back
   - Adds non-linearity through GELU activation
   - Increases model capacity

   **GELU Activation's Importance:**
     - **Non-linearity Introduction**
       - Without GELU, MLP would just be composed linear transformations
       - GELU enables learning of complex, non-linear patterns
       - Critical for modeling sophisticated visual relationships

     - **Modern Transformer Choice**
       - Standard activation in modern transformer architectures
       - Performs better than traditional ReLU or tanh
       - Combines benefits of ReLU with smoother gradient properties

     - **Mathematical Properties**
       - Defined as: GELU(x) = x * Φ(x)
       - Where Φ(x) is the cumulative distribution function of standard normal
       - Provides smooth activation with good gradient characteristics
       - Can be efficiently approximated using tanh-based implementation

     - **Processing Pipeline**
       ```
       Input Features → Linear Expansion → GELU → Linear Projection
       [dim] → [4*dim] → [4*dim] → [dim]
       ```
       - First expands feature space for richer representations
       - Applies non-linear transformation via GELU
       - Projects back to required dimension
       - Maintains dimensionality compatibility with residual connections

     - **Training Benefits**
       - Smooth gradient flow for stable training
       - Probabilistic interpretation helps with regularization
       - Better training dynamics in deep transformer architectures
       - Efficient computation with tanh approximation

6. **Residual Connections**
   - Shown as "+" in the diagram
   - Skip connections that add input to output
   - Help with gradient flow during training
   - Allow model to preserve low-level features

7. **Design Philosophy**
   - Pre-norm architecture (LayerNorm before attention/MLP)
   - Dual processing streams (attention and MLP)
   - Deep network with repeated layers
   - Heavy use of residual connections

The key innovation in SigLIP is not in this encoder structure (which follows standard transformer design), but rather in how the embeddings it produces are used in the contrastive learning setup with sigmoid loss.

<br><br>

# Random Teachings

## Normalization

### Linear Layer and Layer Normalization Example

Layer normalization is a crucial technique used in transformers to stabilize and accelerate training. Let's understand how it works with a concrete example:

![layer-normalization-diagram](layer-normalization.png)

Looking at the diagram, we can break down the process into two key parts:

1. **Linear Layer L₁ (Left Side)**:
   - Input features: [1.1, 2.0, 1.5, 2.1] (in_features = 4)
   - Output features: [1.6, 2.7, 1.1, 3.1] (out_features = 4)
   - Each neuron has:
     - Weight vector (w₁, w₂, w₃, w₄) matching input dimension
     - Bias term (b)
   - Operation: output = input · weights + bias
   ```python
   # Example of one neuron's computation
   # For first output 1.6:
   weights = [w₁₁, w₁₂, w₁₃, w₁₄]  # First neuron's weights
   output₁ = (1.1×w₁₁ + 2.0×w₁₂ + 1.5×w₁₃ + 2.1×w₁₄) + b₁ = 1.6
   ```

2. **Linear Layer L₄ (Right Side)**:
   - Input features: [1.6, 2.7, 1.1, 3.1] (in_features = 4)
   - Output features: [x', x'] (out_features = 2)
   - Each neuron processes all 4 input features:
     ```python
     # For first output neuron
     weights₁ = [w₁₁, w₁₂, w₁₃, w₁₄]
     output₁ = (1.6×w₁₁ + 2.7×w₁₂ + 1.1×w₁₃ + 3.1×w₁₄) + b₁
     
     # For second output neuron
     weights₂ = [w₂₁, w₂₂, w₂₃, w₂₄]
     output₂ = (1.6×w₂₁ + 2.7×w₂₂ + 1.1×w₂₃ + 3.1×w₂₄) + b₂
     ```

### Why Layer Normalization?

The purpose of layer normalization is to:
1. Stabilize the distribution of activations
2. Reduce internal covariate shift
3. Allow for faster training

In the context of PaLI-Gemma and vision transformers:
- Each transformer layer's output is normalized
- This helps maintain stable gradients through the deep network
- Particularly important when processing variable-length sequences (like image patches)

The mathematical process:
1. Calculate mean (μ) and standard deviation (σ) across features
2. Normalize: (x - μ) / σ
3. Apply learnable scale (γ) and shift (β) parameters

This normalization process helps ensure that the network can learn effectively regardless of the scale or distribution of its inputs, which is crucial for both the vision and language components of the model.

<br>

---

### The Problem of Covariate Shift

![problem-covariate-shift](problem-covariate-shift.png)

Looking at the diagram, we can see how covariate shift creates training instability:

1. **Input Distribution Changes**:
   ```
   Batch 1: x = [1.1, 2.0, 1.5, 2.1]  → output = [1.6, 2.7, 1.1, 3.1]
   Batch 2: x = [11.3, 21.7, 31.1, 25.9] → output changes drastically!
   ```
   - When input features change significantly between batches
   - Each layer's output distribution shifts dramatically
   - This cascades through the network

2. **Chain Reaction of Changes**:
   ```
   Input Changes → Layer Output Changes → Loss Changes → Gradient Changes → Weight Updates Unstable → Network Learns Slowly
   ```
   For example, in our diagram:
   - If input vector changes from [1.1, 2.0, 1.5, 2.1] to much larger values
   - L₁'s output will shift from [1.6, 2.7, 1.1, 3.1] to very different values
   - This affects L₄'s computation and final output
   - Loss computation becomes unstable
   - Results in erratic gradient updates

3. **Impact on Training**:
   - Network has to constantly adapt to new distributions
   - Learning becomes inefficient and slow
   - Model might never converge properly
   ```python
   # Example of how distribution shift affects each layer
   class Layer:
       def forward(self, x):
           # Distribution of x keeps changing dramatically
           output = self.weights @ x + self.bias
           # Output distribution also changes dramatically
           # Next layer receives unstable input
           return output
   ```

4. **Why This Is Particularly Bad for Deep Networks**:
   - Changes compound through layers:
     ```
     Layer 1 shift → Layer 2 bigger shift → Layer 3 even bigger shift → ...
     ```
   - In our diagram:
     - L₁'s output distribution shift
     - Makes L₄'s task even harder
     - Each layer amplifies the instability

5. **Solution Through Layer Normalization**:
   ```python
   # Before normalization
   x = [11.3, 21.7, 31.1, 25.9]  # Large, varying values
   
   # After normalization
   mean = 22.5
   std = 8.6
   x_norm = [(11.3 - 22.5)/8.6, (21.7 - 22.5)/8.6, ...]
   # Results in values centered around 0 with unit variance
   ```
   
   This ensures:
   - Each layer receives inputs with stable statistics
   - Gradients flow more smoothly
   - Training can proceed efficiently

By normalizing the activations at each layer, we prevent the cascade of distribution shifts that would otherwise make training difficult or impossible. This is especially important in transformers where we're processing sequences of varying lengths and distributions.

<br>

---

#### Batch Normalization: A Solution to Covariate Shifts

According to the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015), batch normalization was introduced as the first major solution to the covariate shift problem.

![batch-norm-example1](batch-norm-example1.png)

Looking at the diagram with our batch B of images (cat, dog, zebra, etc.), batch normalization works as follows:

1. **Input Structure**:
   ```
   Batch B = [
       cat_features   = [0.8, 0.3, 0.9, ...],  # Feature dimension
       dog_features   = [0.2, 0.7, 0.4, ...],
       zebra_features = [0.5, 0.1, 0.6, ...],
       ...
   ]
   ```
   Each row represents an item (image), and each column represents a feature dimension.

2. **Normalization Process**:
   For each feature dimension j:
   ```python
   # Calculate statistics across batch dimension
   μⱼ = (1/m) * Σᵢ xᵢⱼ  # mean of feature j across batch
   σ²ⱼ = (1/m) * Σᵢ (xᵢⱼ - μⱼ)²  # variance of feature j across batch
   
   # Normalize each feature
   x̂ᵢⱼ = (xᵢⱼ - μⱼ) / √(σ²ⱼ + ε)
   ```

   For example, if we have feature dimension j=1:
   ```python
   # Before normalization (feature 1 across batch)
   x₁ = [0.8,    # from cat
         0.2,    # from dog
         0.5]    # from zebra
   
   μ₁ = 0.5     # mean
   σ₁ = 0.3     # std dev
   
   # After normalization
   x̂₁ = [1.0,   # (0.8 - 0.5)/0.3
        -1.0,    # (0.2 - 0.5)/0.3
         0.0]    # (0.5 - 0.5)/0.3
   ```

3. **Why This Helps**:
   - Different images (cat vs zebra) have very different feature distributions:
     ```
     Cat:   High values in light fur regions
     Zebra: High contrast between black/white stripes
     ```
   - After normalization:
     - All features follow N(0,1) distribution
     - Model sees consistent statistics regardless of input
     - Reduces oscillations in gradients and loss

4. **The Problem with Batch Norm**:
   ```
   Batch Statistics Mixing:
   
   Feature j=1:      Feature j=2:      Feature j=3:
   cat:    0.8      cat:    0.3      cat:    0.9
   dog:    0.2  →   dog:    0.7  →   dog:    0.4   → μ, σ
   zebra:  0.5      zebra:  0.1      zebra:  0.6
   ↓                ↓                 ↓
   μ₁, σ₁           μ₂, σ₂            μ₃, σ₃
   ```
   - Statistics (μ, σ) depend on other items in batch
   - So if we have small batches it creates unstable statistics
   - And if we have different batch compositions, it creates different normalizations

<br>

---

#### Layer Normalization: A Better Solution

Layer normalization improves upon batch normalization by computing statistics independently for each item:

1. **Independent Normalization**:
   ```python
   # For each item i (e.g., cat image):
   μᵢ = (1/d) * Σⱼ xᵢⱼ  # mean across features
   σ²ᵢ = (1/d) * Σⱼ (xᵢⱼ - μᵢ)²  # variance across features
   
   # Normalize each feature of item i
   x̂ᵢⱼ = (xᵢⱼ - μᵢ) / √(σ²ᵢ + ε)
   ```

2. **Why It's Better**:
   ```
   Layer Norm (each item normalized independently):
   
   Cat:   [0.8, 0.3, 0.9] → μ_cat, σ_cat   → normalized_cat
   Dog:   [0.2, 0.7, 0.4] → μ_dog, σ_dog   → normalized_dog
   Zebra: [0.5, 0.1, 0.6] → μ_zebra, σ_zebra → normalized_zebra
   ```
   - Each item's normalization is independent
   - No batch size dependency
   - More stable training
   - Particularly good for transformers

3. **Mathematical Formulation**:
   $$ \text{LayerNorm}(x_i) = \gamma \odot \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} + \beta $$
   
   Where:
   - $x_i$ is the input vector for item i
   - $\mu_i$ is the mean of features for item i
   - $\sigma_i^2$ is the variance of features for item i
   - $\gamma, \beta$ are learnable parameters
   - $\epsilon$ is a small constant for numerical stability

This approach has become the standard in transformer architectures because:
1. It's independent of batch size
2. Each item's normalization is self-contained
3. It works well with variable-length sequences
4. Training is more stable and converges faster

<br>

---

#### Key Difference: Normalization Dimensions

Let's understand the fundamental difference between batch and layer normalization:

1. **Batch Normalization (Along Batch Dimension)**:
   ```
   Batch of Images:
   cat:   [0.8, 0.3, 0.9]  ↓  Calculate stats down this column
   dog:   [0.2, 0.7, 0.4]  ↓  for each feature separately
   zebra: [0.5, 0.1, 0.6]  ↓
   
   For feature 1: mean([0.8, 0.2, 0.5])
   For feature 2: mean([0.3, 0.7, 0.1])
   For feature 3: mean([0.9, 0.4, 0.6])
   ```
   - Calculates statistics DOWN each feature column
   - Each feature gets normalized using other images' values
   - Problem: Depends on what other images are in the batch

2. **Layer Normalization (Along Feature Dimension)**:
   ```
   Single Images:
   cat:   [0.8, 0.3, 0.9] → Calculate stats across this row
   dog:   [0.2, 0.7, 0.4] → Calculate stats across this row
   zebra: [0.5, 0.1, 0.6] → Calculate stats across this row
   
   For cat:   mean([0.8, 0.3, 0.9])
   For dog:   mean([0.2, 0.7, 0.4])
   For zebra: mean([0.5, 0.1, 0.6])
   ```
   - Calculates statistics ACROSS each image's features
   - Each image normalized independently
   - Better: No dependency on batch composition

This dimensional difference is why layer normalization is more stable - each item only depends on its own features, not on what else is in the batch.

<br><br>

---

## Multi-Head Attention

![multi-head-attention](multi-head-attention.png)

Multi-head attention is a key component in both Vision Transformers and Language Models, but they use it in slightly different ways. Let's explore both:

---

### Vision Transformers: Contextualizing Image Patches

In Vision Transformers (top part of diagram):
- Each row represents a patch extracted from the input image
- Each patch is a vector with 1024 dimensions, created by flattening a group of pixels
- Input shape: 4 patches × 1024 dimensions
- Output maintains same shape: 4 patches × 1024 dimensions

The multi-head attention mechanism contextualizes these patches by allowing each patch to attend to all other patches in the sequence. For example:
- Patch 1 can look at patches 2, 3, and 4
- Patch 2 can look at patches 1, 3, and 4
- And so on...

This creates rich contextual representations where each output patch now contains information about:
- Its own local features (from the original pixels)
- Global context (from attending to other patches)
- Spatial relationships (through position embeddings)

### Language Models: Causal Self-Attention

In Language Models (bottom part of diagram):
- Input is a sequence of tokens: ["I", "love", "pepperoni", "pizza"]
- Each token is embedded into a 1024-dimensional vector
- Input shape: 4 tokens × 1024 dimensions
- Output maintains same shape: 4 tokens × 1024 dimensions

Key difference: Language models use causal attention (also known as masked self-attention):
- Each token can only attend to itself and previous tokens
- This creates an autoregressive property
- Example in the diagram:
  - "I" only sees itself
  - "love" sees ["I", "love"]
  - "pepperoni" sees ["I", "love", "pepperoni"]
  - "pizza" sees ["I", "love", "pepperoni", "pizza"]

The output sequence demonstrates this contextualization:
- First token: "I" (only self-context)
- Second token: "I love" (includes previous context)
- Third token: "I love pepperoni" (includes all previous context)
- Fourth token: "I love pepperoni pizza" (full context)

<br>

---

### Parallel Training: A Powerful Feature

The power of transformer architecture lies in its ability to process sequences in parallel:

1. **Parallel Processing**
   - Instead of generating one token at a time
   - All positions are processed simultaneously
   - Multiple attention heads work in parallel
   - Each head can focus on different aspects of the sequence

2. **Training Process**
   ```
   Input:  ["I", "love", "pepperoni", "pizza"]
   Labels: ["love", "pepperoni", "pizza", <end>]
   
   For each position:
   - "I" should predict "love"
   - "I love" should predict "pepperoni"
   - "I love pepperoni" should predict "pizza"
   ```

3. **Parallel Loss Calculation**
   - Loss is computed for all positions simultaneously
   - Backpropagation updates all weights in parallel
   - Model learns to predict next tokens based on all previous contexts
   - This parallel computation is vastly more efficient than sequential processing

4. **Why This Is Powerful**
   - Training is much faster than sequential models
   - Can learn complex patterns across different sequence lengths
   - Multiple attention heads capture different types of relationships
   - Parallel processing enables training on massive datasets

This parallel nature, combined with the ability to capture long-range dependencies, makes transformers extremely effective for both vision and language tasks, despite their different attention patterns (full attention for vision vs. causal attention for language).

<br><br>

---

### How it Works

#### Step 1: From X to Q, K, V Transformations

![step1-qkv](step1-qkv.png)

The first step in multi-head attention is transforming the input sequence X into three different representations: Query (Q), Key (K), and Value (V) matrices. Looking at the diagram, we can see how this transformation process works through matrix multiplication with learned parameter matrices Wq, Wk, and Wv.

---

##### Matrix Multiplication Process

Input sequence X has shape (4, 1024):
- 4 represents sequence length (number of tokens/patches)
- 1024 represents hidden_size (embedding dimension)

Parameter matrices Wq, Wk, Wv each have shape (1024, 1024):
- First 1024 matches input embedding dimension
- Second 1024 is split into 8 heads × 128 dimensions
- Total size remains 1024 (8 * 128 = 1024)

The matrix multiplication works as follows:
```
Input shape:     (4, 1024)
Parameter shape: (1024, 1024) = (1024, 8 * 128)
Output shape:    (4, 8, 128)

Why? Inner dimensions cancel out (1024),
     Outer dimensions remain (4 and 8*128)
```

<br>

---

##### Visualizing the Transformation

Looking at the bottom part of the diagram above, we can visualize the matrix multiplication process in detail:

1. **Input Sequence (X)**:
   - Each row represents a token/patch with 1024 dimensions
   - Shown in the diagram as a 4×1024 matrix
   - For language models, this could be:
     ```
     Row 1: "I"      → [0.1, 0.2, ..., 0.8]  (1024 values)
     Row 2: "love"   → [0.3, 0.7, ..., 0.4]  (1024 values)
     Row 3: "pizza"  → [0.5, 0.1, ..., 0.9]  (1024 values)
     Row 4: "!"      → [0.2, 0.6, ..., 0.3]  (1024 values)
     ```

2. **Parameter Matrix (Wq/Wk/Wv)**:
   - Visualized as a large 1024×1024 matrix
   - Each row (1024 rows total) is made up of smaller vectors
   - Each vector is split into 8 groups (heads), each with 128 dimensions
   - Overall size remains 1024×1024, but vectors are organized into 8 heads
   - Visualized as columns in the diagram:
     ```
     Head 1: Processes dimensions 1-128
     Head 2: Processes dimensions 129-256
     Head 3: Processes dimensions 257-384
     ...and so on
     ```

3. **Output Matrix (Q/K/V)**:
   - The output is a matrix where each token is split into multiple subgroups
   - Size of (4, 8, 128): 4 rows split into 8 groups of smaller embeddings
   - Each head (column) is a sequence 
   - Each smaller embedding is made up of 128 dimensions
   - Each token now has 8 different representations focusing on different aspects:
     ```
     Token "bank":
     Head 1 (dims 1-128):   [0.1, ..., 0.4]  → Financial aspects
     Head 2 (dims 129-256): [0.7, ..., 0.2]  → Geographic aspects
     Head 3 (dims 257-384): [0.3, ..., 0.8]  → Action aspects
     ...and so on
     ```

<br>

---

##### Importance of Matrix Multiplication for Multi-Head Attention

This transformation is crucial because it enables tokens/patches to relate to each other in multiple ways:

1. **Column-wise Processing**:
   - Each head (column) in the diagram processes a specific subset of dimensions
   - Looking at the diagram's bottom section:
     ```
     Head 1 (Column 1): Processes first 128 dimensions of all tokens
     ↓
     Token 1's first 128 dims
     Token 2's first 128 dims
     Token 3's first 128 dims
     Token 4's first 128 dims
     ```

2. **Parallel Feature Processing**:
   - The diagram shows how all heads work in parallel
   - Each head can specialize in different patterns:
     ```
     For an image of a cat playing with yarn:
     Head 1: Focuses on shape features
     Head 2: Focuses on color patterns
     Head 3: Focuses on motion aspects
     Head 4: Focuses on object relationships
     ```

3. **Rich Token Relationships**:
   For example, with the sentence "The bank by the river bank":
   ```
   Without multi-head:
   - Single 1024-dim representation
   - One way to relate "bank" to other words
   
   With multi-head (as shown in diagram):
   Head 1: [Financial context]    → Processes "bank" as institution
   Head 2: [Geographic context]   → Processes "bank" as riverside
   Head 3: [Positional context]   → Processes word order
   Head 4: [Semantic context]     → Processes meaning
   ...and so on
   ```

4. **Parallel Processing Benefits**:
   - Looking at the diagram's structure:
     ```
     All heads process simultaneously:
     Head 1: [128-dim] → Financial aspects
     Head 2: [128-dim] → Geographic aspects
     Head 3: [128-dim] → Syntactic aspects
     Head 4: [128-dim] → Semantic aspects
     Head 5: [128-dim] → Contextual aspects
     Head 6: [128-dim] → Relational aspects
     Head 7: [128-dim] → Temporal aspects
     Head 8: [128-dim] → Structural aspects
     ```

<br><br>

---

#### Step 2: Treat Each Head Independently!

![step2-transpose](step2-transpose.png)

After creating our Q, K, and V matrices in Step 1, we need to reorganize them to enable parallel processing across attention heads. Looking at the diagram, we can see this reorganization through transposition.

##### Initial Matrix Structure (Left Side)
- Starting shape: (4, 8, 128)
  - 4: sequence length (number of tokens/patches)
  - 8: number of attention heads
  - 128: dimensions per head
- Each head is composed of 128 dimensions
- Each token's embedding is split across all heads

---

##### Transposition Process (Arrow in Diagram)
```python
# Before transpose:
shape = (4, 8, 128)  # [sequence_length, n_heads, head_dim]

# After transpose of first two dimensions:
shape = (8, 4, 128)  # [n_heads, sequence_length, head_dim]
```

---

##### Resulting Structure (Right Side)
The transposed structure creates 8 independent sequences, where:
- Each sequence represents one attention head
- Each sequence contains all 4 tokens
- Each token has 128 dimensions (dedicated to that head)

For example:
```
Head 1: [
    Token1[128d], Token2[128d], Token3[128d], Token4[128d]
]
Head 2: [
    Token1[128d], Token2[128d], Token3[128d], Token4[128d]
]
...and so on for all 8 heads
```

---

##### Why This Transformation Matters

1. **Independent Processing**:
   - Each head can now process its sequence independently
   - As shown in the diagram's right side, each head has its own complete view of the sequence
   - But each token is represented by only the dimensions relevant to that head

2. **Parallel Computation**:
   - The diagram shows how we split into 8 parallel sequences
   - Each sequence can be processed simultaneously
   - No need for communication between heads during attention computation

3. **Specialized Learning**:
   ```
   Head 1: Processes [Token1₁₂₈, Token2₁₂₈, Token3₁₂₈, Token4₁₂₈]
          ↓ (learns one type of relationship)
   Head 2: Processes [Token1₁₂₈, Token2₁₂₈, Token3₁₂₈, Token4₁₂₈]
          ↓ (learns different type of relationship)
   ...and so on
   ```
   - Each head learns to relate tokens differently
   - Uses its dedicated 128 dimensions to capture specific patterns

4. **Efficiency Benefits**:
   - As shown by the diagram's structure:
     ```
     Before: Need to coordinate across 8 heads
     After:  8 independent sequences that can run in parallel
     ```
   - This parallelization significantly speeds up computation
   - Each head can specialize without interference from others

This transformation is crucial for enabling the parallel, multi-headed nature of attention mechanisms, allowing each head to develop its own specialized way of relating tokens or patches while maintaining computational efficiency.

<br><br>

---

#### Step 3: Calculate the Attention for Each Head in Parallel

![step3-calculate-attention](step3-calculate-attention.png)

After transposing our matrices in Step 2, we now calculate attention scores for each head independently. Looking at the diagram, we can see how this process works for a single head.

##### Matrix Setup and Multiplication

1. **Query Matrix (QHead₁)**:
   - Shape: (4, 128)
   - Each row represents a token's first 128 dimensions
   - For our example "I love pepperoni pizza":
     ```
     Row 1: "I"         → [dim₁...dim₁₂₈]
     Row 2: "love"      → [dim₁...dim₁₂₈]
     Row 3: "pepperoni" → [dim₁...dim₁₂₈]
     Row 4: "pizza"     → [dim₁...dim₁₂₈]
     ```

2. **Key Matrix Transpose (K^T Head₁)**:
   - Original shape: (4, 128)
   - Transposed shape: (128, 4)
   - Transforms row vectors into column vectors
   - Enables dot product computation with queries

##### Attention Score Computation

The middle matrix in the diagram shows the result of:
```python
Attention = (Q × K^T) / √d_head  # where d_head = 128
```

This creates a 4×4 matrix where:
- Each cell represents the dot product between two tokens
- Rows represent queries (from Q)
- Columns represent keys (from K^T)
- Remember the inner dimensions cancel out which is why it became a 4x4 matrix

Example scores from the diagram:
```
         I    love  pepp  pizza
I     [13.9  21.1  -100  17.5]
love  [-5.0  3.14   1.2  75.3]
pepp  [ ...   ...   ...   ...]
pizza [ ...   ...   ...   ...]
```

The value 13.9 represents how strongly "I" relates to itself, 21.1 shows how strongly "I" relates to "love", and so on.

##### Scaling Factor (√d_head)

We divide by √128 (the head dimension) to:
- Prevent dot products from growing too large
- Maintain stable gradients
- Keep attention scores in a reasonable range

For example:
```python
# Without scaling
v1 · v2 = 1000  # Could lead to extreme softmax values

# With scaling (√128 ≈ 11.3)
(v1 · v2) / √128 ≈ 88.5  # More manageable value
```

##### Attention Mask for Language Models

![step3-2-attentionmask](step3-2-attentionmask.png)

In language models, we need to prevent tokens from attending to future tokens. We achieve this through attention masking:

1. **Creating the Mask**:
   ```
   For token "I":    Can see [I]
   For token "love": Can see [I, love]
   For token "pepp": Can see [I, love, pepp]
   For token "pizza": Can see [I, love, pepp, pizza]
   ```

2. **Applying the Mask**:
   - Add -∞ to positions we want to mask
   - Example from diagram:
     ```
     Original:  [13.9  21.1  -100  17.5]
     Masked:    [13.9  -∞    -∞    -∞   ]  # "I" can only see itself
     ```

##### Softmax Application

The final step converts attention scores to probabilities:
```python
attention_probs = softmax(masked_scores)  # Apply row-wise
```

Results from diagram:
```
         I    love  pepp  pizza
I     [1.0   0.0   0.0   0.0 ]  # "I" only attends to itself
love  [0.6   0.4   0.0   0.0 ]  # Masked future tokens
pepp  [0.2   0.4   0.4   0.0 ]  # Masked future tokens
pizza [0.4   0.2   0.3   0.1 ]  # Full context available
```

Key properties:
- Each row sums to 1.0
- Masked positions become 0 (e^-∞ = 0)
- Higher input scores → higher attention probabilities
- Each head computes this independently

This process allows each head to learn different attention patterns while maintaining the causal nature of language modeling (preventing information leakage from future tokens).

<br><br>

#### Step 4: Multiply by the V Sequence

![step4-multiply-v-seq](step4-multiply-v-seq.png)

After calculating attention weights in Step 3, we multiply these weights with the value (V) sequence to produce the final output. Looking at the diagram, we can see how this multiplication creates weighted combinations of token representations.

##### Matrix Components

1. **Attention Weight Matrix (Left, 4×4)**:
   ```
   Token relationships after softmax and masking:
         I    love  pepp  pizza
   I    [1.0  0.0   0.0   0.0 ]  # "I" only sees itself
   love [0.6  0.4   0.0   0.0 ]  # "love" sees "I" and itself
   pepp [0.2  0.4   0.4   0.0 ]  # "pepp" sees previous tokens
   pizza[0.4  0.2   0.3   0.1 ]  # "pizza" sees all tokens
   ```

2. **Value Matrix (Right, 4×128)**:
   ```
   Each row represents a token's 128-dimensional embedding:
   I:         [v₁₁, v₁₂, ..., v₁₁₂₈]
   love:      [v₂₁, v₂₂, ..., v₂₁₂₈]
   pepperoni: [v₃₁, v₃₂, ..., v₃₁₂₈]
   pizza:     [v₄₁, v₄₂, ..., v₄₁₂₈]
   ```

##### Computing Weighted Sums

The output matrix (4×128) is computed through matrix multiplication, where each output embedding is a weighted sum of value vectors. Let's break down how this works:

1. **First Token ("I") Output**:
   ```
   Weights: [1.0, 0.0, 0.0, 0.0]
   Output embedding = 1.0 × I_values + 0.0 × love_values + 0.0 × pepp_values + 0.0 × pizza_values
   
   For dimension 1:
   out₁₁ = (1.0 × v₁₁) + (0.0 × v₂₁) + (0.0 × v₃₁) + (0.0 × v₄₁)
   = v₁₁  # Only uses "I" token's values
   ```

2. **Second Token ("love") Output**:
   ```
   Weights: [0.6, 0.4, 0.0, 0.0]
   Output embedding = 0.6 × I_values + 0.4 × love_values + 0.0 × pepp_values + 0.0 × pizza_values
   
   For dimension 1:
   out₂₁ = (0.6 × v₁₁) + (0.4 × v₂₁) + (0.0 × v₃₁) + (0.0 × v₄₁)
   # Combines "I" and "love" token values
   ```

3. **Third Token ("pepperoni") Output**:
   ```
   Weights: [0.2, 0.4, 0.4, 0.0]
   Output embedding = 0.2 × I_values + 0.4 × love_values + 0.4 × pepp_values + 0.0 × pizza_values
   
   For dimension 1:
   out₃₁ = (0.2 × v₁₁) + (0.4 × v₂₁) + (0.4 × v₃₁) + (0.0 × v₄₁)
   # Equal contribution from "love" and "pepperoni", less from "I"
   ```

##### Understanding the Weighted Combinations

This multiplication effectively creates context-aware representations:

1. **First Token ("I")**:
   - Only uses its own values (weight 1.0)
   - Output is identical to input embedding
   - No mixing with other tokens due to causal masking

2. **Second Token ("love")**:
   - Combines information from "I" (60%) and itself (40%)
   - Example calculation for one dimension:
     ```
     If v₁₁ = 0.5 (from "I") and v₂₁ = 0.3 (from "love")
     out₂₁ = (0.6 × 0.5) + (0.4 × 0.3)
     = 0.3 + 0.12
     = 0.42  # New contextualized value
     ```

3. **Third Token ("pepperoni")**:
   - Equal weights for "love" and itself (0.4 each)
   - Small contribution from "I" (0.2)
   - Creates a balanced representation of the phrase so far

This process happens in parallel for:
- All 128 dimensions in each token
- All 8 attention heads
- Each head producing its own weighted combinations
- Each focusing on different aspects of the relationships

The final output preserves the sequence length (4) and head dimension (128) while incorporating contextual information through these weighted sums. This allows each token to carry information about relevant previous tokens, weighted by their attention scores.


<br><br>

---

#### Step 5, 6, 7: Transpose Back, Concatenate all the Heads and Multiply by Wo

After computing attention for each head independently, we need to combine their results to produce the final output. This process involves three key steps: transposing back, concatenating heads, and mixing their results.

##### Step 5: Transpose Back

![step5-transpose-back](step5-transpose-back.png)

First, we need to reorganize our attention outputs to prepare for concatenation:

1. **Initial Structure (Left Side)**:
   - Shape: (8, 4, 128)
   - Each head has its own contextualized sequence
   - Each sequence contains partial embeddings (128-dim) for all tokens
   ```python
   # Each head's output:
   Head 1: [token1₁₂₈, token2₁₂₈, token3₁₂₈, token4₁₂₈]
   Head 2: [token1₁₂₈, token2₁₂₈, token3₁₂₈, token4₁₂₈]
   ...and so on for all 8 heads
   ```

2. **Transposed Structure (Right Side)**:
   - Shape: (4, 8, 128)
   - Each token now has 8 different contextualized representations
   - Each representation is 128 dimensions
   ```python
   # After transpose:
   Token 1: [head1₁₂₈, head2₁₂₈, ..., head8₁₂₈]
   Token 2: [head1₁₂₈, head2₁₂₈, ..., head8₁₂₈]
   ...and so on for all 4 tokens
   ```

---

##### Step 6: Concatenate all the Heads

![step6-concatenate-all-heads](step6-concatenate-all-heads.png)

Next, we merge the heads' outputs into a single embedding for each token:

1. **Before Concatenation (Left Side)**:
   - Shape: (4, 8, 128)
   - Each token has 8 separate contextualized embeddings
   ```python
   Token 1: [
       head1: [0.1, ..., 0.8],  # 128 dims
       head2: [0.3, ..., 0.5],  # 128 dims
       ...
       head8: [0.2, ..., 0.9]   # 128 dims
   ]
   ```

2. **After Concatenation (Right Side)**:
   - Shape: (4, 1024)
   - Each token now has one large embedding
   - 1024 = 8 heads × 128 dimensions
   ```python
   Token 1: [head1₁₂₈ | head2₁₂₈ | ... | head8₁₂₈]  # 1024 dims
   Token 2: [head1₁₂₈ | head2₁₂₈ | ... | head8₁₂₈]  # 1024 dims
   ...and so on
   ```

##### Step 7: Multiply by Wo

![step7-multiply-by-wo](step7-multiply-by-wo.png)

Finally, we mix the information from different heads using the Wo parameter matrix:

1. **Input (Left Side)**:
   - Shape: (4, 1024)
   - Concatenated but unmixed head outputs
   ```
   Token 1: [head1_all | head2_all | ... | head8_all]
   Token 2: [head1_all | head2_all | ... | head8_all]
   ...
   ```

2. **Wo Parameter Matrix (Middle)**:
   - Shape: (1024, 1024)
   - Enables mixing between head outputs
   - Each output dimension depends on all head outputs

3. **Final Output (Right Side)**:
   - Shape: (4, 1024)
   - Each dimension is now a mixture of all head outputs
   ```python
   # For first dimension of first token:
   output₁₁ = Σ(token1_concat₁₀₂₄ × Wo_column1₁₀₂₄)
   # Uses all 1024 values from concatenated heads
   ```

##### Why This Process Matters

1. **Transpose Back (Step 5)**:
   - Reorganizes from head-centric to token-centric view
   - Prepares for efficient concatenation
   - Maintains parallel processing benefits

2. **Concatenation (Step 6)**:
   - Preserves all information from each head
   - Restores original embedding dimension
   - But heads remain independent

3. **Wo Multiplication (Step 7)**:
   - Critical for mixing head information
   - Without Wo: Just independent parallel processes
   - With Wo: Rich interactions between head outputs
   ```python
   # Example of mixing:
   Before Wo:
   token1 = [financial_aspect | color_aspect | motion_aspect | ...]
   
   After Wo:
   token1 = [
       dim1 = 0.3×financial + 0.5×color + 0.2×motion + ...,
       dim2 = 0.1×financial + 0.7×color + 0.2×motion + ...,
       ...
   ]
   ```

This three-step process transforms the parallel, independent head outputs into a rich, mixed representation that captures the full complexity of token relationships while maintaining the model's original dimensionality.

