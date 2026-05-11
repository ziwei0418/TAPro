# Paper Method Notes

This file stores detailed notes for each paper/method the user sends.

Important rule for this file:
- Only record the method faithfully and in detail.
- Do not analyze, compare, or propose new ideas here until the user says all papers have been sent.

---

## Paper 1: SSR Framework

### High-Level Goal

SSR is designed for text-guided weakly supervised semantic segmentation under cross-modal supervision. The paper identifies two coupled problems:

1. **Semantic discrepancy between text and image modalities**
   - Text features and visual features are not naturally aligned at precise pixel level.
   - Text-guided activation can therefore spill into semantically similar but non-target foreground regions.

2. **Spatial over-activation during CAM refinement**
   - Affinity-matrix-based propagation can spread noise into background regions.
   - This often causes background over-activation and degraded pseudo masks.

To address this, SSR explicitly separates the problem into:

- **semantic alignment**
- **spatial correction**

and optimizes both jointly.

The framework has two main components:

1. **Cross-Modal Prototype Alignment**
2. **Superpixel-Guided Correction**

Overall supervision:

- prototype contrastive loss
- segmentation loss on pseudo masks

---

### Framework Inputs

The framework takes:

- image modality `I`
- text modality `T`

where `T` contains:

- `K` foreground categories
- `M` background categories from CLIP-ES

The paper describes the input label space as:

- class labels `Y = 1,2,...,C`

Semantic modeling and spatial modeling are performed separately and then combined.

---

## 1. Cross-Modal Prototype Alignment

### 1.1 Multimodal Prototype Generation

The method starts from CLIP image and text features, then refines them with two separate alignment heads:

- **ISA**: Image Semantic Alignment
- **TSA**: Text Semantic Alignment

These two branches:

- have the same structure
- do **not** share parameters

They project image and text features into a new shared space where fine-grained alignment is easier.

Formally:

- image feature after projection:
  - `v'_i = ISA(v_i)`
- text feature after projection:
  - `t'_i = TSA(t_i)`

Dimensions:

- `v'_i ∈ R^(1 × d2)`
- `t'_i ∈ R^(cf × d2)`

where:

- `cf` is the number of categories relevant to the current image

The paper gives two motivations for constructing prototypes in this **projected space** instead of raw CLIP space:

1. preserve CLIP’s original instance discrimination in the original space
2. reduce prototype construction cost using lower-dimensional projected representations

---

### 1.2 Prototype Extraction

For each image, the method computes class-specific CAMs:

- `CAM^c_ij` using GradCAM

Using these CAMs, it extracts:

- image-specific foreground-aware prototype features
- text-side category prototype features

The image-side foreground feature is obtained by:

- masked average pooling over projected image features:
  - `f_image = MAP(CAM^c ⊙ v'_i)`

The text-side feature is simply selected from the projected text features:

- `f_text = t'_i[index]`

where `index` retrieves the target category text feature.

So:

- `f_image` is a foreground-aware image descriptor for the target class
- `f_text` is the matching class text descriptor

These are then collected across the dataset.

---

### 1.3 Dataset-Level Prototype Construction

All foreground-aware image features and text features are gathered over the dataset and clustered by K-means to form:

- image prototypes:
  - `P^I ∈ R^(K × d2) = [P^I_1, ..., P^I_K]`
- text prototypes:
  - `P^T ∈ R^(K × d2) = [P^T_1, ..., P^T_K]`

After clustering:

- cluster pseudo-labels are assigned based on nearest prototype proximity

These cluster pseudo-labels are then used to define positive and negative samples for prototype contrastive learning.

---

### 1.4 Prototype Contrastive Learning

This is the main semantic-alignment mechanism.

The paper claims three simultaneous effects:

1. visual features are matched with text prototypes of the same category
2. text prototypes are aligned with visual prototypes of the same category
3. prototypes of different categories are pushed apart

This is intended to:

- narrow the modality gap
- improve inter-class separation
- reduce category confusion

#### Similarity computation

Similarity between projected image feature and text prototypes:

- `p^I_i = (v'_i · P^T) / τ_proto`

where:

- `τ_proto` is a **learnable temperature**

#### Positive and negative pairs

Using cluster pseudo-label `posidx`:

- positive sample:
  - `S^pos_i = {p_i,j | j = posidx}`
- negative samples:
  - `{S^neg_i1, ..., S^neg_ik} = {p_i,j | j ≠ posidx}`

#### Loss

The loss is standard prototype-based contrastive classification:

- positive sample in numerator
- positive + all negatives in denominator

The final prototype contrastive loss:

- `L_proto`

implemented as an average over the batch with a cross-entropy-style objective.

Main intended effect:

- compact cross-modal semantic alignment in shared embedding space

---

## 2. Superpixel-Guided Correction

This module handles the **spatial** problem, not the semantic problem.

The motivation:

- attention-based affinity propagation often spreads wrong activation into background
- this produces noisy refined CAMs

SSR addresses this by building a **binary spatial mask** over the affinity matrix using superpixel structure.

---

### 2.1 Superpixel Clustering

The paper uses:

- **SLIC** superpixel segmentation

Reason for SLIC:

- efficient
- preserves object boundaries
- lightweight compared with heavier segmentation priors like SAM

Process:

1. run `SLIC(I_i)` on the image
2. cluster the superpixels again using color-space information:
   - `C = K-means(SLIC(I_i))`

Then determine which clustered regions are target regions by computing:

- ratio of high-confidence activation sum to total activation in each region

Only regions above a threshold are retained as target regions.

This yields a target-region indicator used to build a binary mask over the affinity matrix.

---

### 2.2 Binary Affinity Mask

Define a binary mask matrix:

- `Mask`

Its elements are:

- `a_ij = 1` if column `j` belongs to target region
- `a_ij = 0` otherwise

The key idea is:

- instead of letting affinity propagate everywhere
- explicitly zero out column vectors associated with non-target regions

So the mask is used to suppress propagation of noisy semantics into background.

---

### 2.3 Affinity Matrix Correction

The paper combines attention features from:

- **CLIP MHSA**
- **DINO MHSA**

Rationale:

- CLIP MHSA provides strong semantic guidance but weak boundary precision
- DINO MHSA better preserves local-to-global structural consistency

The fused affinity matrix is:

- `A = ConCat(MHSA_CLIP, MHSA_DINO)`

followed by normalization

This fusion aims to combine:

- semantic discriminability from CLIP
- spatial detail from DINO

Then the binary superpixel mask is applied:

- `A* = A ⊙ Mask`

The refined CAM is obtained by propagating the original CAM through the corrected affinity matrix:

- `CAM^c_refine = A* ⊗ CAM^c`

So the full spatial refinement story is:

1. build initial class CAM
2. construct fused affinity matrix from CLIP + DINO attention
3. suppress non-target propagation via superpixel-derived binary mask
4. propagate CAM through corrected affinity matrix

---

## 3. Training Objectives

The full SSR objective is:

- `L_SSR = L_proto + γ L_seg`

### 3.1 Prototype Contrastive Loss

- `L_proto`

Purpose:

- align image features with cross-modal prototypes
- improve compactness within class
- improve separation across classes

### 3.2 Segmentation Loss

- `L_seg`

Uses online-generated pseudo masks

Loss design:

- Dice loss
- Cross-Entropy loss

For segmentation prediction:

- coarse or refined mask supervision

The paper defines:

- `L_Dice`
- `L_CE`
- `L_seg = 0.5 * L_Dice + 0.5 * L_CE`

So final objective:

- contrastive semantic alignment
- plus pseudo-mask segmentation supervision

---

## 4. Core Design Summary

SSR can be summarized as a two-branch correction framework:

### Semantic branch

- CLIP image/text features
- ISA/TSA projection
- dataset-level image/text prototypes
- cluster pseudo-labels
- prototype contrastive learning

Main purpose:

- reduce image-text modality gap
- improve class discrimination

### Spatial branch

- class CAM from GradCAM
- SLIC superpixels
- clustered target-region mask
- fused CLIP+DINO affinity matrix
- binary masking of affinity matrix columns
- refined CAM propagation

Main purpose:

- suppress noisy background propagation
- improve localization precision

---

## 5. Components That Appear Most Distinctive

Without analyzing usefulness yet, the most distinctive pieces in this paper are:

1. **dual alignment heads (ISA/TSA)** before prototype construction
2. **dataset-level cross-modal prototype construction** using projected features
3. **prototype contrastive learning with cluster pseudo-labels**
4. **superpixel-guided binary masking of affinity propagation**
5. **fusion of CLIP and DINO attention maps for CAM refinement**

---

## 6. Immediate Notes for Later Analysis

These are not conclusions, just placeholders for future analysis after all papers are collected:

- prototype alignment is done in a projected space, not raw CLIP space
- text is not just initialization; it participates in prototype contrastive learning
- spatial refinement explicitly constrains **where** affinity can propagate
- superpixels are used as lightweight structure prior instead of large mask models

---

## Paper 2: Vision Prototype Learning to Mitigate Modality Gap

### High-Level Goal

This method starts from the observation that CLIP text prototypes are not ideal for pixel-level segmentation because of the **modality gap** between:

- text space
- vision space

The core claim is:

- text prototypes are semantically useful
- but they do not align precisely enough with the visual feature distribution needed for segmentation

Therefore, instead of directly using text prototypes for segmentation, the paper proposes to:

1. **learn vision prototypes from text supervision**
2. **use these learned vision prototypes to generate better GradCAM pseudo masks**
3. **refine region-level supervision through Regional Semantic Contrast (RSC)**

The full method is called:

- **Weakly-supervised Vision Prototype Learning (VPL)**

It has two main phases:

1. learning optimal vision prototypes to mitigate modality gap
2. regional semantic contrast learning for decoder supervision

---

## 1. Preliminary Setup

For each training image:

- `x_i ∈ R^(w × h × 3)`

The dataset provides only:

- image-level labels `{z_n}`

From CLIP:

- image representation:
  - `x_i = E_v(x_i)`
- text representation:
  - `z_n = E_t(z_n)`
- GradCAM:
  - `M_n = GradCAM(x_i, x_n)`

Representations:

- visual representation:
  - `x_i ∈ R^(S × H × D)`
- text representation:
  - `z_n ∈ R^D`

where:

- `D` is channel dimension
- `S × H` is spatial size

The method is built on a theoretical argument that CLIP’s contrastive learning objective does **not** force text and image embeddings into a truly unified shared space for fine-grained prediction.

---

## 2. Modality Gap Analysis in CLIP

The paper revisits the standard CLIP contrastive objective over a batch of image-text pairs `(v_i, t_i)`.

CLIP optimizes:

- high cosine similarity for matched pairs
- low cosine similarity for unmatched pairs

The contrastive loss uses:

- normalized image embedding `v_i`
- normalized text embedding `t_i`
- learned temperature `τ`

The key argument is:

- even though CLIP aligns image and text semantically at high level
- it does **not** make their distributions identical at the fine-grained feature level needed for segmentation

The paper explicitly states:

- CLIP does not merge image and text into an indistinguishable unified representation space
- text prototypes therefore may not sufficiently capture the nuanced distribution of visual features needed for accurate segmentation

This motivates learning **vision prototypes** rather than relying on text prototypes directly.

---

## 3. Learning Vision Prototypes from Text Supervision

### 3.1 Supervised Vision Prototype Objective

The paper first recalls the ideal fully supervised semantic segmentation classifier:

- pixel features `x_i,k`
- learnable vision classifier/prototype bank `W = {w_n}`

The standard objective:

- optimize `W` so each pixel is classified correctly under cross-entropy

The intuition:

- the optimal vision prototypes `W*` would best reflect semantic structure and class distribution in vision space

This leads to the question:

- can we recover or approximate optimal vision prototypes using only text supervision and pseudo labels?

---

### 3.2 Hypothesis: Text Prototype Decomposition

The paper decomposes each text prototype `z_n` into:

- one component lying in the vision subspace:
  - `z^x_n`
- one component lying in the orthogonal complement:
  - `z^⊥_n`

Form:

- `z_n = sqrt(ϵ) z^x_n + sqrt(1-ϵ) z^⊥_n`

subject to:

- unit norms
- orthogonality between the two components

Interpretation:

- when `ϵ → 1`, text prototype contains more vision-relevant information

This is a theoretical abstraction used to reason about whether text prototypes can fully recover optimal vision-space decision boundaries.

The paper concludes that in practice this assumption is too strong:

- text space does **not** fully cover vision space
- there are missing visual details such as texture/color
- therefore text prototypes alone are insufficient

---

### 3.3 Proposition: Conditions Under Which Text Prototypes Could Match Vision Prototypes

The paper derives a result:

- if the vision component of the text prototype exactly matches the optimal vision prototype
- and if temperatures are correspondingly related

then text-prototype-based prediction could match vision-prototype-based prediction

Specifically:

- `p' = p` only under ideal conditions such as:
  - `z^x_n = w*_n`
  - appropriate relation between `τ_T` and `τ_I`

The interpretation is:

- larger temperature may help narrow the modality gap
- but this ideal match is difficult in practice

---

### 3.4 Theorem: Lower Bound of Modality Gap

The paper then explicitly defines a lower bound on modality gap:

- `Δ_gap = ||Z - W*||^2_F ≥ c_⊥ + ϵ`

where:

- `c_⊥` represents irrelevant dimensions in text space that are orthogonal to useful visual space
- `ϵ` represents approximation loss due to only partial overlap between text and vision spaces

Interpretation:

- some part of text space is fundamentally irrelevant to segmentation-relevant visual structure
- some semantic content is only partially shared
- hence directly minimizing distance between text prototypes and optimal vision prototypes is inherently difficult

This motivates learning **vision-space prototypes** rather than depending on text-space prototypes.

---

## 4. Vision Prototype Learning by KL-Based Optimization

### 4.1 Core Objective

The method uses text prototypes to generate pseudo labels first, then learns better **vision prototypes** by matching distributions.

It constructs a KL-divergence objective:

- minimize the divergence between:
  - distribution from text prototypes `P'_i`
  - distribution from learnable vision prototypes `P_i`

Objective:

- `min_W L(P', W) = Σ_i KL(P'_i || P_i)`

This aligns the learnable vision prototypes to the text-prototype-induced distribution while adapting them into the visual space.

The paper describes two ways to optimize KL divergence:

1. **class-level regularization**
2. **instance-level regularization**

It argues that:

- instance-level regularization better captures real data variation

---

### 4.2 Theorem on Instance-Level Approximation

The paper proves that:

- the distribution defined using anchor classes approximates the one defined using anchor examples

with tightness improving when intra-class features are compact.

Interpretation:

- using instance-level alignment is preferable when modeling real vision-space variability

This supports their choice to optimize vision prototypes based on instance-aware distributions.

---

### 4.3 Learning the Vision Prototypes

The final learnable vision prototypes are denoted:

- `W'* = arg min_W L(P', W)`

The ideal supervised optimum is:

- `W* = arg min_W L(m, W)`

where:

- `m_i` is the pseudo mask distribution generated through text-prototype-based GradCAM

The paper uses standard gradient descent:

1. initialize and optimize `W'*` by minimizing KL-based objective
2. compute gradient `∇_{W'*} L(W)`
3. update:
   - `W'* ← W'* - α ∇_{W'*} L(W)`
4. use updated vision prototypes to generate GradCAM masks

This iterative process yields vision prototypes that are better aligned with segmentation-relevant vision space than the original text prototypes.

The paper emphasizes:

- temperature for learning vision prototypes should be larger than CLIP’s temperature
- their vision prototypes generate GradCAMs with better localization and less background distraction than text prototypes

---

### 4.4 Role of Vision Prototypes

After optimization, the learned vision prototypes:

- replace text prototypes for target localization
- generate improved GradCAMs
- produce pseudo labels for downstream decoder supervision

So the architecture is:

1. text prototypes provide initial semantic guidance
2. learn vision prototypes by KL-based optimization
3. use learned vision prototypes for localization and pseudo-mask generation

This is a central distinction from methods that use text prototypes directly at inference.

---

## 5. Regional Semantic Contrast Learning (RSC)

The paper then introduces a second component for refining local supervision:

- **Regional Semantic Contrast Learning**

Motivation:

- pseudo labels from GradCAM are noisy
- standard mask supervision alone is not sufficient to exploit object regions fully

The method therefore adds region-level contrastive supervision between:

- region embeddings
- vision prototypes

---

### 5.1 Region Embedding Construction

From the decoder:

- dense pixel embeddings `V ∈ R^(B × D × H × S)`

From vision prototypes:

- `W ∈ R^(D × N)`

For each batch sample and class:

- compute masks:
  - `A_i,n = W_n^T V_i`

Then apply masked average pooling to produce region-level embeddings:

- `P_i,n`

This gives feature-level localized image embeddings corresponding to class-specific regions.

---

### 5.2 Regional Semantic Contrast Loss

The similarity between:

- localized image embedding `P_i,n`
- vision prototype `W_n`

is:

- `S_i,n = P_i,n^T W_n`

The loss `L_RSC` is a contrastive objective over the batch:

- positive term for matched region/prototype pair
- negatives from other samples in batch

This is a symmetric batchwise contrastive loss:

- one direction uses `S_i,i`
- the other direction also contrasts along the transposed direction

The stated purpose is:

- align specific region embeddings with the correct vision prototypes
- suppress negative masks obtained from unrelated texts
- improve prototype-region correspondence despite pseudo-label noise

---

## 6. Final Loss

The final loss is:

- `L = λ_ce L_ce(M, A) + λ_RSC L_RSC`

where:

- `L_ce` is cross-entropy segmentation loss
- `M` is the mask from post-processed GradCAM
- `A` is the decoder output / segmentation prediction
- `L_RSC` is the region-level semantic contrast loss

So training combines:

1. pseudo-mask supervision from improved GradCAMs generated by learned vision prototypes
2. region-level contrastive alignment with vision prototypes

---

## 7. Two-Phase Structure

The overall VPL pipeline is clearly two-phase:

### Phase 1: Learn Vision Prototypes

- start from CLIP image/text embeddings
- use text prototypes as supervision signal
- optimize learnable vision prototypes by KL divergence
- generate GradCAM pseudo masks from vision prototypes

### Phase 2: Decoder Supervision with RSC

- use the pseudo masks from learned vision prototypes
- supervise the segmentation decoder with CE loss
- compute region embeddings from decoder features
- align them with vision prototypes via RSC loss

This is the paper’s main structural story.

---

## 8. Core Design Summary

This method is fundamentally about **replacing text prototypes with learned vision prototypes** to mitigate modality gap.

Its distinctive parts are:

1. explicit theoretical treatment of CLIP modality gap
2. decomposition of text prototype into vision-space and orthogonal components
3. lower-bound analysis on modality gap
4. KL-based optimization to learn vision prototypes from text supervision
5. use of those learned vision prototypes to generate improved GradCAM pseudo masks
6. region-level semantic contrast between decoder regions and vision prototypes

---

## 9. Immediate Notes for Later Analysis

These are placeholders only, not conclusions yet:

- text is treated as supervision to learn better **vision** prototypes, not as the final prototype itself
- there is a strong emphasis on modality-gap theory and optimization
- GradCAM is still central, but the prototypes generating it are changed
- RSC is a region-prototype contrastive mechanism layered on top of pseudo-label supervision

---

## Paper 3: Query-Guided Attention Fusion / Location-Aware Parameter Fine-Tuning

### High-Level Goal

This paper targets multimodal image segmentation by using text not only as a global class cue, but also as a prompt source that can interact with visual features during encoding and decoding.

The method appears to be built around three prompt sources:

1. **coarse mask prompt**
2. **point prompt**
3. **text prompt**

These prompts are then injected into a segmentation model built on top of:

- a CLIP image/text encoder branch
- a SAM2 / MedSAM2 image encoder and mask decoder branch

The central proposed component is:

- **QGAF: Query-guided Attention Fusion**

There is also an auxiliary module:

- **GPFC: Global Positional Feature Classification**

The design emphasis is on:

- minimal parameter fine-tuning
- stronger localization from text-image interaction
- prompt generation from cross-modal features

---

## 1. Prompt Generation

The paper first produces prompts from CLIP-derived image and text features before feeding them into the SAM-style segmentation branch.

### 1.1 Coarse Mask Prompt

After the final encoder stage, the method obtains:

- final image feature `F_v^C`
- final text feature `F_t`

It uses the end-of-sequence token:

- `F_t[EOS]`

as a global text feature.

Then it reshapes the final visual feature map and computes an element-wise interaction between:

- global text feature
- pixel-level image features

to generate a pseudo mask:

- `M = F_t[EOS] ⊙ F_v^C`

Interpretation:

- this acts like a similarity map between the global text representation and each image pixel feature
- higher similarity means higher probability that the pixel belongs to the target lesion/object region

So the coarse mask is a text-guided activation prior derived directly from joint image-text similarity.

---

### 1.2 Point Prompt

The point prompt is generated from the coarse mask `M`.

Procedure:

1. each spatial point gets a probability value from `M`
2. candidate points are selected if probability exceeds a threshold:
   - `theta > 0.9`
3. the first point is the candidate with highest likelihood
4. subsequent points are selected using a diversity criterion:
   - choose the point farthest from previously selected points

The distance criterion is:

- `Dis_max(p, P)`

where `P` is the set of already selected points.

Purpose:

- avoid all points clustering in a single region
- ensure selected prompts are both:
  - highly confident
  - spatially well distributed

This means point prompts are not random or uniform; they are top-confidence and coverage-aware.

---

### 1.3 Text Prompt

The text prompt is also derived from the final text representation:

- use `F_t[EOS]` as global information
- pass it through a simple feed-forward network (FFN)

This yields a learned text prompt embedding used by the segmentation prompt encoder.

So text is used in two ways:

1. directly for coarse mask generation
2. as an explicit prompt token after FFN projection

---

## 2. Mask Decoder Pipeline

The segmentation branch uses a MedSAM2-style prompt and mask decoding pipeline.

Inputs to prompt encoder:

- coarse mask prompt `E_m`
- point prompt `E_p`
- text prompt `E_t`

The SAM2 image encoder extracts image features:

- `F_v^S`

Then the decoder combines:

- image features `F_v^S`
- coarse mask embedding `E_m`
- prompt embeddings `[E_p, E_t]`

Final segmentation:

- `S = SAM2Dec(F_v^S + E_m, [E_p, E_t])`

Interpretation:

- the coarse mask embedding is added into image features
- point/text prompts are passed as prompt tokens into the decoder

So the segmentation result is driven by both:

- visual features from SAM2
- prompt information distilled from the CLIP image-text branch

---

## 3. Query-Guided Attention Fusion (QGAF)

This is the main proposed fusion module.

### 3.1 Motivation

The paper states that previous methods often rely on simple attention mechanisms for multimodal fusion.

Instead, QGAF introduces:

- a learnable query token `Q`

to bridge interaction between:

- text tokens `F_t`
- visual features `F_v^C`

The goal is to inject text-guided localization information during encoding while keeping the parameter update lightweight.

---

### 3.2 Structure

During the image encoding stage, QGAF fuses text and image information through a two-stream process.

The learnable query:

- `Q in R^(T x C_t)`

is concatenated with text features.

To improve efficiency:

- a zoom-in operation first reduces feature dimension to `C_f = 64`
- fusion is then performed in the compressed space
- a zoom-out operation restores the original dimension afterward

The fusion uses both:

- self-attention
- cross-attention

---

### 3.3 Update Equations

The paper gives the following updates:

- `F'_t = F_t + MHCA(MHSA([Q, F_t]), MHSA(F_v^C))`
- `F'_v^C = F_v^C + MHCA(MHSA(F_v^C), MHSA([Q, F_t]))`

where:

- `MHSA` = multi-head self-attention
- `MHCA` = multi-head cross-attention
- `[Q, F_t]` = concatenation of learnable query and text features

Interpretation:

- text branch is updated using query-enriched text features and visual attention
- visual branch is also updated using its own self-attention plus guidance from query-conditioned text features

So QGAF is a bidirectional text-image fusion mechanism centered on a learnable query bridge.

---

## 4. Global Positional Feature Classification (GPFC)

This is an auxiliary classification component.

### 4.1 Goal

GPFC predicts coarse global lesion/location information from combined image and text global features.

In the paper’s medical setting, it partitions the lung into six lesion regions:

- upper / middle / lower lobe
- for left and right lungs

This region-level prediction is used to encourage multimodal consistency.

---

### 4.2 Image and Text Global Features

For each encoder block:

1. image feature is refined by an adapter:
   - two linear layers + activation
2. a Softmax produces regional weights
3. weighted aggregation yields global image feature:
   - `V_g`

For text:

- global text feature `T_g` is taken from the EOS token of text representation

Then:

- concatenate `[V_g, T_g]`
- pass through a classification layer
- compute BCE loss against region label `Y`

Loss at encoder block `i`:

- `l_c^i = BCE(Linear([V_g, T_g]), Y)`

Overall classification loss:

- `l_c = sum_{i=1}^4 l_c^i`

So GPFC is applied across all four encoder blocks, not only the final one.

---

## 5. Segmentation Loss

The method uses segmentation supervision on both:

1. coarse mask `M`
2. final segmentation output `S`

Loss composition:

- Dice loss
- Cross-Entropy loss

Definitions:

- `l_Dice`
- `l_CE`
- `l_s = 0.5 l_Dice + 0.5 l_CE`

This segmentation loss is applied to:

- coarse mask prediction
- final decoder segmentation

---

## 6. Final Objective

The overall loss is:

- `l = lambda_1 l_s(M) + lambda_2 l_s(S) + lambda_3 l_c`

Components:

1. segmentation loss on coarse mask
2. segmentation loss on final mask
3. global positional classification loss

So optimization jointly encourages:

- coarse localization quality
- final segmentation quality
- global multimodal positional consistency

---

## 7. End-to-End Pipeline Summary

The whole method can be summarized as:

1. use CLIP image/text encoders to extract multimodal features
2. fuse text and image features during encoding via QGAF
3. derive:
   - coarse mask prompt from text-image similarity
   - point prompts from high-confidence diverse points
   - text prompt from EOS token through FFN
4. encode these prompts with MedSAM2 prompt encoder
5. feed prompt embeddings and SAM2 image features into mask decoder
6. supervise:
   - coarse mask
   - final segmentation
   - global positional classification branch

---

## 8. Components That Look Most Distinctive

Without analysis yet, the most distinctive pieces in this paper are:

1. **query-guided bidirectional attention fusion** between text and image features
2. **coarse mask prompt generated directly from EOS-text × visual feature similarity**
3. **coverage-aware point prompt selection** from top-confidence mask regions
4. **joint use of CLIP-style multimodal encoder and SAM2-style decoder**
5. **GPFC auxiliary head** for global multimodal regional classification

---

## 9. Immediate Notes for Later Analysis

These are placeholders only, not conclusions yet:

- text is not only used as a prototype or class embedding; it also generates prompts and guides feature fusion
- coarse mask is generated directly from global text-to-pixel similarity
- the query token serves as a bridge rather than simply averaging/fusing text and image features
- the segmentation backbone is explicitly prompt-driven through a SAM-style decoder

---

## Paper 4: SemPLeS (Semantic Prompt Learning for WSSS)

### High-Level Goal

SemPLeS is a CLIP-guided weakly supervised semantic segmentation framework designed to improve the semantic alignment between segmented object regions and their target text labels.

The paper starts from the observation that:

- image-level supervision can train a mask generator to produce coarse object masks
- but these masks often include **co-occurring background regions**
- standard text-label matching alone is therefore insufficient for precise segmentation

To address this, the framework consists of three connected stages:

1. **Segment-Label Matching**
2. **Contrastive Prompt Learning**
3. **Prompt-guided Semantic Refinement**

The overall idea is:

- use CLIP to align foreground regions with class text
- learn **background prompts** automatically for each class
- use these learned prompts to suppress class-associated background activation

The final goal is to produce:

- compact and complete refined masks `M'`

without using pixel-level annotations.

---

## 1. Problem Formulation

Training data:

- images `X in R^(H x W x 3)`
- image-level labels `y in {0,1}^K`

where:

- `K` is the number of object categories
- `y` is a multi-hot vector indicating which categories are present

There is no pixel-wise annotation.

The framework learns a mask generator:

- `S`

which predicts object masks from image-level supervision with CLIP guidance.

The method uses:

- frozen CLIP image encoder `E_I`
- frozen CLIP text encoder `E_T`

Only the segmentation mask generator and the learnable prompts are trainable in their respective stages.

---

## 2. Segment-Label Matching

### 2.1 Goal

The first stage uses image-text contrastive learning to train the mask generator to distinguish foreground from background with respect to the correct class label.

Given an input image `X`:

- mask generator produces foreground masks:
  - `M = S(X)`

For each ground-truth category `k` present in the image (`y_k = 1`):

- foreground image:
  - `X_k^f = M_k · X`
- background image:
  - `X_k^b = (1 - M_k) · X`

The text input for class `k` uses a standard CLIP template:

- `t_k = "a photo of {}"`

filled with the class name.

So the triplet used in this stage is:

- foreground crop `X_k^f`
- background crop `X_k^b`
- foreground text label `t_k`

---

### 2.2 Contrastive Matching Objective

The CLIP image encoder and text encoder produce:

- foreground image feature:
  - `v_k^f = E_I(X_k^f)`
- background image feature:
  - `v_k^b = E_I(X_k^b)`
- foreground text feature:
  - `u_k^f = E_T(t_k)`

The objective is:

1. maximize similarity between foreground image and foreground text
2. minimize similarity between background image and foreground text

Matching loss:

- `L_match = E_X[-log(sim(v_k^f, u_k^f))] + E_X[-lambda_b log(1 - sim(v_k^b, u_k^f))]`

where:

- `sim` is cosine similarity
- `lambda_b` weights the background repulsion term

Important design choice:

- CLIP encoders are kept frozen
- this preserves CLIP’s pre-trained joint latent space
- avoids overfitting caused by fine-tuning the large vision-language backbone

This stage trains the mask generator `S` to separate foreground/background using only image-level labels and CLIP similarity.

---

## 3. Contrastive Prompt Learning

### 3.1 Motivation

Even after segment-label matching, the generated masks remain coarse and may still include class-associated co-occurring backgrounds.

Previous CLIP-based refinement methods often rely on:

- manual prompt engineering
- hand-crafted background prompts

SemPLeS instead proposes to **learn prompts automatically**.

The key idea is:

- for each category `k`, learn a prompt `p_k`
- this prompt should describe the category-associated **background**, not the object itself

These prompts are intended for segmentation refinement rather than standard classification prompt tuning.

---

### 3.2 Learnable Background Prompts

For each category `k`, a learnable prompt sequence `p_k` is fed into the frozen CLIP text encoder:

- background prompt embedding:
  - `u_k^b = E_T(p_k)`

The background image representation is:

- `v_k^b = E_I(X_k^b)`

The foreground class text embedding remains:

- `u_k^f = E_T(t_k)`

The prompt learning objective has two parts:

1. align learned prompt with background image
2. repel learned prompt from the foreground class text

Loss:

- `L_prompt = L_prompt^I + lambda_T L_prompt^T`

Expanded form:

- `L_prompt = E_X[-log(sim(u_k^b, v_k^b))] + E_X[-lambda_T log(1 - sim(u_k^b, u_k^f))]`

where:

- `lambda_T` controls repulsion from the class text label

Training protocol for this stage:

- mask generator `S` is fixed
- only the prompts `p_k` are trainable

After learning:

- `p_k` captures co-occurring backgrounds associated with class `k`
- no manual background prompt engineering is required

The paper explicitly distinguishes this from prior prompt learning work:

- not replacing generic templates for classification
- but learning category-specific background semantics for segmentation refinement

---

## 4. Prompt-guided Semantic Refinement

### 4.1 Goal

Once the background prompts `p_k` have been learned, SemPLeS uses them to refine the segmentation masks by suppressing class-associated background regions from the predicted foreground.

This stage again updates:

- mask generator `S`

while keeping:

- learned prompts `p_k` fixed

The foreground representation remains:

- `v_k^f = E_I(X_k^f)`

The learned background prompt representation is:

- `u_k^b = E_T(p_k)`

The refinement objective is to:

- reduce similarity between predicted foreground and learned background semantics

Refinement loss:

- `L_refine = E_X[-log(1 - sim(v_k^f, u_k^b))]`

Interpretation:

- if a foreground mask still contains co-occurring background content, its visual embedding will remain similar to the learned background prompt
- minimizing this similarity suppresses those background regions

---

### 4.2 Total Loss

The final objective combines:

- segment-label matching
- prompt-guided refinement

Total loss:

- `L_total = L_match + lambda L_refine`

where:

- `lambda` weights the refinement term

So overall optimization:

1. encourages masks to match the correct object class text
2. discourages masks from matching the learned class-associated background prompts

This is the paper’s central formulation for semantic refinement.

---

## 5. Training Schedule / Module Roles

The three stages interact as follows:

### Stage A: Segment-Label Matching

- train mask generator `S`
- CLIP image/text encoders frozen
- obtain initial coarse masks `M`

### Stage B: Contrastive Prompt Learning

- fix mask generator `S`
- learn prompts `p_k`
- CLIP encoders remain frozen
- prompts learn to encode class-associated background semantics

### Stage C: Prompt-guided Semantic Refinement

- fix learned prompts `p_k`
- use them to guide mask generator `S`
- suppress co-occurring background regions from masks

After training:

- only the mask generator `S` is needed at inference for producing final masks `M'`
- CLIP and learned prompts are only used during training as guidance

This is an important practical characteristic of the framework.

---

## 6. Core Design Summary

SemPLeS can be summarized as a CLIP-guided WSSS framework where:

1. the mask generator first learns object-vs-background separation through foreground/background-text matching
2. category-specific **background prompts** are then learned automatically from CLIP latent space
3. these learned prompts are used to suppress co-occurring background semantics from the predicted foreground masks

The method’s distinct semantic idea is:

- learn prompts that represent **what should be excluded**
- not just prompts that represent the target object

---

## 7. Components That Look Most Distinctive

Without analysis yet, the most distinctive pieces in this paper are:

1. **foreground-background-text triplet matching** for training the mask generator
2. **learnable category-specific background prompts**
3. **contrastive prompt learning against both background image and class text**
4. **prompt-guided refinement by explicit background suppression**
5. **training-only use of CLIP/prompt guidance, with mask generator alone used at inference**

---

## 8. Immediate Notes for Later Analysis

These are placeholders only, not conclusions yet:

- prompts are used to model co-occurring backgrounds rather than foreground classes
- refinement works by repelling predicted foreground from learned background prompt embeddings
- CLIP remains frozen throughout; only mask generator or prompts are updated depending on stage
- the method’s semantic refinement is subtractive/suppressive rather than additive

---

## Paper 5: SAPL (Semantic-Agnostic Prompt Learning)

### High-Level Goal

SAPL is proposed for weakly supervised image manipulation localization under image-level binary supervision.

The setting is:

- no pixel annotations
- only image-level binary labels (e.g. REAL / FAKE)

The paper’s main claim is that conventional CLIP-based prompting tends to focus on high-level semantics, whereas manipulation localization depends more on:

- low-level inconsistencies
- subtle boundary artifacts
- non-semantic irregularities

To address this, SAPL introduces two core components:

1. **Edge-aware Contextual Prompt Learning (ECPL)**
2. **Hierarchical Edge Contrastive Learning (HECL)**

The central design theme is:

- use edge information to build **semantic-agnostic prompts**
- reinforce patch-level edge representations across multiple feature depths

The model performs both:

- image-level classification
- manipulation localization

using CLIP-guided weak supervision.

---

## 1. Overall Pipeline

During training:

1. input image is passed through the CLIP image encoder
2. multi-layer patch features are extracted
3. an adaptive soft edge generator produces a soft edge map
4. ECPL uses patch features + soft edge map to generate edge-aware prompts
5. these prompts are concatenated with:
   - REAL prompt `T_R`
   - FAKE prompt `T_F`
6. frozen text encoder produces text features
7. HECL selects high-confidence edge patches from selected image layers
8. dual queues and contrastive loss are used to strengthen edge feature discrimination

During inference:

- multi-layer patch-text similarity maps are computed
- maps are aggregated for localization
- image-level classification is determined by the prompt with highest similarity score

So SAPL jointly learns:

- text-side prompts biased toward edge artifacts
- image-side edge representations across hierarchy

---

## 2. Edge-aware Contextual Prompt Learning (ECPL)

### 2.1 Motivation

The paper argues that standard CLIP prompts emphasize:

- semantic object concepts

but manipulation localization depends on:

- subtle low-level irregularities
- diverse manipulation traces
- boundary inconsistencies

Because manipulation patterns are diverse, a single global prompt is insufficient.

So ECPL aims to:

- learn prompts automatically
- inject edge cues into prompt generation
- make prompts focus on non-semantic manipulation signals rather than object semantics

---

### 2.2 Adaptive Soft Edge Generator

The paper states that standard binary edge maps are too sparse and do not provide sufficient hierarchical detail.

To address this, it proposes an **Adaptive Soft Edge Generator**.

Procedure:

1. apply the Canny operator to obtain initial edge map
2. perform distance transform on non-edge regions
3. compute Euclidean distance from each pixel to nearest edge
4. dynamically compute an exponential decay factor `k` according to edge density
5. apply exponential decay to obtain a soft edge map

Soft edge formula:

- `E_soft(x, y) = exp(-k * min_(i,j in C(I)) ||(x-i, y-j)||_2 )`

where:

- `k` is adaptively adjusted according to edge density
- `I` is the input image
- `C(I)` is the edge set extracted by Canny

Behavior:

- pixels near boundaries get weights close to 1
- pixels farther away decay smoothly toward 0

Intended benefit:

- preserve precise boundaries
- create continuous multi-scale weights around boundaries
- supply richer edge information than a hard binary edge map

---

### 2.3 Learnable Global Text Prompts

The method introduces learnable prompts for the binary classes:

- REAL
- FAKE

Prompt templates are:

- `T_R = [R1][R2]...[Real][Image]`
- `T_F = [F1][F2]...[Fake][Image]`

These learnable prompt tokens are intended to encode diverse manipulation characteristics in the text embedding space rather than standard semantic labels.

Purpose:

- amplify anomalous signals from manipulated regions during image-text alignment
- improve sensitivity to non-semantic details

---

### 2.4 Edge-aware Contextual Prompt Generation

Because a single global prompt is still insufficient, SAPL combines:

- adaptive soft edge map
- multi-layer patch features
- learnable prompts

using multi-head self-attention.

The edge-aware prompt generation is defined as:

- `Q = f_cls`
- `K = P(sum_i f_patch_i)`
- `V = P(sum_i f_patch_i) ∘ E_soft(x, y)`
- `T_ECPL = MHSA(Q, K, V)`

where:

- `f_cls` is the CLS feature from image encoder
- `f_patch_i` is the patch feature from layer `i`
- `P(.)` is a projection function
- `E_soft(x, y)` is the soft edge map
- `∘` is element-wise multiplication

Interpretation:

- patch features provide contextual image information
- soft edge map modulates values to emphasize boundary-sensitive content
- CLS token acts as query to summarize/manipulation-condition prompts

The resulting `T_ECPL` is then concatenated with learnable REAL/FAKE prompts before passing into the frozen text encoder.

Purpose:

- preserve generalizability of learnable prompts
- adapt prompts to image-specific boundary variations
- focus prompts on manipulation artifacts rather than semantics

---

## 3. Hierarchical Edge Contrastive Learning (HECL)

### 3.1 Motivation

Even with edge-aware prompting in the text domain, the paper argues that this is not enough to fully capture latent manipulation patterns in the image domain.

Challenges:

- manipulated regions may be small
- boundaries may be blurred
- semantics may be similar between manipulated and authentic regions

Therefore HECL is introduced to:

- explore local manipulation patterns at patch level
- strengthen the separability of manipulated vs authentic edges
- provide complementary supervision in the image feature domain

The key assumption is:

- manipulation artifacts are especially prominent along edges

---

### 3.2 Patch Selection with Edge-weighted Similarity

At selected feature layers, HECL constructs similarity maps between:

- image patch features
- learnable text prompt features

Then these similarity maps are weighted by the soft edge map.

For each selected layer `i`, it selects the top `K` high-confidence patch locations for:

- manipulated edge positives
- authentic edge negatives

These are denoted:

- `{f_e^+}^K`
- `{f_e^-}^K`

The paper defines:

- `{f_e^+}^K = { {sim(f_patch_i, t^+) ∘ E_soft(x, y)} ∘ f_patch_i }^K`
- `{f_e^-}^K = { {sim(f_patch_i, t^-) ∘ E_soft(x, y)} ∘ f_patch_i }^K`

Then a projection head maps them into a unified embedding space:

- `z_i^+ = P({f_e^+}^K)`
- `z_i^- = P({f_e^-}^K)`

where:

- `t^+` and `t^-` are text prompt features for manipulated/authentic classes
- `P(.)` is the projection head

Purpose:

- select high-confidence edge-aligned features
- focus contrastive learning on the most informative boundary patches

---

### 3.3 Dual Queues

To avoid overfitting to perturbations from a single image or small batch, HECL uses:

- one positive queue
- one negative queue

These queues act as dynamic memory banks storing recent feature vectors.

Design motivation:

- increase sample diversity
- improve training stability
- reduce the impact of ambiguity or mis-sampled instances under weak supervision

Update mechanism:

- newly extracted samples are pushed into the queue
- oldest samples are popped out

This ensures the queue contains:

- recent dynamic features
- sufficient historical context

---

### 3.4 Hierarchical Contrastive Loss

HECL is applied across multiple feature layers, forming a hierarchical structure from coarse to fine features.

For each selected layer `i`, define:

- positive sample feature `z_i^+`
- negative sample feature `z_i^-`

Positive key:

- `p_i` = prototype in positive queue with highest cosine similarity to `z_i^+`

Negative key:

- `n_i` = prototype in negative queue with highest cosine similarity to `z_i^-`

All positive queue entries:

- `p_j`, `j = 1,...,L2`

All negative queue entries:

- `n_j`, `j = 1,...,L2`

Positive contrastive loss:

- `L_pos_i = -log( exp(sim(z_i^+, p_i)/tau) / ( exp(sim(z_i^+, p_i)/tau) + sum_j exp(sim(z_i^+, n_j)/tau) ) )`

Negative contrastive loss:

- `L_neg_i = -log( exp(sim(z_i^-, n_i)/tau) / ( exp(sim(z_i^-, n_i)/tau) + sum_j exp(sim(z_i^-, p_j)/tau) ) )`

Overall HECL loss:

- `L_HECL = (1/N) sum_i (L_pos_i + L_neg_i)`

where:

- `N` is the number of Edge Contrastive Learning blocks / selected layers

Interpretation:

- manipulated edge features should be close to manipulated-edge memory
- authentic edge features should be close to authentic-edge memory
- manipulated and authentic edge representations should repel one another

The hierarchical design is claimed to:

- capture large-scale perturbations in lower layers
- focus on fine-grained artifacts in higher layers

---

## 4. Classification and Inference

The framework still performs CLIP-style image-text similarity for binary classification.

Text features are derived from:

- REAL prompt branch
- FAKE prompt branch
- edge-aware prompt augmentation

Classification score:

- similarity between image representation and text prompt representation

Inference:

1. compute patch-text similarity maps at multiple selected layers
2. aggregate these maps to produce localization result
3. select image-level class based on the prompt with highest similarity score

So the localization comes from multi-layer similarity maps, not from a separate decoder.

---

## 5. Loss Function

SAPL uses two losses:

1. binary classification loss `L_cls`
2. hierarchical edge contrastive loss `L_HECL`

Overall loss:

- `L_all = L_cls + w(t) * L_HECL`

where:

- `w(t)` is a linearly increasing weight schedule

The purpose of this schedule is:

- early training focuses on basic semantic alignment / classification
- later training increasingly emphasizes precise edge-level distinction

This means HECL is introduced progressively rather than with a fixed constant weight from the beginning.

---

## 6. Core Design Summary

SAPL is built around the idea that manipulation localization should rely on:

- edge anomalies
- low-level perturbations
- semantic-agnostic prompts

rather than conventional semantic category prompting.

The two key strategies are:

1. **ECPL**
   - use soft edge maps and patch features to generate edge-aware text prompts
2. **HECL**
   - select high-confidence edge patches and perform hierarchical contrastive learning with dual queues

Together they provide:

- text-domain bias toward boundary artifacts
- image-domain discrimination of manipulated vs authentic edges

---

## 7. Components That Look Most Distinctive

Without analysis yet, the most distinctive pieces in this paper are:

1. **adaptive soft edge map** instead of binary edge supervision
2. **semantic-agnostic prompt learning** centered on REAL/FAKE prompts
3. **edge-conditioned prompt generation via MHSA**
4. **top-K edge patch selection weighted by text similarity and soft edge map**
5. **dual positive/negative memory queues**
6. **hierarchical edge contrastive learning across feature depths**
7. **linearly scheduled HECL weighting**

---

## 8. Immediate Notes for Later Analysis

These are placeholders only, not conclusions yet:

- the prompts are intentionally semantic-agnostic and edge-centric
- edge maps influence both text prompt generation and image-domain contrastive learning
- the method reinforces localization through multi-layer patch-text similarity rather than a heavy decoder
- the queue-based contrastive design is explicitly built to stabilize weak supervision
