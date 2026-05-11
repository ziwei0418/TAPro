# Method: AB-TAGD for Weakly Supervised Histopathology Segmentation

This note is the Markdown version of the MICCAI-style method section. It describes the WSSS method implemented in the current codebase, without placing concrete hyperparameter values in the method text.

## Problem Setting

We address weakly supervised semantic segmentation for histopathology. Each training image has only image-level tissue labels, not pixel masks. Given an H&E patch \(x\) and a multi-label vector \(y \in \{0,1\}^{C}\), the goal is to produce dense pseudo masks for the parent tissue classes.

The method uses no CRF, no TTA, and no SAM. The main contribution is the pseudo-mask generation pipeline, especially the activation-balanced attention-boundary tissue-adaptive graph diffusion module.

## 1. Pathology Feature Pyramid

A frozen pathology foundation encoder extracts dense image tokens. A lightweight feature pyramid converts them into multi-scale feature maps:

\[
\{F_{\ell}\}_{\ell=1}^{4} = \mathrm{FPN}(E(x)),
\qquad
F_{\ell} \in \mathbb{R}^{B \times d_{\ell} \times H_{\ell} \times W_{\ell}} .
\]

Lower pyramid levels preserve finer spatial detail, while deeper levels carry stronger tissue semantics. The encoder is frozen; the trainable parts are the FPN/projection heads and scale-specific logit parameters.

## 2. Hybrid Text-Visual Tissue Prototypes

For each subclass \(k\), the model has:

- a visual prototype \(v_k\),
- a text prototype \(t_k\),
- a parent class assignment \(\pi(k)\).

The hybrid prototype is:

\[
p_k =
\frac{
    \alpha_{\pi(k)} \bar{t}_k +
    (1-\alpha_{\pi(k)}) \bar{v}_k
}{
    \left\|
    \alpha_{\pi(k)} \bar{t}_k +
    (1-\alpha_{\pi(k)}) \bar{v}_k
    \right\|_2
}.
\]

Here \(\bar{t}_k\) and \(\bar{v}_k\) are normalized text and visual prototypes. The coefficient \(\alpha_{\pi(k)}\) controls how much the parent class relies on text versus visual information.

Each hybrid prototype is projected to each feature-pyramid level:

\[
p_{\ell,k} = \mathrm{Proj}_{\ell}(p_k).
\]

## 3. Multi-Scale Prototype CAM Learning

At each pyramid level, subclass CAMs are computed by cosine similarity between dense features and projected prototypes:

\[
A_{\ell,k}(u)
=
s_{\ell}
\left\langle
\frac{F_{\ell}(u)}{\|F_{\ell}(u)\|_2},
\frac{p_{\ell,k}}{\|p_{\ell,k}\|_2}
\right\rangle .
\]

Image-level subclass predictions are obtained by global average pooling:

\[
q_{\ell,k} = \mathrm{GAP}(A_{\ell,k}).
\]

Subclass predictions are merged into parent tissue predictions:

\[
q_{\ell,c}^{\mathrm{par}}
=
\mathrm{Merge}\left(\{q_{\ell,k}: k \in \mathcal{K}_c\}\right).
\]

The classification loss uses only image-level labels:

\[
\mathcal{L}_{\mathrm{cls}}
=
\sum_{\ell=1}^{4}
\rho_{\ell}
\,
\mathrm{BCE}\!\left(q_{\ell}^{\mathrm{par}}, y\right).
\]

The deepest CAMs are also used to mine foreground and background tokens for prototype regularization:

\[
\mathcal{L}_{\mathrm{proto}}
=
\mathcal{L}_{\mathrm{FG}}
+
\mathcal{L}_{\mathrm{BG}}.
\]

The total training loss is:

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{cls}}
+
\lambda_{\mathrm{proto}}\mathcal{L}_{\mathrm{proto}}
+
\lambda_{\mathrm{cam}}\mathcal{R}_{\mathrm{cam}}.
\]

## 4. Parent CAM Fusion and Activation Balance

At inference, subclass CAMs are merged into parent CAMs:

\[
A_{\ell,c}^{\mathrm{par}}(u)
=
\mathrm{Merge}\left(
\{A_{\ell,k}(u): k \in \mathcal{K}_c\}
\right).
\]

Absent classes are suppressed using the image-level labels. The CAMs used for pseudo-mask generation are fused across selected scales:

\[
S_c
=
\sum_{\ell \in \mathcal{M}}
\omega_{\ell} A_{\ell,c}^{\mathrm{par}},
\qquad
\sum_{\ell \in \mathcal{M}}\omega_{\ell}=1.
\]

Raw fused CAMs are biased by tissue size and morphology. To reshape class competition, the method applies class-wise power calibration:

\[
\widehat{S}_c(u)
=
m(u)
\frac{
    (S_c(u)+\epsilon)^{\gamma_c}
}{
    \sum_{j=1}^{C}
    (S_j(u)+\epsilon)^{\gamma_j}
},
\qquad
m(u)=\sum_{j=1}^{C}S_j(u).
\]

The balanced CAM is a residual interpolation:

\[
\widetilde{S}_c
=
(1-\eta)S_c + \eta\widehat{S}_c.
\]

This preserves the total foreground activation mass while changing the relative competition among classes.

## 5. Attention-Boundary Tissue-Adaptive Graph Diffusion

The balanced CAMs are refined on a token graph built from the deepest feature map. Let \(F_4\) be flattened into normalized token features \(\{f_i\}_{i=1}^{n}\). Feature similarity is:

\[
K_{ij} = \langle f_i, f_j \rangle .
\]

The graph keeps sufficiently similar token pairs plus self-edges:

\[
\mathcal{N}(i)
=
\{j : K_{ij} \geq \delta\} \cup \{i\}.
\]

To avoid diffusion across true tissue boundaries, the method uses self-attention profiles from the frozen encoder. Let \(a_i\) be the attention profile of token \(i\). The attention-boundary distance is:

\[
D_{ij}
=
\frac{1}{2}
\left\|
a_i-a_j
\right\|_1 .
\]

The graph logits combine feature similarity and boundary penalty:

\[
L_{ij}
=
\frac{K_{ij}}{\tau}
-
\beta D_{ij},
\qquad j \in \mathcal{N}(i).
\]

The normalized graph is:

\[
W_{ij}
=
\frac{\exp(L_{ij})}
{\sum_{r \in \mathcal{N}(i)} \exp(L_{ir})}.
\]

For class \(c\), the balanced CAM is downsampled to graph resolution and denoted \(s_c\). The one-step class-adaptive diffusion update is:

\[
z_c
=
(1-\lambda_c)s_c
+
\lambda_c W s_c .
\]

The refined maps are upsampled and converted to a pseudo mask by foreground argmax:

\[
\hat{Y}(u)
=
\arg\max_{c \in \{1,\ldots,C\}} Z_c(u).
\]

## Design Rationale

Repeated graph diffusion acts as a low-pass filter and can erase small histological structures. AB-TAGD uses only one residual diffusion step. The feature graph propagates CAM activations inside coherent tissue regions, while the attention-boundary penalty suppresses propagation across likely tissue boundaries. Class-adaptive diffusion weights allow compact classes and diffuse classes to use different propagation strengths.

