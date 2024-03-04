---
layout: review
title: "Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers"
tags: transformer interpretability, custom attention rollout
author: "Jeremie"
cite:
    authors: "Hila Chefer, Shir Gur, Lior Wolf"
    title:   "Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers"
    venue:   "arXiv"
pdf: "https://arxiv.org/abs/2103.15679.pdf"
---


# Note

Chefer, Hila, Shir Gur, et Lior Wolf. « Generic Attention-Model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers ». In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 387‑96. Montreal, QC, Canada: IEEE, 2021. https://doi.org/10.1109/ICCV48922.2021.00045.

# Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers

There exist three types of transformers, with their respective associated explanation methods:

 - pure self-attention
 - self-attention combined with co-attention
 - encoder-decoder attention (generative models, with encoder's inputs and outputs being from a different domain)

The multi-modal Transformers changed the state of the art enabling zero-shot performance and conditional generation.
There are also two main methods to merge two modalities in a Transformer:

 - applying a symmetric contrastive-loss 
 - concatenation

In computer vision, an explainability method can be:

 -  class-dependent (class influences the relevancy of each image location)
 - class-agnostic (depends only on the input and the model)

## Method
Their method build **relevancy maps** computed for each interaction within and between each modalities of the model. 

 - The relevancy matrices are calculated and then updated by a forward pass through the attention layers. The final attention map $\mathbf{\bar{A}}\in \mathbb{R}^{s \times q}$ where $s$ and $q$ are the number of tokens for each respective modality is computed as follows
 
$$
\begin{equation}
\mathbf{\bar{A}} = \mathbb{E}_h((\nabla \mathbf{A}\odot \mathbf{A})^+)
\end{equation}
$$

where $\nabla \mathbf{A}:= \frac{\partial y_t}{\partial \mathbf{A}}$ and where $y_t$ is the targeted class. 

- $\mathbf{A}^{tt}$ and $\mathbf{A}^{ii}$ represent the self-attention interactions for the text and image tokens respectively ;  $\mathbf{A}^{ti}$ and $\mathbf{A}^{it}$ represent the influence of the image tokens on each text token, and the influence of the text tokens on each image token, respectively 
- For the **self-attention layers**, with one single modality, $s=q$ (where $s$ and $q$ indicate the domains and the number of tokens in each domain, can be $s$ query tokens and $q$ key tokens). The relevancy updating follows the following rule for self-attention layers that satisfy $\mathbf{\bar{A}} \in \mathbb{R}^{s \times s}$:

$$
\mathbf{R}^{ss} = \mathbf{R}^{ss} + \mathbf{\bar{A}} \cdot \mathbf{R}^{ss} \\
\mathbf{R}^{sq} = \mathbf{R}^{sq} + \mathbf{\bar{A}} \cdot \mathbf{R}^{sq}
$$

the second line is only applied when there is also co-attention modules in the network (in that case $q\neq s$)

- For the **bi-modal layers**, $\mathbf{\bar{A}} \in \mathbb{R}^{s \times q}$ and the update rule is:

$$
\mathbf{R}^{sq} = \mathbf{R}^{sq} + (\mathbf{\bar{R}}^{ss})^\top \cdot \mathbf{\bar{A}}  \cdot \mathbf{\bar{R}}^{qq} \\
\mathbf{R}^{ss} = \mathbf{R}^{ss} + \mathbf{\bar{A}}  \cdot \mathbf{\bar{R}}^{qs} 
$$

where $\mathbf{\bar{R}}^{ss}$ is the row-normalized of $\mathbf{\hat{R}}^{ss}=\mathbf{R}^{ss} - \mathbf{I}^{s \times s}$, the whole added to $\mathbf{I}^{s \times s}$ (the purpose is to separate the influence of each token on the other $i$-th token, and the identity matrix setting that value for each token w.r.t to itself to $1$).

$$
\begin{aligned}
\mathbf{\hat{S}}_{m,n}^{xx} = \sum_{k=1}^{|x|} \mathbf{\hat{R}}^{xx}_{m,k} \\
\mathbf{\bar{R}}^{xx} = \mathbf{\hat{R}}^{xx}/\;\mathbf{\hat{S}}_{m,n}^{xx} + \mathbf{I}^{x\times x}
\end{aligned}
$$

- Note that the second equation in each update rule accounts for the interaction between two modalities (*e.g.* when the previous bi-model attention layer inserts context from $q$ into $s$, the self-attention matrix $\mathbf{\bar{A}} \in \mathbb{R}^{s \times s}$ mixes the context $q$ in each token from $s$ as well)
- Finally, note that to retrieve per-token relevancies for classification task, one can consider the row corresponding to the [CLS] token in the corresponding relevancy map. If there are two modalities, we consider $\mathbf{R}^{ss}$ and $\mathbf{R}^{sq}$, where the [CLS] token was added to the "$s$" modality. 

## Experiments and results
The baselines used for comparison purposes are:

 - Raw attention (using last layer's attention map)
 - Rollout attention

$$
\begin{aligned}
\mathbf{\hat{A}}^{(b)} & = \mathbf{I} + \mathbb{E}_h\mathbf{A}^{(b)}\\
\mathbf{R}^{xx} & = \mathbf{\hat{A}}^{(1)} \cdot \mathbf{\hat{A}}^{(2)} \cdot \ldots \cdot \mathbf{\hat{A}}^{(B)} \\
\mathbf{R}^{sq} & = (\mathbf{R}^{ss})^\top \cdot \mathbf{\bar{A}} \cdot \mathbf{R}^{qq}
\end{aligned} 
$$

where $\mathbf{\bar{A}}\in\mathbb{R}^{s\times q}$ is the last bi-attention map

- Adapted Grad-CAM
- Partial LRP values of the last attention layer (average across the heads)
- Transformer attribution method, with $\mathbf{\bar{A}} =\mathbb{E}_h((\nabla \mathbf{A} \odot \mathbf{R}^{\mathbf{A}}))$

Two models are studied:

 - VisualBERT, self-attention based architecture 
 - LXMERT, combining self-attention and co-attention (2 modalities)

Each time, the area-under-the-curve (AUC) to evaluate the decrease in the model's accuracy (**perturbation test**) is measured.

![](/collections/images/chefer_bib/VisualBERT_AUC.jpg)
![](/collections/images/chefer_bib/LXMERT_AUC.jpg)

This methods shows improvement in performance when using the target class instead of the predicted class for gradient propagation

![](/collections/images/chefer_bib/tableres_genericAttention.jpg)

## Conclusions

Explainability can be very useful to debug, to support downstream tasks and to better interpret models such as transformers.

 1. The proposed method uses and transforms raw attention, proposes a specific averaging technique to take into account the importance differences between attention heads, and mixes the input and the context relevancies for each layer
 2. This method outperforms the state-of-the-art in perturbation method tests (AUC-ROC), and is very close to Transformer attribution results (*cf.* Chefer et al. Transformer Visualisation Beyond Attention) in the self-attention case.
 3. It contributes in the way that most of the explainability literature focuses on pure self-attention maps, regardless of the co-attention maps. **This method provides a way to compute relevancy maps for co-attention modules as well as self-attention modules**.
 4. This methods shows improvement in performance when using the target class instead of the predicted class for gradient propagation