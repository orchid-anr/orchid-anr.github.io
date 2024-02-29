---
layout: review
title: "A Rigourous Study of the Deep Taylor Decomposition"
tags: network interpetability, theoritical framework
author: "Jeremie"
cite:
    authors: "Leon Sixt, Tim Landgraf"
    title:   "A Rigourous Study of the Deep Taylor Decomposition"
    venue:   "arXiv"
pdf: "https://arxiv.org/abs/2211.08425"
---

# Note

This paper was published in Transctions on Machine Learning Research (11/2022)

# A rigorous study of the DTD

The authors discuss the **Deep Taylor Decomposition** to compute the relevance of an input regarding the output. When combined with the **Layer-wiser Relevancy Propagation**, this method seems to provide a rigorous and theoretically founded explanation method, starting from the output backward to the input, then attributing saliency scores to each input elements.

However, recent works have proven that the DTD method is actually equivalent to the input x gradient explanation method, which makes it less consistent. This is due to the choice of the ***root point*** in the vicinity of which the Taylor formula is expanded. This raises major issues:

 - LRP and DTD backpropagation rules create explanations partially independent of the model parameters, especially of the last layers
 - These methods appear to respond only to lower-level image structure
 - Jacobian matrix quickly tend to a 1-ranked matrix, and then the Taylor root points are locally constants (locally input-dependent)

The authors argue DTD root points do not lie in the same linear region than the input, contrary to a fundamental assumption of the Taylor theorem.

## Theoretical framework

Let's recall Taylor's Theorem, applied to the **locally linear function ReLU**:

$$
f(x) = f(\tilde{x}) + \left.\frac{\partial f(x)}{\partial x}\right|_{x=\tilde{x}}\cdot (x-\tilde{x})
$$

 - Where $\tilde{x}$ is the appointed ***root point***. Since the Taylor's Theorem requires that $f\in \mathcal{C}^1$, **the root point must be in the same ***linear region*** as the input**
 - The higher order terms are zero due to the local linearity of ReLU networks

But directly using the Taylor's Theorem on the network output to compute the relevancy score entails several issues:

 - **Adversarial perturbations**: small input perturbations can lead to a large change in the output, with a enormous difference between the output but a tiny $|x-\tilde{x}|$: uninterpretable 
 - **Difficult root point**: solvability not guaranteed, especially when the minimization problem is not convex:

$$
\min_\xi ||\xi - x ||^2 \quad \textit{s.t.}\quad f(\xi) = 0
$$

 - **Shattered gradient**: while the value of $f(x)$ is generally accurate, the gradient of the function is noisy

Therefore, the relevance of the input is defined to be the point-wise product of the partial derivatives with the input differences:

$$
R(x) = \left.\frac{\partial f(x)}{\partial x}\right|_{x=\tilde{x}}\odot (x-\tilde{x})
$$

with the element-wise product instead (to see the element-wise effect of the input shift, before summing as a dot product).

Then, in a **ReLU network** setup with $n$ layers, we can see the function $f$ as follows:

$$
f = f_n\circ f_{n-1}\circ \ldots \circ f_1
$$

where $f_l :  \mathbb{R}^{d_l} \to \mathbb{R}^{d_{l+1}}_{\geq 0}$ has the form $f_l(a_l) = [W_l a_l]^+$

The idea is to recursively apply the Taylor theorem to the layers, starting from the output layer to the input. 

- $R_{[j]}^{l+1}(a_{l+1})$ is the relevance of $a_l$ (the input to layer $l$) to the $j$-th coordinate of $R^{l+1}(a_{n+1})$
- $\tilde{a}_l$ be the root point chosen depending on the layers's input $a_l$ ($\tilde{a}_l(.)$ is a function) in the **linear region** of $a_l$. Therefore, according the the Taylor theorem, it follows that:

$$
R_{[j]}^{l+1}(a_{l+1}) = R_{[j]}^{l+1}(f_l(\tilde{a}_l)) + \left.\frac{\partial R_{[j]}^{l+1}(f_l(a_l))}{\partial a_l}\right|_{a_l=\tilde{a}_l(a_l)} \cdot (a_l - \tilde{a}_l(a_l))
$$

- ⚠️ Note that $\frac{\partial R_{[j]}^{l+1}(f_l(a_l))}{\partial a_l} \in \mathbb{R}^{d_{l}}$, it is the relevancy of $a_l$ computed for the $j$-th neuron in the $(l+1)$ layer. Then, the total relevance of the input $a_l$ to the layer $l$ is given by the sum over all $d_{l+1}$ hidden neurons.
- At the base of the recursive application, the relevance of the network output is set to the value of the explained logit $a_{n+1_{[\xi]}}$. Then the relevance input computed with the **recursive Taylor method** is:

$$
R_{[j]}^{l+1}(a_{l+1}) = R_{[j]}^{l+1}(f_l(\tilde{a}_{l})) + \left.\frac{\partial R_{[j]}^{l+1}(f_l(a_l))}{\partial a_l}\right|_{a_l=\tilde{a}_l(a_l)} \cdot (a_l - \tilde{a}_l(a_l))
$$

## Concrete example 

For instance, we can start from a point $x$ that sets the relevance of the $j$-th neuron to $0$, i.e. $w_j^\top x + b_j=0$ because the idea is to compute the output with the first-order derivate of the Taylor formula, which is possible in that case. To remain in the **linear region** of $x$, we can chose a direction $v_j$ such that for any $t\in \mathbb{R}$:

 - $\tilde{x}_j$ is in the intersection of the line $x + t v_j$
 - and of the hyperplan defined by $w_j^\top x + b_j = 0$, because the advantage of $f(\tilde{x}) = 0$ is that the output is absorbed to the first-order term of the Taylor's formula.

The equations above give $t = - \frac{w_j^\top x + b_j}{w_j^\top v_j}$ and then 

$$
\tilde{x}_j = x - \frac{w_j^\top x + b_j}{w_j^\top v_j}v_j
$$

which results in the following equation for a one-layer ReLU network:

$$
R^x(x) = \sum_{j=1}^d\left(w_j \odot \frac{w_j^\top x + b_j}{w_j^\top v_j}v_j\right)
$$

where the relevance is computed for the input $x$ and where $w_j^\top x + b_j = R_{[j]}(x)$, the relevance of the input of the hidden layer. 

Then, there are several rules to select a root point and to compute the relevance propagation from it, **these rules give a method to find the direction** $v_j$ to compute the root $\tilde{x}_j$

This formula hardly generalizes to deep network with several layers, it need strong assumptions about the layer and the associated relevancy. 

 - Assuming that $R^{l+1}(a_{l+1}) = a_{l+1}\odot c_{l+1}$ where $c_{l+1}$ is a positive constant does not take into account the possible huge impact of a small input variation
 - The root point $\tilde{a}_l$ in note ensured to be in the same linear region as $a_l$

## Analysis and Critique of the Recursive Application of the Taylor Theorem

- The valid region for root points of the relevancy score is restricted by the network $f$, then the size of admissible regions for the root points cannot be increased
Let's denote $\mathcal{N}_f(x)$ the set of all points $x'\in \mathcal{X}$ that have the same gradient as the input $x$ and for which all points in the line between $x$ and $x'$ have the same gradient as well.
Then the previous proposition implies that $\mathcal{N}_{R^l}(a_l) \subseteq \mathcal{N}_{f_{n_\xi}\circ\ldots\circ f_l}(a_l)$.
- Locally constant roots imply equivalence of recursive Taylor method and the Gradient x Input method, meaning that: $R(x) = R(\tilde{x}) + \nabla f_{[\xi]}(x) \odot (x-\tilde{x})$, which implies the equivalence when $\tilde{x}=0$.
- The choice of the root point is not very well restricted. It must be from the linear region $\mathcal{N}_{R_{[k]}^l}(a_l)$, but the relevancy $R^{(l-1)}(a_{(l-1)})$ depends upon the Jacobian $\partial \tilde{a}_l^{(j)}/\partial a_l$, and thus, it depends of the choice of the root point. This can lead to several arbitrary explanation results
- The problem with choosing another activation function (like Softplus for instance) is that the Taylor's formula includes a lot more non-zero high-order terms: for a $n$-layered network, we would get $n$-ordered derivatives. The problem is that it is unclear how these chains of high-order derivatives behave
