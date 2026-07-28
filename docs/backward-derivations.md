## 1. Layer Normalization
This implementation performs row-wise normalization without learnable scale ($\gamma$) or shift ($\beta$) parameters.

Given an input row vector $x$ of size $d$, row mean $\mu$, and variance $\sigma^2$:

$$y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

Let $\text{std} = \sqrt{\sigma^2 + \epsilon}$.

### Gradient
Let $dy_i$ ($\frac{\partial L}{\partial y_i}$) be the incoming gradient from the subsequent layer for this element.

$$\frac{\partial L}{\partial x_i} = \frac{1}{\text{std}} \left( dy_i - \text{mean}(dy) - y_i \cdot \text{mean}(dy \cdot y) \right)$$

Where the averages are evaluated across the single row dimension:
* $\text{mean}(dy) = \frac{1}{d} \sum_{j=1}^{d} dy_j$
* $\text{mean}(dy \cdot y) = \frac{1}{d} \sum_{j=1}^{d} dy_j \cdot y_j$


## 2. Self-Attention
Given Attention matrix $A = \text{softmax}(S)$ where $S = \frac{Q K^T}{\sqrt{d_k}}$, and output $O = A V$.

### Gradients
Given downstream gradient $dO$:

* **Value Gradient ($dV$):**
  $$dV = A^T dO$$

* **Attention Weights Gradient ($dA$):**
  $$dA = dO V^T$$

* **Score Matrix Gradient ($dS$):**
  From the softmax derivative:
  $$dS_{ij} = A_{ij} \left( dA_{ij} - \sum_k dA_{ik} A_{ik} \right)$$

* **Query ($dQ$) and Key ($dK$) Gradients:**
  $$dQ = \frac{dS \cdot K}{\sqrt{d_k}}$$
  $$dK = \frac{dS^T \cdot Q}{\sqrt{d_k}}$$