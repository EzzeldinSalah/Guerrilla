# Mathematical Specifications & Derivations

This document details the complete mathematical formulation for the N-Layer Transformer Encoder forward pass, loss functions, backpropagation gradients, and optimizers implemented in Guerrilla.

## 1. N-Layer Transformer Encoder Workflow

The forward pass transforms an input sequence $X \in \mathbb{R}^{T \times D_{\text{model}}}$ through $N$ identical encoder layers:

### A. Multi-Head Self-Attention (MHA)
1. **Query, Key, Value Projections:**
   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad \in \mathbb{R}^{T \times D_{\text{model}}}$$
2. **Head Slicing:** $Q, K, V$ are sliced along column boundaries into $h$ heads of dimension $d_k = D_{\text{model}} / h$.
3. **Scaled Dot-Product Attention (per head $i$):**
   $$\text{Attention}(Q_i, K_i, V_i) = \text{Softmax}\left( \frac{Q_i K_i^T}{\sqrt{d_k}} \right) V_i$$
   Softmax includes numerical max-subtraction ($\text{exp}(x - x_{\max})$) per row to prevent floating-point overflow.
4. **Concatenation & Output Projection:**
   $$\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$

### B. Sub-Layer 1: Residual Connection & Layer Normalization
$$X_{\text{mid}} = \text{LayerNorm}(X + \text{MHA}(X))$$
Row-wise layer normalization (without learnable affine parameters $\gamma, \beta$) computes:
$$\text{LN}(x_{i,j}) = \frac{x_{i,j} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$
Where $\mu_i = \frac{1}{D} \sum_{j=1}^D x_{i,j}$, $\sigma_i^2 = \frac{1}{D} \sum_{j=1}^D (x_{i,j} - \mu_i)^2$, and $\epsilon = 10^{-5}$.

### C. Sub-Layer 2: Feed-Forward Network (FFN)
Passes activations through a two-layer feed-forward network with LeakyReLU ($\alpha = 0.01$):
$$\text{FFN}(X_{\text{mid}}) = \text{LeakyReLU}(X_{\text{mid}} W_1 + B_1) W_2 + B_2$$

### D. Sub-Layer 2: Residual Connection & Layer Normalization
$$\text{Output} = \text{LayerNorm}(X_{\text{mid}} + \text{FFN}(X_{\text{mid}}))$$

### E. Mean Pooling & Classification Head
1. **Mean Pooling:** Averages token vectors across the temporal dimension $T$:
   $$P_j = \frac{1}{T} \sum_{i=1}^T \text{Output}_{i,j} \quad \in \mathbb{R}^{1 \times D_{\text{model}}}$$
2. **Logits & Softmax:** Linear projection to classes followed by Softmax:
   $$\text{logits} = P W_{\text{class}} + B_{\text{class}}, \quad \hat{y} = \text{Softmax}(\text{logits}) \quad \in \mathbb{R}^{1 \times C}$$
3. **Cross-Entropy Loss:** For ground-truth class label $y \in \{0, \dots, C-1\}$:
   $$L = -\ln(\hat{y}_y)$$

## 2. Gradient Derivations & Backpropagation

### A. Cross-Entropy & Softmax Gradient
For single sample cross-entropy paired with softmax:
$$\frac{\partial L}{\partial \text{logits}_j} = \hat{y}_j - \mathbf{1}(j = y)$$

### B. Matrix Multiplication Gradients
For $C = A B$ where $A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}, C \in \mathbb{R}^{M \times N}$:
$$\nabla_A = \nabla_C B^T, \quad \nabla_B = A^T \nabla_C$$

### C. Layer Normalization Backward
Given upstream gradient $dY = \frac{\partial L}{\partial Y}$ for $Y = \text{LN}(X)$:
Let $\hat{x}_j = \frac{x_j - \mu}{\sigma_{\epsilon}}$ where $\sigma_{\epsilon} = \sqrt{\sigma^2 + \epsilon}$.
$$\frac{\partial L}{\partial x_j} = \frac{1}{\sigma_{\epsilon}} \left( dY_j - \bar{dY} - \hat{x}_j \cdot \overline{dY \cdot \hat{x}} \right)$$
Where row averages are:
$$\bar{dY} = \frac{1}{D} \sum_{k=1}^D dY_k, \quad \overline{dY \cdot \hat{x}} = \frac{1}{D} \sum_{k=1}^D dY_k \cdot \hat{x}_k$$

### D. LeakyReLU Backward
For $Y = \text{LeakyReLU}(X, \alpha)$:
$$\frac{\partial L}{\partial X_{i,j}} = \begin{cases} \frac{\partial L}{\partial Y_{i,j}} & \text{if } X_{i,j} > 0 \\ \alpha \cdot \frac{\partial L}{\partial Y_{i,j}} & \text{if } X_{i,j} \le 0 \end{cases}$$

### E. Scaled Dot-Product Attention Backward
Given $O = \text{Softmax}(S) V$ where $S = \frac{Q K^T}{\sqrt{d_k}}$:
1. **Value Gradient:** $dV = A^T dO$
2. **Softmax Weights Gradient:** $dA = dO V^T$
3. **Score Matrix Gradient:** $dS_{ij} = A_{ij} \left( dA_{ij} - \sum_k dA_{ik} A_{ik} \right)$
4. **Query & Key Gradients:**
   $$dQ = \frac{dS \cdot K}{\sqrt{d_k}}, \quad dK = \frac{dS^T \cdot Q}{\sqrt{d_k}}$$

## 3. Optimizers

### SGD (`sgd` / `sgdTransformer`)
$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot \nabla_\theta L$$

### Adam (`adam` / `adamTransformer`)
Tracks first ($m$) and second ($v$) moment estimates:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
Bias correction:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
Parameter update:
$$\theta^{(t+1)} = \theta^{(t)} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
*(Default hyperparameter settings: $\eta = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$)*