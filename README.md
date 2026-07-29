# Guerrilla

This project is about building a small transformer from the ground up in pure C. The code keeps the tensor math, the training loop, the backpropagation chain rule, and the validation steps visible instead of hiding them behind a framework.

## Why ?

The reason for doing it this way is simple: the point is to understand how the machine actually behaves. If you can trace the data from input to logits and from loss back to gradients, then the framework stops feeling magical.

This is also why the codebase stays explicit. The tensor shapes, the loop order, the normalization formulas, and the attention derivations are all visible in the source and the docs.

If you are reading this because you want to build something similar, the practical path is to keep the math small, keep the memory layout simple, and verify every step against a known-good reference before adding the next layer.

## What exists right now

A complete tensor library, full forward and backward passes of a multi-layer transformer encoder, SGD and Adam optimizers, automated PyTorch mathematical validation, and CPU benchmarking.

```
Guerrilla/
├── include/
│   ├── attention.h
│   ├── encoder.h
│   └── tensor.h
├── src/
│   ├── attention.c
│   ├── encoder.c
│   ├── main.c
│   └── tensor.c
├── training/
│   ├── attentionGrad.c / attentionGrad.h
│   ├── encoderGrad.c / encoderGrad.h
│   ├── lossFunctions.c / lossFunctions.h
│   ├── optimizer.c / optimizer.h
│   ├── tensorGrad.c / tensorGrad.h
│   └── trainLoop.c / trainLoop.h
├── tests/
│   ├── attentionTest.c
│   ├── backwardTest.c
│   ├── encoderTest.c
│   ├── tensorTest.c
│   ├── testUtils.c
│   └── tests.h
├── scripts/
│   ├── benchmark_vs_pytorch.py
│   └── validate_against_pytorch.py
├── docs/
│   ├── benchmarks.md
│   ├── contributing.md
│   ├── how-did-i.md
│   └── math.md
├── benchmarks/
│   ├── matmulBench.c
│   ├── speed_report.txt
│   └── validationReport.txt
├── data/
├── weights/
├── Makefile
├── requirements.txt
└── README.md
```

## Design & Architecture

Guerrilla implements a complete Transformer Encoder stack for sequence classification without external neural network libraries. Every tensor operation, forward activation, gradient accumulation, and optimizer update is executed via explicit C functions operating on contiguous memory blocks.

For complete mathematical derivations, forward equations, and analytical gradient formulas, see [docs/math.md](docs/math.md).

```
[ Input Tensor (T x D) ]
           │
           ▼
┌─────────────────────────────────────────┐
│        N-Layer Encoder Stack            │
│  ┌───────────────────────────────────┐  │
│  │ Multi-Head Attention              │  │
│  │ Residual Add + LayerNorm          │  │
│  │ Feed-Forward Network (LeakyReLU)  │  │
│  │ Residual Add + LayerNorm          │  │
│  └───────────────────────────────────┘  │  (x N Layers)
└─────────────────────────────────────────┘
           │
           ▼
[ Mean Pooling (1 x D) ]
           │
           ▼
[ Classification Head (Linear + Softmax) ] ──► [ Class Probabilities (1 x C) ]
                                                            │
                                                            ▼
                                                    [ Cross-Entropy Loss ]   
```

### 1. High-Level Forward Pipeline

The forward pipeline transforms sequence tokens into class probabilities through $N$ stacked encoder layers:

1. **Multi-Head Self-Attention:** Projects inputs to $Q, K, V$, slices them into $h$ heads, computes scaled dot-product attention per head, concatenates head outputs, and projects with $W_O$.
2. **Sub-Layer 1 Normalization:** Adds residual connection and applies row-wise Layer Normalization.
3. **Feed-Forward Network:** Projects normalized activations through a 2-layer FFN with LeakyReLU ($\alpha = 0.01$).
4. **Sub-Layer 2 Normalization:** Adds residual connection and applies row-wise Layer Normalization.
5. **Pooling & Classification:** Averages token vectors across the temporal dimension $T$ (`meanPool`), projects to class logits, and applies Softmax.

### 2. Backpropagation Engine & Caching

Guerrilla performs exact analytical backpropagation without dynamic graph building or autograd tape allocation.

```
                    BACKWARD FLOW
  [ dLoss / dLogits ] = (Probabilities - TrueClass) / BatchSize
           │
           ▼
  [ classificationHeadBackward ] ──► Gradients for classW, classB
           │
           ▼
  [ meanPoolBackward ] ────────────► Distributes dPooled across time steps
           │
           ▼
  [ encoderStackBackward ] ────────► Loops backward: Layer N-1 down to Layer 0
           │
           ├── Recomputes layer forward activations internally
           ├── layerNormBackward (computes analytical dLN wrt mean & variance)
           ├── leakyReluBackward & addBiasBackward (updates W2, B2, W1, B1 grads)
           ├── layerNormBackward (Sub-layer 1)
           ├── multiplyBackwardB & multiplyBackwardAData (updates W_O grad)
           └── singleHeadAttentionBackward (computes dQ, dK, dV & updates W_Q, W_K, W_V grads)
```

#### Memory Strategy: Caching Inputs & On-Demand Recomputation
- **Input Caching:** `trainForwardBackward()` caches only the input tensor to each encoder layer in `layerInputs[N+1]`.
- **Internal Recomputation:** `encoderLayerBackward()` recomputes internal layer forward activations on demand during the backward pass. This eliminates intermediate activation memory overhead across layers.

For step-by-step mathematical proofs and derivative formulas, refer to [docs/math.md](docs/math.md).

### 3. Training Loop & Optimizers

The training loop (`trainSgd` / `trainAdam`) operates in 4 steps per iteration:

1. **Zero Gradients:** `zeroTransformerGrad()` resets all parameter `.grad` buffers to `0.0f`.
2. **Forward & Loss:** Computes predictions and cross-entropy loss $L = -\ln(\hat{y}_{\text{true}})$.
3. **Backpropagation:** Executes backward functions to populate `.grad` across all parameter matrices.
4. **Parameter Update:** Updates parameter buffers (`.data`) using **SGD** or **Adam** (with first/second moment tracking $m_t, v_t$ and bias correction).

## Roadmap

**Forward pass:**
- [x] Tensor library
- [x] Single-head attention
- [x] Multi-head attention
- [x] tensorConcat = stitch head outputs back together
- [x] Feedforward block = two linear layers with leaky ReLU
- [x] Full encoder block = layernorm + attention + residual + layernorm + feedforward + residual
- [x] Stack N encoder blocks
- [x] Classification head = linear + softmax

**Backward pass and training (all in C):**
- [x] Gradient storage on every tensor
- [x] multiply backward = dA = dC x Bt, dB = At x dC
- [x] add, scale, ReLU, and leaky ReLU backward
- [x] softmax backward
- [x] layernorm backward
- [x] single-head attention backward = chain rule through all four steps
- [x] Cross-entropy loss
- [x] SGD optimizer
- [x] Adam optimizer
- [x] Training loop = forward, loss, backward, update

**Validation:**
- [x] Train the same architecture in PyTorch
- [x] Compare accuracy between C model and PyTorch model
- [x] The C model should get close. If it does not, something is wrong with the math.

**Data pipeline:**
- [ ] Syscall collection via dtrace on macOS
- [ ] Tokenizer = map syscall names to integer IDs
- [ ] Sequence dataset builder in C

**Live inference:**
- [ ] Load trained weights from binary file
- [ ] Hook into live syscall stream
- [ ] Score each process in real time
- [ ] Flag anomalies above threshold

**Optimization:**
- [ ] Cache-friendly matmul via loop reordering
- [ ] SIMD with ARM NEON intrinsics
- [x] Benchmark against PyTorch CPU inference

## Build & Run

```bash
# Build main test binary
make
./guerrilla

# Run automated numerical validation against PyTorch
make validate-pytorch

# Run CPU speed benchmark vs PyTorch
make bench

# Clean build artifacts
make clean
```

---

*"Guerrilla is built on understanding the machine. Infinite freedom begins where abstractions end."*