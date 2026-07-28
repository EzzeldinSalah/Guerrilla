# Guerrilla

A host-based malware detector that learns normal syscall sequences on your machine and flags anything that deviates. A small transformer that trains and runs entirely in C, understands process behavior, and tells you when something looks wrong.

## What exists right now

A complete tensor library, full forward and backward passes of a transformer encoder, SGD and Adam optimizers, and automated PyTorch validation.

**Tensor library:**
- Matrix creation with flat contiguous memory layout
- Addition, multiplication, transpose, scale
- Softmax with log-sum-exp numerical stability fix
- Leaky ReLU and standard ReLU
- Layer normalization per row with epsilon

**Transformer and training components:**
- Scaled dot-product attention = Q x Kt / sqrt(dk), softmax, weighted sum over V
- Multi-head attention = slices Q, K, V into heads, runs attention on each independently
- ModelConfig struct driving all dimensions so nothing is hardcoded
- Tensor backward pass chain rule derivations and gradient storage on all parameter matrices
- Optimizers: SGD and Adam with moment tracking ($m, v$) and bias correction
- End-to-end training loop (`trainSgd`, `trainAdam`) driving loss down to 0.00001
- Verified mathematical equivalence vs PyTorch CPU (26x faster execution)

## What is being built

```
Guerrilla/
├── include/
│   ├── tensor.h
│   ├── attention.h
│   └── encoder.h
├── src/
│   ├── tensor.c
│   ├── attention.c
│   ├── encoder.c
│   └── main.c
├── training/
│   ├── tensorGrad.c / tensorGrad.h
│   ├── attentionGrad.c / attentionGrad.h
│   ├── encoderGrad.c / encoderGrad.h
│   ├── lossFunctions.c / lossFunctions.h
│   ├── optimizer.c / optimizer.h
│   └── trainLoop.c / trainLoop.h
├── tests/
│   ├── tests.h
│   ├── testUtils.c
│   ├── tensorTest.c
│   ├── attentionTest.c
│   ├── encoderTest.c
│   └── backwardTest.c
├── scripts/
│   ├── validate_against_pytorch.py
│   └── benchmark_vs_pytorch.py
├── docs/
│   └── backward-derivations.md
├── benchmarks/
│   ├── matmulBench.c
│   ├── validation_report.txt
│   └── speed_report.txt
├── data/
├── weights/
├── Makefile
└── README.md
```

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

## Build & Test

```bash
make
./guerrilla
```

```bash
make validate-pytorch   # Verify numerical parity against PyTorch
make bench              # Speed comparison vs PyTorch CPU
make clean
```

No dependencies. That is the point.


## Contributing

Welcome! Do you want to learn and build a transformer in pure C? You are in the right place. 
We do not use big frameworks, libraries, or automatic tools. Because of this, our code rules are very strict.

### The AI and LLM Rule

We have a strict rule about using AI (like ChatGPT, Claude, or Copilot).

- **BANNED:** Do not use AI (especially agentic or vibe coding) to write the C code. The goal of this project is to learn the math and memory layout yourself. If we see AI-written code, we will close your Pull Request immediately.
- **ALLOWED:** You can use AI to research math formulas, fix typos, write text documentation, or ask for better code practices (with a deep manual review).

You must also free all the memory you create. Check the roadmap above and open a PR !


## Why ?

Because writing `import torch` felt like cheating.

And because the people who built the tools everyone else uses had to understand what actually happens when you multiply two matrices, how the memory sits, why the loop order matters, what softmax is doing numerically when the values get large. That understanding does not come from calling library functions.

---

*"Guerrilla is built on understanding the machine. Infinite freedom begins where abstractions end."*