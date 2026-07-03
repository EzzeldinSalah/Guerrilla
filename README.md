# Guerrilla

A host-based malware detector that learns normal syscall sequences on your machine and flags anything that deviates. A small transformer that trains and runs entirely in C, understands process behavior, and tells you when something looks wrong.

## What exists right now

A complete tensor library and the forward pass of a transformer encoder.

**Tensor library:**
- Matrix creation with flat contiguous memory layout
- Addition, multiplication, transpose, scale
- Softmax with log-sum-exp numerical stability fix
- Leaky ReLU and standard ReLU
- Layer normalization per row with epsilon

**Transformer components:**
- Scaled dot-product attention = Q x Kt / sqrt(dk), softmax, weighted sum over V
- Multi-head attention = slices Q, K, V into heads, runs attention on each independently
- ModelConfig struct driving all dimensions so nothing is hardcoded

## What is being built

```
Guerrilla/
├── include/
│   ├── tensor.h       <- tensor library interface
│   ├── attention.h    <- transformer components + ModelConfig
│   └── mainTest.h
├── src/
│   ├── tensor.c       <- all math primitives
│   ├── attention.c    <- attention, multi-head attention, slicing
│   └── main.c
├── tests/
│   └── mainTest.c
├── weights/           <- trained weights live here
├── data/              <- syscall sequence datasets live here
└── Makefile
```

## Roadmap

**Forward pass:**
- [x] Tensor library
- [x] Single-head attention
- [x] Multi-head attention
- [x] tensorConcat = stitch head outputs back together
- [ ] Feedforward block = two linear layers with leaky ReLU
- [ ] Full encoder block = layernorm + attention + residual + layernorm + feedforward + residual
- [ ] Stack N encoder blocks
- [ ] Classification head = linear + softmax

**Backward pass and training (all in C):**
- [ ] Gradient storage on every tensor
- [ ] multiply backward = dA = dC x Bt, dB = At x dC
- [ ] softmax backward
- [ ] layernorm backward
- [ ] attention backward = chain rule through all four steps
- [ ] Cross-entropy loss
- [ ] SGD optimizer
- [ ] Adam optimizer
- [ ] Training loop = forward, loss, backward, update

**Data pipeline:**
- [ ] Syscall collection via dtrace on macOS
- [ ] Tokenizer = map syscall names to integer IDs
- [ ] Sequence dataset builder in C

**Validation:**
- [ ] Train the same architecture in PyTorch
- [ ] Compare accuracy between C model and PyTorch model
- [ ] The C model should get close. If it does not, something is wrong with the math.

**Live inference:**
- [ ] Load trained weights from binary file
- [ ] Hook into live syscall stream
- [ ] Score each process in real time
- [ ] Flag anomalies above threshold

**Optimization:**
- [ ] Cache-friendly matmul via loop reordering
- [ ] SIMD with ARM NEON intrinsics
- [ ] Benchmark against PyTorch CPU inference

## Build

```bash
make
./guerrilla
```

```bash
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

*"No safety rails. No dependencies. Guerrilla treats memory as a flat grid of execution space where efficiency is carved by hand."*