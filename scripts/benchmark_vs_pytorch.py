import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import fmean

try:
    import torch
except Exception as exc:
    print("PyTorch is required for benchmark: python3 -m pip install torch")
    sys.exit(2)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "speedReport.txt"
BUILD_FLAGS = (ROOT / ".buildflags").read_text().split()


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)


def fill_pattern(shape, seed):
    total = 1
    for dim in shape:
        total *= dim

    data = []
    for i in range(total):
        value = (((i + 1) * seed) + seed * seed + (i % 3)) % 23
        data.append((float(value) - 11.0) * 0.02)

    return torch.tensor(data, dtype=torch.float32).reshape(shape)


def pytorch_benchmark(d_model, seq_len, layers, heads, dk, d_ff, warmup=2, iterations=10):
    torch.set_num_threads(1)

    true_class = 1
    lr = 0.01

    layer_params = []
    for i in range(layers):
        seed = 10 + i * 20
        p = {
            "W_Q": fill_pattern((d_model, d_model), seed + 1).requires_grad_(),
            "W_K": fill_pattern((d_model, d_model), seed + 2).requires_grad_(),
            "W_V": fill_pattern((d_model, d_model), seed + 3).requires_grad_(),
            "W_O": fill_pattern((d_model, d_model), seed + 4).requires_grad_(),
            "W1": fill_pattern((d_model, d_ff), seed + 5).requires_grad_(),
            "W2": fill_pattern((d_ff, d_model), seed + 6).requires_grad_(),
            "B1": fill_pattern((1, d_ff), seed + 7).requires_grad_(),
            "B2": fill_pattern((1, d_model), seed + 8).requires_grad_(),
        }
        layer_params.append(p)

    classW = fill_pattern((d_model, 2), 101).requires_grad_()
    classB = fill_pattern((1, 2), 102).requires_grad_()

    all_params = [classW, classB]
    for lp in layer_params:
        all_params.extend(lp.values())

    x = fill_pattern((seq_len, d_model), 3)

    def step():
        curr_x = x
        d_head = d_model // heads

        for lp in layer_params:
            Q, K, V = curr_x @ lp["W_Q"], curr_x @ lp["W_K"], curr_x @ lp["W_V"]
            head_outputs = []
            for h in range(heads):
                q = Q[:, h * d_head : (h + 1) * d_head]
                k = K[:, h * d_head : (h + 1) * d_head]
                v = V[:, h * d_head : (h + 1) * d_head]
                scores = (q @ k.t()) * (1.0 / math.sqrt(float(dk)))
                head_outputs.append(torch.softmax(scores, dim=1) @ v)

            concat = torch.cat(head_outputs, dim=1)
            att = concat @ lp["W_O"]
            res1 = curr_x + att
            mean1 = res1.mean(dim=1, keepdim=True)
            var1 = ((res1 - mean1) ** 2).mean(dim=1, keepdim=True)
            norm1 = (res1 - mean1) / torch.sqrt(var1 + 1e-5)

            pre = norm1 @ lp["W1"] + lp["B1"]
            h_ffn = torch.where(pre > 0, pre, pre * 0.01)
            ffn = h_ffn @ lp["W2"] + lp["B2"]
            res2 = norm1 + ffn
            mean2 = res2.mean(dim=1, keepdim=True)
            var2 = ((res2 - mean2) ** 2).mean(dim=1, keepdim=True)
            curr_x = (res2 - mean2) / torch.sqrt(var2 + 1e-5)

        pooled = curr_x.mean(dim=0, keepdim=True)
        logits = pooled @ classW + classB
        probs = torch.softmax(logits, dim=1)
        loss = -torch.log(torch.clamp(probs[0, true_class], min=1e-7))
        loss.backward()

        with torch.no_grad():
            for p in all_params:
                p -= lr * p.grad
                p.grad.zero_()

    for _ in range(warmup):
        step()

    start = time.perf_counter()
    for _ in range(iterations):
        step()
    elapsed = time.perf_counter() - start

    return elapsed


def build_c_benchmark(tmp_path, d_model, seq_len, layers, heads, dk, d_ff, warmup, iterations, extra_flags=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bench_src = tmp_path / "bench_c.c"
    bench_bin = tmp_path / "bench_c"

    bench_src.write_text(f"""\
#include <stdio.h>
#include <time.h>
#include "tensor.h"
#include "attention.h"
#include "encoder.h"
#include "trainLoop.h"

static void fillPattern (Tensor *tensor, int seed) {{
    int totalSize = tensor->rows * tensor->cols;
    for (int i = 0; i < totalSize; i++) {{
        int value = (((i + 1) * seed) + seed * seed + (i % 3)) % 23;
        tensor->data[i] = ((float)value - 11.0f) * 0.02f;
    }}
}}

static void fillTransformerPattern (Transformer *transformer, ModelConfig *modelConfig) {{
    for (int i = 0; i < modelConfig->layers; i++) {{
        int seed = 10 + i * 20;
        fillPattern(transformer->layers[i].W_Q, seed + 1);
        fillPattern(transformer->layers[i].W_K, seed + 2);
        fillPattern(transformer->layers[i].W_V, seed + 3);
        fillPattern(transformer->layers[i].W_O, seed + 4);
        fillPattern(transformer->layers[i].W1, seed + 5);
        fillPattern(transformer->layers[i].W2, seed + 6);
        fillPattern(transformer->layers[i].B1, seed + 7);
        fillPattern(transformer->layers[i].B2, seed + 8);
    }}
    fillPattern(transformer->classW, 101);
    fillPattern(transformer->classB, 102);
}}

int main() {{
    ModelConfig modelConfig = {{
        .seqLen = {seq_len},
        .dModel = {d_model},
        .heads = {heads},
        .layers = {layers},
        .dk = {dk}
    }};

    Tensor *input = tensorCreate(modelConfig.seqLen, modelConfig.dModel);
    Transformer *transformer = transformerCreate(&modelConfig);
    fillPattern(input, 3);
    fillTransformerPattern(transformer, &modelConfig);

    for (int i = 0; i < {warmup}; i++) {{
        trainSgd(transformer, input, 1, &modelConfig, 0.01f);
    }}

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < {iterations}; i++) {{
        trainSgd(transformer, input, 1, &modelConfig, 0.01f);
    }}

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("C_TIME %.6f\\n", elapsed);

    tensorFree(input);
    transformerFree(transformer, &modelConfig);
    return 0;
}}
""")

    sources = [
        str(bench_src),
        "src/tensor.c",
        "src/attention.c",
        "src/encoder.c",
        "training/attentionGrad.c",
        "training/encoderGrad.c",
        "training/lossFunctions.c",
        "training/optimizer.c",
        "training/tensorGrad.c",
        "training/trainLoop.c",
    ]

    cc = os.environ.get("CC", "gcc")
    extra = extra_flags or []
    cmd = [
        cc,
        *BUILD_FLAGS,
        "-Iinclude",
        "-Itraining",
        *extra,
        *sources,
        "-o",
        str(bench_bin),
        "-lm",
    ]
    run(cmd)

    return bench_bin


def run_c_benchmark(bench_bin):
    out = run([str(bench_bin)]).stdout
    return float(out.split()[1])


def main():
    run_scalar = "--scalar" in sys.argv or "--all" in sys.argv

    d_model = 64
    seq_len = 1024
    layers = 6
    heads = 8
    dk = d_model // heads
    d_ff = d_model * 4
    warmup = 2
    iterations = 10
    trials = 3

    print(f"Benchmarking Guerrilla Transformer vs PyTorch")
    print(f"Config: d_model={d_model}, seq_len={seq_len}, layers={layers}, heads={heads}")
    print("Compiling and running C training loop ...")

    c_times = []
    scalar_times = []
    torch_times = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bench_bin = build_c_benchmark(tmp_path, d_model, seq_len, layers, heads, dk, d_ff, warmup, iterations)

        scalar_bin = None
        if run_scalar:
            scalar_bin = build_c_benchmark(tmp_path / "scalar", d_model, seq_len, layers, heads, dk, d_ff, warmup, iterations, extra_flags=["-DFORCE_SCALAR"])

        for _ in range(trials):
            c_times.append(run_c_benchmark(bench_bin))
            if scalar_bin:
                scalar_times.append(run_c_benchmark(scalar_bin))
            torch_times.append(
                pytorch_benchmark(
                    d_model=d_model,
                    seq_len=seq_len,
                    layers=layers,
                    heads=heads,
                    dk=dk,
                    d_ff=d_ff,
                    warmup=warmup,
                    iterations=iterations,
                )
            )

    c_time = fmean(c_times)
    torch_time = fmean(torch_times)

    c_us_per_step = (c_time / iterations) * 1e6
    torch_us_per_step = (torch_time / iterations) * 1e6
    speedup = torch_time / c_time if c_time > 0 else 0

    lines = []
    lines.append(
        f"Training Step Speed Benchmark ({trials} trials, {iterations} iterations per trial, single-threaded)"
    )
    lines.append(f"Config: d_model={d_model}, seq_len={seq_len}, layers={layers}, heads={heads}")
    lines.append("")
    lines.append(f"C Guerrilla:  {c_time:.4f} sec avg  ({c_us_per_step:.2f} us/step)")
    if scalar_times:
        s_time = fmean(scalar_times)
        s_us_per_step = (s_time / iterations) * 1e6
        neon_speedup = s_time / c_time if c_time > 0 else 0
        lines.append(f"C Scalar:     {s_time:.4f} sec avg  ({s_us_per_step:.2f} us/step)  [{neon_speedup:.2f}x NEON speedup]")
    lines.append(f"PyTorch CPU:  {torch_time:.4f} sec avg  ({torch_us_per_step:.2f} us/step)")
    lines.append("")
    if speedup >= 1.0:
        lines.append(f"Winner: C Guerrilla ({speedup:.2f}x faster than PyTorch CPU)")
    else:
        lines.append(f"Winner: PyTorch CPU ({1.0 / speedup:.2f}x faster than C Guerrilla)")

    report_text = "\n".join(lines) + "\n"
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(report_text)

    print(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())