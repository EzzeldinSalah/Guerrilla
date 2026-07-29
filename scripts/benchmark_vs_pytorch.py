import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from statistics import fmean
from pathlib import Path

try:
    import torch
except Exception as exc:
    print("PyTorch is required for benchmark: .venv/bin/python3 -m pip install torch")
    sys.exit(2)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "speed_report.txt"


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)


def pytorch_benchmark(warmup=10, iterations=1000):
    torch.manual_seed(0)
    torch.set_num_threads(1)

    seq_len, d_model, heads, dk, d_ff, true_class, lr = 3, 4, 2, 2, 16, 1, 0.01

    W_Q = torch.randn(d_model, d_model, requires_grad=True)
    W_K = torch.randn(d_model, d_model, requires_grad=True)
    W_V = torch.randn(d_model, d_model, requires_grad=True)
    W_O = torch.randn(d_model, d_model, requires_grad=True)
    W1 = torch.randn(d_model, d_ff, requires_grad=True)
    W2 = torch.randn(d_ff, d_model, requires_grad=True)
    B1 = torch.randn(1, d_ff, requires_grad=True)
    B2 = torch.randn(1, d_model, requires_grad=True)
    classW = torch.randn(d_model, 2, requires_grad=True)
    classB = torch.randn(1, 2, requires_grad=True)
    params = [W_Q, W_K, W_V, W_O, W1, W2, B1, B2, classW, classB]

    x = torch.randn(seq_len, d_model)

    def step():
        Q, K, V = x @ W_Q, x @ W_K, x @ W_V
        head_outputs = []
        d_head = d_model // heads
        for h in range(heads):
            q = Q[:, h * d_head:(h + 1) * d_head]
            k = K[:, h * d_head:(h + 1) * d_head]
            v = V[:, h * d_head:(h + 1) * d_head]
            scores = (q @ k.t()) * (1.0 / math.sqrt(float(dk)))
            head_outputs.append(torch.softmax(scores, dim=1) @ v)
        concat = torch.cat(head_outputs, dim=1)
        att = concat @ W_O
        res1 = x + att
        norm1 = (res1 - res1.mean(dim=1, keepdim=True)) / torch.sqrt(((res1 - res1.mean(dim=1, keepdim=True))**2).mean(dim=1, keepdim=True) + 1e-5)
        h = torch.where(norm1 @ W1 + B1 > 0, norm1 @ W1 + B1, (norm1 @ W1 + B1) * 0.01)
        ffn = h @ W2 + B2
        res2 = norm1 + ffn
        norm2 = (res2 - res2.mean(dim=1, keepdim=True)) / torch.sqrt(((res2 - res2.mean(dim=1, keepdim=True))**2).mean(dim=1, keepdim=True) + 1e-5)
        pooled = norm2.mean(dim=0, keepdim=True)
        logits = pooled @ classW + classB
        probs = torch.softmax(logits, dim=1)
        loss = -torch.log(torch.clamp(probs[0, true_class], min=1e-7))
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
                p.grad.zero_()

    for _ in range(warmup):
        step()

    start = time.perf_counter()
    for _ in range(iterations):
        step()
    elapsed = time.perf_counter() - start

    return elapsed


def build_c_benchmark(tmp_path, warmup, iterations):
    bench_src = tmp_path / "bench_c.c"
    bench_bin = tmp_path / "bench_c"

    bench_src.write_text(f"""\
#include <stdio.h>
#include <time.h>
#include "tensor.h"
#include "attention.h"
#include "encoder.h"
#include "trainLoop.h"

int main() {{
    ModelConfig modelConfig = {{
        .seqLen = 3,
        .dModel = 4,
        .heads = 2,
        .layers = 1,
        .dk = 2
    }};

    Tensor *input = tensorCreate(modelConfig.seqLen, modelConfig.dModel);
    Transformer *transformer = transformerCreate(&modelConfig);

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
    cmd = [
        cc,
        "-Wall",
        "-O3",
        "-march=native",
        "-flto",
        "-Iinclude",
        "-Itraining",
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
    parser = argparse.ArgumentParser(description="Compare Guerrilla training speed with PyTorch.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before timing each trial.")
    parser.add_argument("--iterations", type=int, default=2000, help="Timed iterations per trial.")
    parser.add_argument("--trials", type=int, default=20, help="Number of independent trials to average.")
    args = parser.parse_args()

    c_times = []
    torch_times = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bench_bin = build_c_benchmark(tmp_path, args.warmup, args.iterations)

        for _ in range(args.trials):
            c_times.append(run_c_benchmark(bench_bin))
            torch_times.append(pytorch_benchmark(warmup=args.warmup, iterations=args.iterations))

    c_time = fmean(c_times)
    torch_time = fmean(torch_times)

    c_us_per_step = (c_time / args.iterations) * 1e6
    torch_us_per_step = (torch_time / args.iterations) * 1e6
    speedup = torch_time / c_time if c_time > 0 else 0

    lines = []
    lines.append(
        f"Training Step Speed Benchmark ({args.trials} trials, {args.iterations:,} iterations per trial, single-threaded)"
    )
    lines.append("")
    lines.append(f"C Guerrilla:  {c_time:.4f} sec avg  ({c_us_per_step:.2f} us/step)")
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
