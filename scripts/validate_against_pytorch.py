import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import torch
except Exception as exc:
    print("PyTorch is required for validation: python3 -m pip install torch")
    print(f"import error: {exc}")
    sys.exit(2)


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks" / "validationReport.txt"


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True)


def parse_c_results(output):
    losses = []
    for line in output.splitlines():
        if not line.startswith("STEP "):
            continue
        parts = line.split()
        losses.append(float(parts[3]))
    return losses


def fill_pattern(shape, seed):
    total = 1
    for dim in shape:
        total *= dim

    data = []
    for i in range(total):
        value = (((i + 1) * seed) + seed * seed + (i % 3)) % 23
        data.append((float(value) - 11.0) * 0.02)

    return torch.tensor(data, dtype=torch.float32).reshape(shape)


def pytorch_results(seq_len=1024, d_model=64, heads=8, layers=6, num_steps=10, lr=0.01):
    torch.manual_seed(0)
    torch.set_num_threads(1)

    dk = d_model // heads
    d_ff = d_model * 4
    true_class = 1

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

    optimizer = torch.optim.Adam(all_params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    x = fill_pattern((seq_len, d_model), 3)

    losses = []
    d_head = d_model // heads

    for _ in range(num_steps):
        curr_x = x
        for lp in layer_params:
            Q = curr_x @ lp["W_Q"]
            K = curr_x @ lp["W_K"]
            V = curr_x @ lp["W_V"]

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

            h_ffn = torch.where(
                norm1 @ lp["W1"] + lp["B1"] > 0,
                norm1 @ lp["W1"] + lp["B1"],
                (norm1 @ lp["W1"] + lp["B1"]) * 0.01,
            )
            ffn = h_ffn @ lp["W2"] + lp["B2"]
            res2 = norm1 + ffn
            mean2 = res2.mean(dim=1, keepdim=True)
            var2 = ((res2 - mean2) ** 2).mean(dim=1, keepdim=True)
            curr_x = (res2 - mean2) / torch.sqrt(var2 + 1e-5)

        pooled = curr_x.mean(dim=0, keepdim=True)
        logits = pooled @ classW + classB
        probs = torch.softmax(logits, dim=1)
        loss = -torch.log(torch.clamp(probs[0, true_class], min=1e-7))

        losses.append(float(loss.detach().item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


def main():
    seq_len = 1024
    d_model = 64
    heads = 8
    dk = d_model // heads
    layers = 6
    num_steps = 10
    lr = 0.01
    tolerance = 1e-4

    print(f"Validating Guerrilla Transformer vs PyTorch ({num_steps} Adam steps)")
    print(f"Config: d_model={d_model}, seq_len={seq_len}, layers={layers}, heads={heads}, lr={lr}")
    print("Compiling and running C training loop...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        validator_src = tmp_path / "validate_training.c"
        validator_bin = tmp_path / "validate_training"

        validator_src.write_text(f"""\
#include <stdio.h>
#include "tensor.h"
#include "attention.h"
#include "encoder.h"
#include "optimizer.h"
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

    AdamOptimizer *optimizer = adamCreate(transformer, &modelConfig, {lr}f);

    for (int step = 0; step < {num_steps}; step++) {{
        float loss = trainAdam(transformer, input, 1, &modelConfig, optimizer);
        printf("STEP %d LOSS %.9f\\n", step, loss);
    }}

    adamFree(optimizer, &modelConfig);
    tensorFree(input);
    transformerFree(transformer, &modelConfig);

    return 0;
}}
""")

        sources = [
            str(validator_src),
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
            "-ffast-math",
            "-Iinclude",
            "-Itraining",
            *sources,
            "-o",
            str(validator_bin),
            "-lm",
        ]
        run(cmd)

        c_output = run([str(validator_bin)]).stdout

    c_losses = parse_c_results(c_output)

    print("Running PyTorch reference training loop...")
    torch_losses = pytorch_results(
        seq_len=seq_len,
        d_model=d_model,
        heads=heads,
        layers=layers,
        num_steps=num_steps,
        lr=lr,
    )

    report_lines = []
    report_lines.append(f"C Model vs PyTorch ({num_steps} Adam Steps: d_model={d_model}, seq_len={seq_len}, layers={layers}, heads={heads})")
    report_lines.append("")
    report_lines.append(f"{'Step':<8}  {'C Loss':<14}  {'PyTorch Loss':<14}  {'Loss Diff':<12}  {'Status'}")
    report_lines.append("-" * 62)

    max_drift = 0.0
    worst_step = 0
    failed = False

    log_checkpoints = set([0, 9, 24, 49, 99, 149, num_steps - 1] + [i for i in range(num_steps) if (i + 1) % 10 == 0])

    for step in range(num_steps):
        c_l = c_losses[step]
        t_l = torch_losses[step]
        diff = abs(c_l - t_l)

        if diff > max_drift:
            max_drift = diff
            worst_step = step

        if diff > tolerance:
            failed = True

        status = "ok" if diff <= tolerance else "DIVERGED"

        if step in log_checkpoints or status == "DIVERGED":
            report_lines.append(f"{step:<8}  {c_l:<14.9f}  {t_l:<14.9f}  {diff:<12.2e}  {status}")

    report_lines.append("")
    report_lines.append(f"tolerance: {tolerance:.2e}")
    report_lines.append(f"worst drift: step {worst_step} at {max_drift:.2e}")

    if failed:
        report_lines.append("result: FAIL (DIVERGED)")
    else:
        report_lines.append("result: pass")

    report_text = "\n".join(report_lines) + "\n"
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(report_text)

    print("\n" + report_text)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
