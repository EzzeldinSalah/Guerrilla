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
    results = {}

    for line in output.splitlines():
        if not line.startswith("RESULT "):
            continue

        _, name, value = line.split()
        results[name] = float(value)

    return results


def fill_pattern(shape, seed):
    total = 1
    for dim in shape:
        total *= dim

    data = []
    for i in range(total):
        value = (((i + 1) * seed) + seed * seed + (i % 3)) % 23
        data.append((float(value) - 11.0) * 0.02)

    return torch.tensor(data, dtype=torch.float32).reshape(shape)


def layer_norm_rows(x):
    mean = x.mean(dim=1, keepdim=True)
    variance = ((x - mean) * (x - mean)).mean(dim=1, keepdim=True)
    return (x - mean) / torch.sqrt(variance + 1e-5)


def pytorch_results():
    torch.manual_seed(0)

    seq_len = 3
    d_model = 4
    heads = 2
    dk = 2
    d_ff = d_model * 4
    true_class = 1
    lr = 0.01

    params = {
        "W_Q": fill_pattern((d_model, d_model), 11).requires_grad_(),
        "W_K": fill_pattern((d_model, d_model), 12).requires_grad_(),
        "W_V": fill_pattern((d_model, d_model), 13).requires_grad_(),
        "W_O": fill_pattern((d_model, d_model), 14).requires_grad_(),
        "W1": fill_pattern((d_model, d_ff), 15).requires_grad_(),
        "W2": fill_pattern((d_ff, d_model), 16).requires_grad_(),
        "B1": fill_pattern((1, d_ff), 17).requires_grad_(),
        "B2": fill_pattern((1, d_model), 18).requires_grad_(),
        "classW": fill_pattern((d_model, 2), 101).requires_grad_(),
        "classB": fill_pattern((1, 2), 102).requires_grad_(),
    }

    x = fill_pattern((seq_len, d_model), 3)

    Q = x @ params["W_Q"]
    K = x @ params["W_K"]
    V = x @ params["W_V"]

    head_outputs = []
    d_head = d_model // heads
    for h in range(heads):
        col_start = h * d_head
        col_end = col_start + d_head

        q = Q[:, col_start:col_end]
        k = K[:, col_start:col_end]
        v = V[:, col_start:col_end]

        scores = (q @ k.t()) * (1.0 / math.sqrt(float(dk)))
        softed = torch.softmax(scores, dim=1)
        head_outputs.append(softed @ v)

    concatenated = torch.cat(head_outputs, dim=1)
    attention_out = concatenated @ params["W_O"]
    first_residual = x + attention_out
    hidden_x = layer_norm_rows(first_residual)

    theta = hidden_x @ params["W1"]
    hidden = theta + params["B1"]
    leaky_hidden = torch.where(hidden > 0, hidden, hidden * 0.01)
    beta = leaky_hidden @ params["W2"]
    ffn_out = beta + params["B2"]
    second_residual = hidden_x + ffn_out
    encoded = layer_norm_rows(second_residual)

    pooled = encoded.mean(dim=0, keepdim=True)
    logits = pooled @ params["classW"] + params["classB"]
    probs = torch.softmax(logits, dim=1)
    loss = -torch.log(torch.clamp(probs[0, true_class], min=1e-7))
    loss.backward()

    with torch.no_grad():
        for param in params.values():
            param -= lr * param.grad

    results = {}
    with torch.no_grad():
        results["loss"] = float(loss)
        results["classW0"] = float(params["classW"][0, 0])
        results["classW3"] = float(params["classW"].reshape(-1)[3])
        results["classB0"] = float(params["classB"][0, 0])
        results["classB1"] = float(params["classB"][0, 1])
        results["WQ0"] = float(params["W_Q"].reshape(-1)[0])
        results["WK1"] = float(params["W_K"].reshape(-1)[1])
        results["WV2"] = float(params["W_V"].reshape(-1)[2])
        results["WO3"] = float(params["W_O"].reshape(-1)[3])
        results["W10"] = float(params["W1"].reshape(-1)[0])
        results["W20"] = float(params["W2"].reshape(-1)[0])
        results["B10"] = float(params["B1"].reshape(-1)[0])
        results["B20"] = float(params["B2"].reshape(-1)[0])

    return results


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        validator_src = tmp_path / "validate_training.c"
        validator_bin = tmp_path / "validate_training"

        validator_src.write_text("""\
#include <stdio.h>
#include "tensor.h"
#include "attention.h"
#include "encoder.h"
#include "trainLoop.h"

static void fillPattern (Tensor *tensor, int seed) {
    int totalSize = tensor->rows * tensor->cols;

    for (int i = 0; i < totalSize; i++) {
        int value = (((i + 1) * seed) + seed * seed + (i % 3)) % 23;
        tensor->data[i] = ((float)value - 11.0f) * 0.02f;
    }
}

static void fillTransformerPattern (Transformer *transformer, ModelConfig *modelConfig) {
    for (int i = 0; i < modelConfig->layers; i++) {
        int seed = 10 + i * 20;

        fillPattern(transformer->layers[i].W_Q, seed + 1);
        fillPattern(transformer->layers[i].W_K, seed + 2);
        fillPattern(transformer->layers[i].W_V, seed + 3);
        fillPattern(transformer->layers[i].W_O, seed + 4);
        fillPattern(transformer->layers[i].W1, seed + 5);
        fillPattern(transformer->layers[i].W2, seed + 6);
        fillPattern(transformer->layers[i].B1, seed + 7);
        fillPattern(transformer->layers[i].B2, seed + 8);
    }

    fillPattern(transformer->classW, 101);
    fillPattern(transformer->classB, 102);
}

int main() {
    ModelConfig modelConfig = {
        .seqLen = 3,
        .dModel = 4,
        .heads = 2,
        .layers = 1,
        .dk = 2
    };

    Tensor *input = tensorCreate(modelConfig.seqLen, modelConfig.dModel);
    Transformer *transformer = transformerCreate(&modelConfig);

    fillPattern(input, 3);
    fillTransformerPattern(transformer, &modelConfig);

    float loss = trainSgd(transformer, input, 1, &modelConfig, 0.01f);

    printf("RESULT loss %.9f\\n", loss);
    printf("RESULT classW0 %.9f\\n", transformer->classW->data[0]);
    printf("RESULT classW3 %.9f\\n", transformer->classW->data[3]);
    printf("RESULT classB0 %.9f\\n", transformer->classB->data[0]);
    printf("RESULT classB1 %.9f\\n", transformer->classB->data[1]);
    printf("RESULT WQ0 %.9f\\n", transformer->layers[0].W_Q->data[0]);
    printf("RESULT WK1 %.9f\\n", transformer->layers[0].W_K->data[1]);
    printf("RESULT WV2 %.9f\\n", transformer->layers[0].W_V->data[2]);
    printf("RESULT WO3 %.9f\\n", transformer->layers[0].W_O->data[3]);
    printf("RESULT W10 %.9f\\n", transformer->layers[0].W1->data[0]);
    printf("RESULT W20 %.9f\\n", transformer->layers[0].W2->data[0]);
    printf("RESULT B10 %.9f\\n", transformer->layers[0].B1->data[0]);
    printf("RESULT B20 %.9f\\n", transformer->layers[0].B2->data[0]);

    tensorFree(input);
    transformerFree(transformer, &modelConfig);

    return 0;
}
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
            "-Iinclude",
            "-Itraining",
            *sources,
            "-o",
            str(validator_bin),
            "-lm",
        ]
        run(cmd)

        c_output = run([str(validator_bin)]).stdout

    c_results = parse_c_results(c_output)
    torch_results = pytorch_results()

    tolerance = 2e-6
    max_abs = 0.0
    worst_name = ""
    failed = False
    report_lines = []

    report_lines.append("C model vs PyTorch on identical weights, one SGD step, lr=0.01")
    report_lines.append("")

    for name, c_value in c_results.items():
        torch_value = torch_results[name]
        diff = abs(c_value - torch_value)

        if diff > max_abs:
            max_abs = diff
            worst_name = name

        if diff > tolerance:
            failed = True

        status = "ok" if diff <= tolerance else "MISMATCH"
        report_lines.append(f"{name:<12}  C={c_value:.9f}  PyTorch={torch_value:.9f}  diff={diff:.2e}  {status}")

    report_lines.append("")
    report_lines.append(f"tolerance: {tolerance}")
    report_lines.append(f"worst drift: {worst_name} at {max_abs:.2e}")

    if failed:
        report_lines.append("result: FAIL")
    else:
        report_lines.append("result: pass")

    report_text = "\n".join(report_lines) + "\n"
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(report_text)

    print(report_text)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
