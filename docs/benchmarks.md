# Benchmarks

This project includes a PyTorch comparison script and stores benchmark output under `benchmarks/`.

## Setup

Create a virtual environment and install the Python dependencies first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
make bench
```

The benchmark runs single-threaded at the current target configuration: `d_model=64`, `seq_len=1024`, `layers=6`, and `heads=8`. It warms up before timing and averages multiple trials while comparing the C training step against a PyTorch implementation that performs the same sequence of operations.

## Important Caveat

This is a training-step performance comparison, not an end-to-end application benchmark. The C and PyTorch sides use deterministic, matching initialization patterns, and the reported timing is specific to the configured CPU, compiler, flags, and model dimensions.

The benchmark is useful for tracking regressions and improvements in this implementation. It is not a general claim about C versus PyTorch performance on other hardware or workloads.

## Validation Targets

```bash
make validate              # Gate A: one SGD step, tolerance 2e-8
make validate-scalar       # Gate A with scalar kernels
make validate-drift        # Gate B: 200 Adam steps, tolerance 1e-4
make validate-drift-scalar # Gate B with scalar kernels
make bench                 # Native benchmark
make bench-all             # Native and scalar benchmark comparison
```

Gate A compares selected C and PyTorch parameter results after one SGD step. Gate B compares the 200-step Adam loss trajectory at real scale. Both gates must report `result: pass`; the scalar variants exercise the non-NEON fallback path.

## Outputs

- `benchmarks/speedReport.txt` stores the latest speed comparison.
- `benchmarks/validationReport.txt` stores the latest validation output.
- `benchmarks/adam_drift_report.txt` stores the latest Adam drift validation output.