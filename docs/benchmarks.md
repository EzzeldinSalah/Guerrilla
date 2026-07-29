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

The benchmark script runs single-threaded, warms up before timing, and averages multiple trials. It compares the C training step against a PyTorch implementation that performs the same sequence of operations.

## Important Caveat

The current benchmark is a microbenchmark. The default tensor sizes are intentionally tiny, so the result mostly measures interpreter and autograd overhead versus direct C execution.

That makes it useful for checking the implementation and the overhead profile, but it is not a general proof that the C code will beat PyTorch at larger shapes (till now!).

## Notes

If you want to compare larger tensor shapes, adjust the benchmark script first and then rerun `make bench` so the reported numbers match the same dimensions on both sides.

## Outputs

- `benchmarks/speed_report.txt` stores the latest speed comparison.
- `benchmarks/validation_report.txt` stores the latest validation output.