CC = gcc
SRC_DIR = src
TRAINING_DIR = training
TEST_DIR = tests

CFLAGS = $(shell cat .buildflags) -Iinclude -I$(TRAINING_DIR) -I$(TEST_DIR)
LIBS = -lm

SRCS = $(wildcard $(SRC_DIR)/*.c) \
       $(wildcard $(TRAINING_DIR)/*.c) \
       $(wildcard $(TEST_DIR)/*.c)

TARGET = guerrilla

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

validate:
	@if [ -f .venv/bin/python3 ]; then .venv/bin/python3 scripts/validate_against_pytorch.py; elif command -v python >/dev/null 2>&1; then python scripts/validate_against_pytorch.py; else python3 scripts/validate_against_pytorch.py; fi

validate-scalar:
	@if [ -f .venv/bin/python3 ]; then .venv/bin/python3 scripts/validate_against_pytorch.py --scalar; elif command -v python >/dev/null 2>&1; then python scripts/validate_against_pytorch.py --scalar; else python3 scripts/validate_against_pytorch.py --scalar; fi

validate-drift:
	@if [ -f .venv/bin/python3 ]; then .venv/bin/python3 scripts/validate_adam_drift.py; elif command -v python >/dev/null 2>&1; then python scripts/validate_adam_drift.py; else python3 scripts/validate_adam_drift.py; fi

validate-drift-scalar:
	@if [ -f .venv/bin/python3 ]; then .venv/bin/python3 scripts/validate_adam_drift.py --scalar; elif command -v python >/dev/null 2>&1; then python scripts/validate_adam_drift.py --scalar; else python3 scripts/validate_adam_drift.py --scalar; fi

bench:
	@if [ -f .venv/bin/python3 ]; then .venv/bin/python3 scripts/benchmark_vs_pytorch.py; elif command -v python >/dev/null 2>&1; then python scripts/benchmark_vs_pytorch.py; else python3 scripts/benchmark_vs_pytorch.py; fi

bench-all:
	@if [ -f .venv/bin/python3 ]; then .venv/bin/python3 scripts/benchmark_vs_pytorch.py --all; elif command -v python >/dev/null 2>&1; then python scripts/benchmark_vs_pytorch.py --all; else python3 scripts/benchmark_vs_pytorch.py --all; fi

clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)
	rm -rf *.dSYM *.o

.PHONY: all validate validate-scalar validate-drift validate-drift-scalar bench bench-all clean