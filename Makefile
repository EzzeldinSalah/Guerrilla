CC = gcc
SRC_DIR = src
TRAINING_DIR = training
TEST_DIR = tests

CFLAGS = -Wall -O3 -march=native -Iinclude -I$(TRAINING_DIR) -I$(TEST_DIR)
LIBS = -lm

SRCS = $(wildcard $(SRC_DIR)/*.c) \
       $(wildcard $(TRAINING_DIR)/*.c) \
       $(wildcard $(TEST_DIR)/*.c)

TARGET = guerrilla

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

validate-pytorch:
	@if [ -f .venv/bin/python3 ]; then .venv/bin/python3 scripts/validate_against_pytorch.py; else python3 scripts/validate_against_pytorch.py; fi

bench:
	@if [ -f .venv/bin/python3 ]; then .venv/bin/python3 scripts/benchmark_vs_pytorch.py; else python3 scripts/benchmark_vs_pytorch.py; fi

clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)
	rm -rf *.dSYM *.o

.PHONY: all validate-pytorch bench clean
