CC = gcc
CFLAGS = -w -Wall -g -Iinclude -Itests/include
LIBS = -lm

SRC_FILES = src/main.c \
            src/tensor.c \
            src/attention.c \
            src/encoder.c

TEST_FILES = tests/src/mainTest.c \
             tests/src/tensorTest.c \
             tests/src/attentionTest.c \
             tests/src/encoderTest.c \
             tests/src/testUtils.c

guerrilla: $(SRC_FILES) $(TEST_FILES)
	$(CC) $(CFLAGS) $(SRC_FILES) $(TEST_FILES) -o guerrilla $(LIBS)

clean:
	rm -f guerrilla
	rm -rf *.dSYM