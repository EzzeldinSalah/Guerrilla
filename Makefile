CC = gcc
CFLAGS = -w -Wall -g -Iinclude
LIBS = -lm

tensor: src/main.c src/tensor.c src/attention.c src/encoder.c tests/mainTest.c
	$(CC) $(CFLAGS) src/main.c src/tensor.c src/attention.c src/encoder.c tests/mainTest.c -o guerrilla $(LIBS)

clean:
	rm -f guerrilla
	rm -rf *.dSYM