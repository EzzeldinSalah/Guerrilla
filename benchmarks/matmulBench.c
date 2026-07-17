#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../include/tensor.h"

#define N 512

int main() {
    srand(42);

    Tensor *A = tensorCreate(N, N), *B = tensorCreate(N, N);

    if (!A || !B) {
        printf("Failed to allocate benchmarking tensors.\n");
        if (A) tensorFree(A);
        if (B) tensorFree(B);
        return 1;
    }

    for (int i = 0; i < N * N; i++) {
        A->data[i] = (float)rand() / (float)RAND_MAX;
        B->data[i] = (float)rand() / (float)RAND_MAX;
    }

    printf("Running matrix multiplication benchmark (%dx%d)...\n", N, N);

    clock_t start = clock();
    Tensor *C = multiply(A, B); 
    double t_duration = (double)(clock() - start) / CLOCKS_PER_SEC;

    if (!C) {
        printf("Multiplication failed or returned NULL.\n");
        tensorFree(A), tensorFree(B);
        return 1;
    }

    printf("Current multiply() implementation time: %f seconds\n", t_duration);
    tensorFree(A), tensorFree(B), tensorFree(C);

    return 0;
}