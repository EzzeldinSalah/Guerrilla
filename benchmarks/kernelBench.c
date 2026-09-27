#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/tensor.h"

/* tensorGrad.h depends on the C++ standard header <cstddef>, but this
 * benchmark is compiled as C. Declare the benchmarked gradient kernels here
 * instead of pulling the C++-only header into the C translation unit. */
Tensor *multiplyBackwardA(Tensor *a, Tensor *b, Tensor *dC);
Tensor *multiplyBackwardB(Tensor *a, Tensor *b, Tensor *dC);
Tensor *layerNormalization(Tensor *x);
Tensor *layerNormBackward(Tensor *x, Tensor *dC);
Tensor *leakyRelu(Tensor *x, float alpha);
Tensor *leakyReluBackward(Tensor *x, Tensor *dC, float alpha);

static double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    const int seqLen = 1024;
    const int dModel = 64;
    const int warmup = 10;
    const int iters = 100;

    Tensor *X = tensorCreate(seqLen, dModel);
    Tensor *W = tensorCreate(dModel, dModel);
    Tensor *dC = tensorCreate(seqLen, dModel);

    for (int i = 0; i < seqLen * dModel; i++) {
        X->data[i] = ((float)(i % 17) - 8.0f) * 0.05f;
        dC->data[i] = ((float)(i % 19) - 9.0f) * 0.05f;
    }
    for (int i = 0; i < dModel * dModel; i++) {
        W->data[i] = ((float)(i % 23) - 11.0f) * 0.05f;
    }

    tensorRequiresGrad(X);
    tensorRequiresGrad(W);

    // 1. Forward Matmul
    for (int i = 0; i < warmup; i++) {
        Tensor *out = multiply(X, W);
        tensorFree(out);
    }
    double t0 = get_time_sec();
    for (int i = 0; i < iters; i++) {
        Tensor *out = multiply(X, W);
        tensorFree(out);
    }
    double fwd_matmul_us = ((get_time_sec() - t0) / iters) * 1e6;
    printf("KERNEL multiply %.2f\n", fwd_matmul_us);

    // 2. Backward Matmul A
    for (int i = 0; i < warmup; i++) {
        multiplyBackwardA(X, W, dC);
    }
    t0 = get_time_sec();
    for (int i = 0; i < iters; i++) {
        multiplyBackwardA(X, W, dC);
    }
    double bwd_matmul_a_us = ((get_time_sec() - t0) / iters) * 1e6;
    printf("KERNEL multiplyBackwardA %.2f\n", bwd_matmul_a_us);

    // 3. Backward Matmul B
    for (int i = 0; i < warmup; i++) {
        multiplyBackwardB(X, W, dC);
    }
    t0 = get_time_sec();
    for (int i = 0; i < iters; i++) {
        multiplyBackwardB(X, W, dC);
    }
    double bwd_matmul_b_us = ((get_time_sec() - t0) / iters) * 1e6;
    printf("KERNEL multiplyBackwardB %.2f\n", bwd_matmul_b_us);

    // 4. LayerNorm Forward + Backward
    for (int i = 0; i < warmup; i++) {
        Tensor *normed = layerNormalization(X);
        layerNormBackward(X, dC);
        tensorFree(normed);
    }
    t0 = get_time_sec();
    for (int i = 0; i < iters; i++) {
        Tensor *normed = layerNormalization(X);
        layerNormBackward(X, dC);
        tensorFree(normed);
    }
    double layernorm_us = ((get_time_sec() - t0) / iters) * 1e6;
    printf("KERNEL layerNorm_fwd_bwd %.2f\n", layernorm_us);

    // 5. LeakyReLU Forward + Backward
    for (int i = 0; i < warmup; i++) {
        Tensor *act = leakyRelu(X, 0.01f);
        leakyReluBackward(X, dC, 0.01f);
        tensorFree(act);
    }
    t0 = get_time_sec();
    for (int i = 0; i < iters; i++) {
        Tensor *act = leakyRelu(X, 0.01f);
        leakyReluBackward(X, dC, 0.01f);
        tensorFree(act);
    }
    double leaky_us = ((get_time_sec() - t0) / iters) * 1e6;
    printf("KERNEL leakyRelu_fwd_bwd %.2f\n", leaky_us);

    tensorFree(X);
    tensorFree(W);
    tensorFree(dC);

    return 0;
}
