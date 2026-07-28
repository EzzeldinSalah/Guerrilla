#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "attentionGrad.h"
#include "attention.h"
#include "tensorGrad.h"

void singleHeadAttentionBackward (Tensor *Q, Tensor *K, Tensor *V, Tensor *dScores, Tensor *dQ, Tensor *dK, Tensor *dV, int d_k) {
    // 1. re-compute needed forward intermediates
    Tensor *Kt = transpose(K), *scores = multiply(Q, Kt);
    float scale_factor = 1.0f / sqrtf((float)d_k);
    Tensor *scaledScores = scale(scores, scale_factor), *softed = softmax(scaledScores);


    // 2. Output = softed * V
    // dV = softed^T * dScores
    multiplyBackwardB(softed, V, dScores);
    if (dV && V->grad) {
        for (int i = 0; i < dV->rows * dV->cols; i++)
            dV->data[i] += V->grad[i];
    }

    // dSofted = dScores * V^T
    Tensor *dSofted = tensorCreate(softed->rows, softed->cols);
    multiplyBackwardA(softed, V, dScores);
    for (int i = 0; i < softed->rows * softed->cols; i++) {
        dSofted->data[i] = softed->grad[i];
    }

    // 3. Backprop Softmax
    Tensor *dScaled = tensorCreate(scaledScores->rows, scaledScores->cols);
    softmaxBackward(scaledScores, softed, dSofted);
    for (int i = 0; i < dScaled->rows * dScaled->cols; i++)
        dScaled->data[i] = scaledScores->grad[i];

    // 4. Backprop scaling factor
    Tensor *dUnscaled = scale(dScaled, scale_factor);

    // 5. Backprop [Q * K^T]
    multiplyBackwardA(Q, Kt, dUnscaled), multiplyBackwardB(Q, Kt, dUnscaled);

    if (dQ && Q->grad) {
        for (int i = 0; i < dQ->rows * dQ->cols; i++)
            dQ->data[i] += Q->grad[i];
    }

    if (dK && Kt->grad) {
        for (int i = 0; i < dK->rows; i++)
            for (int j = 0; j < dK->cols; j++)
                dK->data[i * dK->cols + j] += Kt->grad[j * Kt->cols + i];
    }

    tensorFree(Kt), tensorFree(scores), tensorFree(scaledScores), tensorFree(softed);
    tensorFree(dSofted), tensorFree(dScaled), tensorFree(dUnscaled);
}
