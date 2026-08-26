#include <math.h>
#include <stdio.h>
#include "tensorGrad.h"

void addBackward (Tensor *A, Tensor *B, Tensor *dC) {
    if (A->rows != B->rows || A->cols != B->cols ||
        A->rows != dC->rows || A->cols != dC->cols) {
        printf("addBackward: shape mismatch (A: %dx%d, B: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols, dC->rows, dC->cols);

        return;
    }

    if (!A->grad) tensorRequiresGrad(A);
    if (!B->grad) tensorRequiresGrad(B);

    int totalSize = A->rows * A->cols;
    for (int i = 0; i < totalSize; i++)
        A->grad[i] += dC->data[i], B->grad[i] += dC->data[i];
}

void transposeBackward (Tensor *A, Tensor *dC) {
    if (A->rows != dC->cols || A->cols != dC->rows) {
        printf("transposeBackward: shape mismatch (A: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, dC->rows, dC->cols);

        return;
    }

    if (!A->grad) tensorRequiresGrad(A);

    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            A->grad[i * A->cols + j] += dC->data[j * dC->cols + i];
}

void scaleBackward (Tensor *A, Tensor *dC, float scale) {
    if (A->rows != dC->rows || A->cols != dC->cols) {
        printf("scaleBackward: shape mismatch (A: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, dC->rows, dC->cols);

        return;
    }

    if (!A->grad) tensorRequiresGrad(A);

    int totalSize = A->rows * A->cols;
    for (int i = 0; i < totalSize; i++)
        A->grad[i] += dC->data[i] * scale;
}

void reluBackward (Tensor *A, Tensor *dC) {
    if (A->rows != dC->rows || A->cols != dC->cols) {
        printf("reluBackward: shape mismatch (A: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, dC->rows, dC->cols);

        return;
    }

    if (!A->grad) tensorRequiresGrad(A);

    int totalSize = A->rows * A->cols;
    for (int i = 0; i < totalSize; i++)
        A->grad[i] += A->data[i] > 0 ? dC->data[i] : 0.0f;
}

void leakyReluBackward (Tensor *A, Tensor *dC, float alpha) {
    if (A->rows != dC->rows || A->cols != dC->cols) {
        printf("leakyReluBackward: shape mismatch (A: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, dC->rows, dC->cols);

        return;
    }

    if (!A->grad) tensorRequiresGrad(A);

    int totalSize = A->rows * A->cols;
    for (int i = 0; i < totalSize; i++)
        A->grad[i] += A->data[i] > 0 ? dC->data[i] : alpha * dC->data[i];
}

void layerNormBackward (Tensor *x, Tensor *dy) {
    if (!x || !dy) return;
    if (!x->grad) tensorRequiresGrad(x);

    for (int i = 0; i < x->rows; i++) {
        // int rowOffset = i * cols; -> computed once

        float mean = 0.0f, variance = 0.0f;
        for (int j = 0; j < x->cols; j++)
            mean += x->data[i * x->cols + j];
        mean /= x->cols;

        for (int j = 0; j < x->cols; j++) {
            float diff = x->data[i * x->cols + j] - mean;
            variance += diff * diff;
        }
        variance /= x->cols;

        float invStd = 1.0f / sqrtf(variance + 1e-5f); // fun fact: / is 5x to 10x slower than *

        float meanDy = 0.0f, meanDyY = 0.0f;
        for (int j = 0; j < x->cols; j++) {
            float xi = x->data[i * x->cols + j], yi = (xi - mean) * invStd;
            float dyi = dy->data[i * x->cols + j];

            meanDy += dyi, meanDyY += dyi * yi;
        }
        meanDy /= x->cols, meanDyY /= x->cols;

        for (int j = 0; j < x->cols; j++) {
            float xi = x->data[i * x->cols + j], yi = (xi - mean) * invStd;
            float dyi = dy->data[i * x->cols + j];

            float grad_val = (dyi - meanDy - yi * meanDyY) * invStd;
            x->grad[i * x->cols + j] += grad_val;
        }
    }
}

void softmaxBackward (Tensor *scores, Tensor *A, Tensor *dA) {
    if (!scores || !A || !dA) return;

    if (scores->rows != A->rows || scores->cols != A->cols ||
        A->rows != dA->rows || A->cols != dA->cols) {
        printf("softmaxBackward: shape mismatch (scores: %dx%d, A: %dx%d, dA: %dx%d)\n",
               scores->rows, scores->cols, A->rows, A->cols, dA->rows, dA->cols);

        return;
    }
    
    if (!scores->grad) tensorRequiresGrad(scores);


    for (int i = 0; i <  A->rows; i++) {
        int offset = i * A->cols;

        float dot = 0.0f;
        for (int j = 0; j < A->cols; j++)
            dot += A->data[offset + j] * dA->data[offset + j];

        for (int j = 0; j < A->cols; j++) {
            float a = A->data[offset + j], da = dA->data[offset + j];
            scores->grad[offset + j] += a * (da - dot);
        }
    }
}


void multiplyBackwardA (Tensor *A, Tensor *B, Tensor *dC) {
    if (A->rows != dC->rows || B->cols != dC->cols || A->cols != B->rows) {
        printf("multiplyBackwardA: shape mismatch (A: %dx%d, B: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols, dC->rows, dC->cols);
        return;
    }

    if (!A->grad) tensorRequiresGrad(A);

    const int TILE = 32;
    for (int ii = 0; ii < A->rows; ii += TILE) {
        for (int jj = 0; jj < A->cols; jj += TILE) {
            for (int kk = 0; kk < dC->cols; kk += TILE) {

                int iEnd = (ii + TILE < A->rows) ? ii + TILE : A->rows;
                int jEnd = (jj + TILE < A->cols) ? jj + TILE : A->cols;
                int kEnd = (kk + TILE < dC->cols) ? kk + TILE : dC->cols;

                for (int i = ii; i < iEnd; i++) {
                    for (int j = jj; j < jEnd; j++) {
                        float tempSum = 0.0f;
                        for (int k = kk; k < kEnd; k++)
                            tempSum += dC->data[i * dC->cols + k] * B->data[j * B->cols + k];

                        A->grad[i * A->cols + j] += tempSum;
                    }
                }

            }
        }
    }
}

void multiplyBackwardAData (Tensor *A, Tensor *B, Tensor *dC, Tensor *dA) {
    if (A->rows != dC->rows || B->cols != dC->cols || A->cols != B->rows ||
        A->rows != dA->rows || A->cols != dA->cols) {
        printf("multiplyBackwardAData: shape mismatch (A: %dx%d, B: %dx%d, dC: %dx%d, dA: %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols, dC->rows, dC->cols, dA->rows, dA->cols);
        return;
    }

    const int TILE = 32;
    for (int ii = 0; ii < A->rows; ii += TILE) {
        for (int jj = 0; jj < A->cols; jj += TILE) {
            for (int kk = 0; kk < dC->cols; kk += TILE) {

                int iEnd = (ii + TILE < A->rows) ? ii + TILE : A->rows;
                int jEnd = (jj + TILE < A->cols) ? jj + TILE : A->cols;
                int kEnd = (kk + TILE < dC->cols) ? kk + TILE : dC->cols;

                for (int i = ii; i < iEnd; i++) {
                    for (int j = jj; j < jEnd; j++) {
                        float tempSum = 0.0f;
                        for (int k = kk; k < kEnd; k++)
                            tempSum += dC->data[i * dC->cols + k] * B->data[j * B->cols + k];

                        dA->data[i * dA->cols + j] += tempSum;
                    }
                }

            }
        }
    }
}

void multiplyBackwardB (Tensor *A, Tensor *B, Tensor *dC) {
    if (A->rows != dC->rows || B->cols != dC->cols || A->cols != B->rows) {
        printf("multiplyBackwardB: shape mismatch (A: %dx%d, B: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols, dC->rows, dC->cols);
        return;
    }

    if (!B->grad) tensorRequiresGrad(B);

    const int TILE = 32;
    for (int ii = 0; ii < A->rows; ii += TILE) {
        for (int kk = 0; kk < A->cols; kk += TILE) {
            for (int jj = 0; jj < B->cols; jj += TILE) {

                int iEnd = (ii + TILE < A->rows) ? ii + TILE : A->rows;
                int kEnd = (kk + TILE < A->cols) ? kk + TILE : A->cols;
                int jEnd = (jj + TILE < B->cols) ? jj + TILE : B->cols;

                for (int i = ii; i < iEnd; i++) {
                    for (int k = kk; k < kEnd; k++) {
                        float r = A->data[i * A->cols + k];
                        for (int j = jj; j < jEnd; j++)
                            B->grad[k * B->cols + j] += r * dC->data[i * dC->cols + j];
                    }
                }

            }
        }
    }
}

void addBiasBackward (Tensor *bias, Tensor *upstream) {
    if (bias->rows != 1 || bias->cols != upstream->cols) {
        printf("addBiasBackward: shape mismatch (bias: %dx%d, upstream: %dx%d)\n",
               bias->rows, bias->cols, upstream->rows, upstream->cols);
        return;
    }

    if (!bias->grad) tensorRequiresGrad(bias);

    for (int i = 0; i < upstream->rows; i++)
        for (int j = 0; j < upstream->cols; j++)
            bias->grad[j] += upstream->data[i * upstream->cols + j];
}
