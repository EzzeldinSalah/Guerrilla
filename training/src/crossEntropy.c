#include <math.h>
#include <stdio.h>
#include "../../include/tensor.h"

// fine since it's 1 x classes
float crossEntropyLoss(Tensor *probs, int trueClass) {
    return -logf(probs->data[trueClass] + 1e-7f);
}

void layerNormBackward(Tensor *x, Tensor *dy) {
    if (!x || !dy) return;
    if (!x->grad) {
        printf("Allocating Gradient Storage ...\n");
        tensorRequiresGrad(x);
    }

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

        float invStd = 1.0f / sqrtf(variance + 1e-5f); // Division is 5x to 10x slower than multiplication

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

void crossEntropyBackward (Tensor *dLogits, Tensor *probs, int trueClass) {
    for (int i = 0; i < probs->rows; i++) {
        for (int j = 0; j < probs->cols; j++) {       
            dLogits->data[i * probs->cols + j] = 
                probs->data[i * probs->cols + j] - (j == trueClass ? 1.0f : 0.0f);
        }
    }
}

void multiplyBackwardA(Tensor *A, Tensor *B, Tensor *dC) {
    if (A->rows != dC->rows || B->cols != dC->cols || A->cols != B->rows) {
        printf("multiplyBackwardA: shape mismatch (A: %dx%d, B: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols, dC->rows, dC->cols);
        return;
    }

	if (!A->grad) {
		printf("Allocating Gradient Storage ...\n");
		tensorRequiresGrad(A);
	}

    Tensor *BT = transpose(B);
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            for (int k = 0; k < dC->cols; k++)
                A->grad[i * A->cols + j] += dC->data[i * dC->cols + k] * BT->data[k * BT->cols + j];

    tensorFree(BT);
}

void multiplyBackwardB(Tensor *A, Tensor *B, Tensor *dC) {
	if (A->rows != dC->rows || B->cols != dC->cols || A->cols != B->rows) {
        printf("multiplyBackwardB: shape mismatch (A: %dx%d, B: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols, dC->rows, dC->cols);
        return;
    }

    if (!B->grad) {
		printf("Allocating Gradient Storage ...\n");
		tensorRequiresGrad(B);
	}

    Tensor *AT = transpose(A);
    for (int i = 0; i < B->rows; i++)
        for (int j = 0; j < B->cols; j++)
            for (int k = 0; k < AT->cols; k++)
                B->grad[i * B->cols + j] += AT->data[i * AT->cols + k] * dC->data[k * dC->cols + j];

    tensorFree(AT);
}

void addBiasBackward(Tensor *bias, Tensor *upstream) {
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

void classificationHeadBackward(Tensor *pooled, Tensor *classW, Tensor *classB, Tensor *dLogits) {
    addBiasBackward(classB, dLogits);   
    multiplyBackwardA(pooled, classW, dLogits), multiplyBackwardB(pooled, classW, dLogits);
}