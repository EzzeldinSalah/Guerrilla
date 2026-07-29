#include <math.h>
#include <stdio.h>
#include "lossFunctions.h"

float crossEntropyLoss (Tensor *probs, int trueClass) {
    if (!probs || trueClass < 0 || trueClass >= probs->cols) {
        printf("crossEntropyLoss: bad class index %d for %d classes\n", trueClass, probs ? probs->cols : 0);
        return 0.0f;
    }

    float loss = 0.0f;
    for (int i = 0; i < probs->rows; i++) {
        float p = probs->data[i * probs->cols + trueClass];
        if (p < 1e-7f) p = 1e-7f;
        loss -= logf(p);
    }

    return loss / (float)probs->rows;
}

void crossEntropyBackward (Tensor *dLogits, Tensor *probs, int trueClass) {
    if (!dLogits || !probs) return;

    if (dLogits->rows != probs->rows || dLogits->cols != probs->cols) {
        printf("crossEntropyBackward: shape mismatch (dLogits: %dx%d, probs: %dx%d)\n",
            dLogits->rows, dLogits->cols, probs->rows, probs->cols);

        return;
    }

    if (trueClass < 0 || trueClass >= probs->cols) {
        printf("crossEntropyBackward: bad class index %d for %d classes\n",
               trueClass, probs->cols);

        return;
    }

    float batchScale = 1.0f / (float)probs->rows;
    for (int i = 0; i < probs->rows; i++)
        for (int j = 0; j < probs->cols; j++)
            dLogits->data[i * probs->cols + j] =
                (probs->data[i * probs->cols + j] - (j == trueClass ? 1.0f : 0.0f)) * batchScale;
}
