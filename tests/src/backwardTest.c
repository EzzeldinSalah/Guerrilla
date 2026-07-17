#include <stdio.h>
#include <stdlib.h>
#include "../../include/tensor.h"
#include "../../include/encoder.h"
#include "../../training/include/crossEntropy.h"
#include "../include/testUtils.h"

extern ModelConfig modelConfig;

void backwardTest() {
    Tensor *input = tensorCreate(3, modelConfig.dModel);
    randomDataFill(input);

    Transformer *transformer = transformerCreate(&modelConfig);
    if (!transformer) {
        tensorFree(input);
        return;
    }

    for (int i = 0; i < modelConfig.layers; i++) {
        randomDataFill(transformer->layers[i].W_Q), randomDataFill(transformer->layers[i].W_K);
        randomDataFill(transformer->layers[i].W_V), randomDataFill(transformer->layers[i].W_O);
        randomDataFill(transformer->layers[i].W1), randomDataFill(transformer->layers[i].W2);
        randomDataFill(transformer->layers[i].B1), randomDataFill(transformer->layers[i].B2);
    }

    tensorRequiresGrad(transformer->classW), tensorRequiresGrad(transformer->classB);

    for (int i = 0; i < transformer->classW->rows * transformer->classW->cols; i++)
        transformer->classW->data[i] = ((float)rand() / (float)RAND_MAX) * 0.1f;
    
    for (int i = 0; i < transformer->classB->cols; i++)
        transformer->classB->data[i] = 0.0f;

    Tensor *output = encoderStack(input, transformer, modelConfig.layers, &modelConfig);
    Tensor *pooled = meanPool(output);
    Tensor *probs = classificationHead(pooled, transformer->classW, transformer->classB);

    Tensor *dLogits = tensorCreate(probs->rows, probs->cols);
    crossEntropyBackward(dLogits, probs, 1);
    classificationHeadBackward(pooled, transformer->classW, transformer->classB, dLogits);

    for (int i = 0; i < transformer->classW->rows * transformer->classW->cols; i++)
        printf("%f ", transformer->classW->grad[i]);
    printf("\n");

    for (int i = 0; i < transformer->classB->cols; i++)
        printf("%f ", transformer->classB->grad[i]);
    printf("\n");

    tensorFree(dLogits); tensorFree(pooled), tensorFree(probs);
    tensorFree(input), tensorFree(output);
    transformerFree(transformer, &modelConfig);
}