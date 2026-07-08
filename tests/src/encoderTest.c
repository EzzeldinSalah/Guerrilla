#include <stdio.h>
#include <stdlib.h>

#include "../../include/tensor.h"
#include "../../include/attention.h"
#include "../../include/encoder.h"
#include "../include/testUtils.h"

extern ModelConfig modelConfig;

void encoderTest() {
    Tensor *input = tensorCreate(3, modelConfig.dModel);
    randomFill(input);

    Transformer *transformer = transformerCreate(&modelConfig);
    if (!transformer) {
        printf("Error: Transformer allocation failed.\n");
        tensorFree(input);
        return;
    }

    for (int i = 0; i < modelConfig.layers; i++) {
        randomFill(transformer->layers[i].W_Q), randomFill(transformer->layers[i].W_K);
        randomFill(transformer->layers[i].W_V), randomFill(transformer->layers[i].W_O);
        randomFill(transformer->layers[i].W1), randomFill(transformer->layers[i].W2);
        randomFill(transformer->layers[i].B1), randomFill(transformer->layers[i].B2);
    }


    for (int i = 0; i < transformer->classW->rows * transformer->classW->cols; i++)
        transformer->classW->data[i] = ((float)rand() / (float)RAND_MAX) * 0.1f;
    for (int i = 0; i < transformer->classB->cols; i++)
        transformer->classB->data[i] = 0.0f;

    Tensor *output = encoderStack(input, transformer, modelConfig.layers, &modelConfig);
    printf("Encoder Dimensions:\n");
    tensorPrint(output);

    printf("Pooled:\n");
    Tensor *pooled = meanPool(output);
    tensorPrint(pooled);

    printf("Classes:\n");
    Tensor *probs = classificationHead(pooled, transformer->classW, transformer->classB);
    tensorPrint(probs);


    tensorFree(pooled), tensorFree(probs);
    tensorFree(input), tensorFree(output);
    transformerFree(transformer, &modelConfig);
}