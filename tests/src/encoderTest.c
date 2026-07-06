#include <stdio.h>

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

    Tensor *output = encoderStack(input, transformer, modelConfig.layers, &modelConfig);
    printf("Encoder Dimensions: %d x %d\n", output->rows, output->cols);
    tensorPrint(output);

    tensorFree(input), tensorFree(output);
    transformerFree(transformer, &modelConfig);
}