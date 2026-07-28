#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "attention.h"
#include "encoder.h"
#include "../training/lossFunctions.h"
#include "../training/tensorGrad.h"
#include "tests.h"

extern ModelConfig modelConfig;

void encoderTest() {
    Tensor *input = tensorCreate(3, modelConfig.dModel);
    randomDataFill(input);

    Transformer *transformer = transformerCreate(&modelConfig);
    if (!transformer) {
        printf("Error: Transformer allocation failed.\n");
        tensorFree(input);
        return;
    }

    for (int i = 0; i < modelConfig.layers; i++) {
        randomDataFill(transformer->layers[i].W_Q), randomDataFill(transformer->layers[i].W_K);
        randomDataFill(transformer->layers[i].W_V), randomDataFill(transformer->layers[i].W_O);
        randomDataFill(transformer->layers[i].W1), randomDataFill(transformer->layers[i].W2);
        randomDataFill(transformer->layers[i].B1), randomDataFill(transformer->layers[i].B2);


        tensorRequiresGrad(transformer->layers[i].W_Q), tensorRequiresGrad(transformer->layers[i].W_K);
        tensorRequiresGrad(transformer->layers[i].W_V), tensorRequiresGrad(transformer->layers[i].W_O);
        tensorRequiresGrad(transformer->layers[i].W1), tensorRequiresGrad(transformer->layers[i].W2);
        tensorRequiresGrad(transformer->layers[i].B1), tensorRequiresGrad(transformer->layers[i].B2);
    }

    tensorRequiresGrad(transformer->classW), tensorRequiresGrad(transformer->classB);

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