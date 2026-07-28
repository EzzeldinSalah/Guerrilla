#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "attention.h"
#include "../training/attentionGrad.h"
#include "encoder.h"
#include "../training/encoderGrad.h"
#include "../training/lossFunctions.h"
#include "../training/optimizer.h"
#include "../training/tensorGrad.h"
#include "../training/trainLoop.h"
#include "tests.h"

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
    float loss = crossEntropyLoss(probs, 1);
    crossEntropyBackward(dLogits, probs, 1);
    classificationHeadBackward(pooled, transformer->classW, transformer->classB, dLogits);

    printf("classification head gradients\n");
    printf("cross entropy loss: %f\n", loss);
    printf("classW->grad[0]: %f\n", transformer->classW->grad[0]);
    printf("classB->grad[0]: %f\n\n", transformer->classB->grad[0]);

    float oldWeight = transformer->classW->data[0];
    sgd(transformer->classW, 0.01f);
    zeroGrad(transformer->classW);

    printf("sgd optimizer test\n");
    printf("classW delta: %f\n", transformer->classW->data[0] - oldWeight);
    printf("classW grad after zero: %f\n\n", transformer->classW->grad[0]);

    transformer->classW->grad[0] = 0.5f;
    AdamOptimizer *adamOptimizer = adamCreate(transformer, &modelConfig, 0.001f);
    oldWeight = transformer->classW->data[0];
    adam(transformer->classW, &adamOptimizer->classW, 0.001f, 0.9f, 0.999f, 1e-8f, 1);

    printf("adam optimizer test\n");
    printf("classW delta: %f\n\n", transformer->classW->data[0] - oldWeight);
    adamFree(adamOptimizer, &modelConfig);


    Tensor *logitsDummy = tensorCreate(probs->rows, probs->cols), *dA_softmax = tensorCreate(probs->rows, probs->cols);
    randomDataFill(dA_softmax);

    softmaxBackward(logitsDummy, probs, dA_softmax);

    printf("softmax backward test\n");
    printf("logitsDummy->grad[0]: %f\n\n", logitsDummy->grad ? logitsDummy->grad[0] : 0.0f);

    Tensor *dOutput = tensorCreate(output->rows, output->cols);
    randomDataFill(dOutput);

    layerNormBackward(output, dOutput);

    printf("layerNorm backward test\n");
    printf("output->grad[0]: %f\n\n", output->grad ? output->grad[0] : 0.0f);

    Tensor *Q = tensorCreate(3, modelConfig.dk), *K = tensorCreate(3, modelConfig.dk), *V = tensorCreate(3, modelConfig.dk);
    Tensor *dScores = tensorCreate(3, modelConfig.dk), *dQ = tensorCreate(3, modelConfig.dk);
    Tensor *dK = tensorCreate(3, modelConfig.dk), *dV = tensorCreate(3, modelConfig.dk);
    randomDataFill(Q), randomDataFill(K), randomDataFill(V), randomDataFill(dScores);
    for (int i = 0; i < dQ->rows * dQ->cols; i++)
        dQ->data[i] = 0.0f, dK->data[i] = 0.0f, dV->data[i] = 0.0f;

    singleHeadAttentionBackward(Q, K, V, dScores, dQ, dK, dV, modelConfig.dk);

    printf("attention backward test\n");
    printf("dQ[0]: %f\n", dQ->data[0]);
    printf("dK[0]: %f\n", dK->data[0]);
    printf("dV[0]: %f\n\n", dV->data[0]);

    AdamOptimizer *optimizer = adamCreate (transformer, &modelConfig, 0.01f);
    printf("Training Loop | Adam | 200 Steps\n");
    float trainLoss = 0.0f;
    for (int step = 1; step <= 200; step++) {
        trainLoss = trainAdam (transformer, input, 1, &modelConfig, optimizer);
        if (step == 1 || step % 50 == 0)
            printf("step %d loss: %f\n", step, trainLoss);
    }
    printf("\n");
    adamFree (optimizer, &modelConfig);

    tensorFree(Q), tensorFree(K), tensorFree(V), tensorFree(dScores), tensorFree(dQ), tensorFree(dK), tensorFree(dV);
    tensorFree(logitsDummy), tensorFree(dA_softmax), tensorFree(dOutput), tensorFree(dLogits);
    tensorFree(pooled), tensorFree(probs), tensorFree(input), tensorFree(output);
    transformerFree(transformer, &modelConfig);
}
