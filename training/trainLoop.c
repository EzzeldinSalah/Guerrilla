#include <stdlib.h>
#include "trainLoop.h"
#include "lossFunctions.h"
#include "encoderGrad.h"

static float trainForwardBackward (Transformer *transformer, Tensor *input, int trueClass, ModelConfig *modelConfig) {
    Tensor **layerInputs = malloc(sizeof(Tensor*) * (modelConfig->layers + 1));
    if (!layerInputs) return 0.0f;

    layerInputs[0] = input;
    for (int i = 0; i < modelConfig->layers; i++)
        layerInputs[i + 1] = encoderLayerForward(layerInputs[i], &transformer->layers[i], modelConfig);

    Tensor *pooled = meanPool(layerInputs[modelConfig->layers]);
    Tensor *probs = classificationHead(pooled, transformer->classW, transformer->classB);
    float loss = crossEntropyLoss(probs, trueClass);

    Tensor *dLogits = tensorCreate(probs->rows, probs->cols);
    crossEntropyBackward(dLogits, probs, trueClass);
    classificationHeadBackward(pooled, transformer->classW, transformer->classB, dLogits);

    Tensor *dPooled = tensorCreate(pooled->rows, pooled->cols);
    for (int i = 0; i < pooled->rows * pooled->cols; i++)
        dPooled->data[i] = pooled->grad ? pooled->grad[i] : 0.0f;

    Tensor *dEncoderOutput = meanPoolBackward(layerInputs[modelConfig->layers], dPooled);
    Tensor *dInput = encoderStackBackward(layerInputs, transformer, modelConfig->layers, dEncoderOutput, modelConfig);

    tensorFree(dInput), tensorFree(dEncoderOutput), tensorFree(dPooled), tensorFree(dLogits);
    tensorFree(pooled), tensorFree(probs);

    for (int i = 1; i <= modelConfig->layers; i++)
        tensorFree(layerInputs[i]);
    free(layerInputs);

    return loss;
}

float trainSgd (Transformer *transformer, Tensor *input, int trueClass, ModelConfig *modelConfig, float learningRate) {
    zeroTransformerGrad(transformer, modelConfig);
    float loss = trainForwardBackward(transformer, input, trueClass, modelConfig);
    sgdTransformer(transformer, modelConfig, learningRate);
    
    return loss;
}

float trainAdam (Transformer *transformer, Tensor *input, int trueClass, ModelConfig *modelConfig, AdamOptimizer *optimizer) {
    zeroTransformerGrad(transformer, modelConfig);
    float loss = trainForwardBackward(transformer, input, trueClass, modelConfig);
    adamTransformer(transformer, modelConfig, optimizer);

    return loss;
}
