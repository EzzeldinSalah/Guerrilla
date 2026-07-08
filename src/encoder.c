#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/attention.h"
#include "../include/encoder.h"

Transformer *transformerCreate(ModelConfig *modelConfig) {
    Transformer *transformer = malloc(sizeof(Transformer));
    if (!transformer) return NULL;

    transformer->layers = malloc(sizeof(EncoderLayer) * modelConfig->layers);
    if (!transformer->layers) {
        free(transformer);
        return NULL;
    }

    int dFF = modelConfig->dModel * 4;
    for (int i = 0; i < modelConfig->layers; i++) {
        transformer->layers[i].W_Q = tensorCreate(modelConfig->dModel, modelConfig->dModel);
        transformer->layers[i].W_K = tensorCreate(modelConfig->dModel, modelConfig->dModel);
        transformer->layers[i].W_V = tensorCreate(modelConfig->dModel, modelConfig->dModel);
        transformer->layers[i].W_O = tensorCreate(modelConfig->dModel, modelConfig->dModel);
        
        transformer->layers[i].W1 = tensorCreate(modelConfig->dModel, dFF);
        transformer->layers[i].W2 = tensorCreate(dFF, modelConfig->dModel);
        transformer->layers[i].B1 = tensorCreate(1, dFF);
        transformer->layers[i].B2 = tensorCreate(1, modelConfig->dModel);
    }

    transformer->classes = 2;
    transformer->classW = tensorCreate(modelConfig->dModel, transformer->classes);
    transformer->classB = tensorCreate(1, transformer->classes);

    return transformer;
}

void transformerFree(Transformer *transformer, ModelConfig *modelConfig) {
    if (!transformer) return;

    if (transformer->layers) {
        for (int i = 0; i < modelConfig->layers; i++) {
            tensorFree(transformer->layers[i].W_Q);
            tensorFree(transformer->layers[i].W_K);
            tensorFree(transformer->layers[i].W_V);
            tensorFree(transformer->layers[i].W_O);
            tensorFree(transformer->layers[i].W1);
            tensorFree(transformer->layers[i].W2);
            tensorFree(transformer->layers[i].B1);
            tensorFree(transformer->layers[i].B2);
        }
        free(transformer->layers);
    }

    tensorFree(transformer->classW), tensorFree(transformer->classB);

    free(transformer);
}

Tensor *feedForward(Tensor *x, Tensor *W1, Tensor *B1, Tensor *W2, Tensor *B2) {
    Tensor *theta = multiply(x, W1);
    Tensor *h = addBias(theta, B1);
    tensorFree(theta);

    Tensor *leakyH = leakyRelu(h, 0.01);
    tensorFree(h);

    Tensor *beta = multiply(leakyH, W2);
    tensorFree(leakyH);

    Tensor *FFN = addBias(beta, B2);
    tensorFree(beta);

    return FFN;
}

Tensor *encoderLayerForward(Tensor *input, EncoderLayer *layer, ModelConfig *modelConfig) {
    Tensor *Q = multiply(input, layer->W_Q), *K = multiply(input, layer->W_K), *V = multiply(input, layer->W_V);

    Tensor **heads = multiHeadAttention(Q, K, V, modelConfig);
    tensorFree(Q); tensorFree(K); tensorFree(V);

    Tensor *concatenatedHeads = tensorConcat(heads, modelConfig);
    for (int h = 0; h < modelConfig->heads; h++) tensorFree(heads[h]);
    free(heads);

    Tensor *attentionOut = multiply(concatenatedHeads, layer->W_O);
    tensorFree(concatenatedHeads);

    Tensor *firstResidualConnection = add(input, attentionOut);
    tensorFree(attentionOut); 

    Tensor *x = layerNormalization(firstResidualConnection);
    tensorFree(firstResidualConnection);

    Tensor *ffnOut = feedForward(x, layer->W1, layer->B1, layer->W2, layer->B2);

    Tensor *y = add(x, ffnOut);
    tensorFree(x); tensorFree(ffnOut);

    Tensor *output = layerNormalization(y);
    tensorFree(y);

    return output;
}


Tensor *encoderStack(Tensor *input, Transformer *transformer, int numLayers, ModelConfig *modelConfig) {
    Tensor *current = input;
    for (int i = 0; i < numLayers; i++) {
        Tensor *next = encoderLayerForward(current, &transformer->layers[i], modelConfig);
        if (i) tensorFree(current);
        current = next;
    }
    return current;
}


Tensor *meanPool(Tensor *input) {
    if (!input) return NULL;

    Tensor *output = tensorCreate(1, input->cols);

    for (int j = 0; j < input->cols; j++) {
        float tempSum = 0.0f;
        for (int i = 0; i < input->rows; i++)
            tempSum += input->data[i * input->cols + j];

        output->data[j] = tempSum / (float)input->rows; 
    }

    return output;
}

Tensor *classificationHead(Tensor *pooled, Tensor *classW, Tensor *classB) {
    Tensor *logits = multiply(pooled, classW);
    Tensor *biasedLogits = addBias(logits, classB);
    tensorFree(logits);

    Tensor *prob = softmax(biasedLogits);
    tensorFree(biasedLogits);

    return prob;
}