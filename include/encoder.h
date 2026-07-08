#pragma once
#include "tensor.h"
#include "attention.h"

typedef struct {
    Tensor *W_Q;
    Tensor *W_K;
    Tensor *W_V;
    Tensor *W_O;
    Tensor *W1;
    Tensor *W2;
    Tensor *B1;
    Tensor *B2;
} EncoderLayer;

typedef struct {
    EncoderLayer *layers;
    int classes; // normal or anomalous
    Tensor *classW;
    Tensor *classB;
} Transformer;

Transformer *transformerCreate(ModelConfig *modelConfig);
void transformerFree(Transformer *transformer, ModelConfig *modelConfig);

Tensor *encoderStack(Tensor *input, Transformer *transformer, int numLayers, ModelConfig *modelConfig);
Tensor *meanPool(Tensor *input);
Tensor *classificationHead(Tensor *pooled, Tensor *classW, Tensor *classB);