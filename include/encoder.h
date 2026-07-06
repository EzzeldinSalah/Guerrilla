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
    // final classification layers will live down here
} Transformer;

Tensor *feedForward(Tensor *x, Tensor *W1, Tensor *B1, Tensor *W2, Tensor *B2);
Tensor *encoderLayerForward(Tensor *input, EncoderLayer *layer, ModelConfig *modelConfig);
Tensor *encoderStack(Tensor *input, Transformer *transformer, int numLayers, ModelConfig *modelConfig);

Transformer *transformerCreate(ModelConfig *modelConfig);
void transformerFree(Transformer *transformer, ModelConfig *modelConfig);