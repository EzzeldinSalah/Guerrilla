#pragma once
#include "tensor.h"
#include "encoder.h"

typedef struct {
    Tensor *m;
    Tensor *v;
} AdamState;

typedef struct {
    AdamState W_Q;
    AdamState W_K;
    AdamState W_V;
    AdamState W_O;
    AdamState W1;
    AdamState W2;
    AdamState B1;
    AdamState B2;
} AdamLayerState;

typedef struct {
    AdamLayerState *layers;
    AdamState classW;
    AdamState classB;
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    int timestep;
} AdamOptimizer;

void zeroGrad (Tensor *tensor);
void zeroTransformerGrad (Transformer *transformer, ModelConfig *modelConfig);
void sgd (Tensor *tensor, float learningRate);
void sgdTransformer (Transformer *transformer, ModelConfig *modelConfig, float learningRate);
AdamOptimizer *adamCreate (Transformer *transformer, ModelConfig *modelConfig, float learningRate);
void adamFree (AdamOptimizer *optimizer, ModelConfig *modelConfig);
void adam (Tensor *tensor, AdamState *state, float learningRate, float beta1, float beta2, float epsilon, int timestep);
void adamTransformer (Transformer *transformer, ModelConfig *modelConfig, AdamOptimizer *optimizer);
