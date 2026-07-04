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

Tensor *feedForward (Tensor *input, Tensor *W1, Tensor *W2, Tensor *B1, Tensor *B2, ModelConfig *modelConfig);
