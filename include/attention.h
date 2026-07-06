#pragma once
#include "tensor.h"

#define SEQ_LEN    1024
#define D_MODEL    64
#define NUM_HEADS  4
#define NUM_LAYERS 6

typedef struct {
    int seqLen;
    int dModel;
    int heads;
    int layers;
    int dk;
} ModelConfig;


Tensor **multiHeadAttention(Tensor *query, Tensor *key, Tensor *value, ModelConfig *modelConfig);
Tensor *tensorConcat (Tensor **heads, ModelConfig *modelConfig);