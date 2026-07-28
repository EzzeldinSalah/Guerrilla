#pragma once
#include "tensor.h"
#include "encoder.h"
#include "optimizer.h"

float trainSgd (Transformer *transformer, Tensor *input, int trueClass, ModelConfig *modelConfig, float learningRate);
float trainAdam (Transformer *transformer, Tensor *input, int trueClass, ModelConfig *modelConfig, AdamOptimizer *optimizer);
