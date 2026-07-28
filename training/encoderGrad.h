#pragma once
#include "tensor.h"
#include "attention.h"
#include "encoder.h"

void classificationHeadBackward (Tensor *pooled, Tensor *classW, Tensor *classB, Tensor *dLogits);
Tensor *meanPoolBackward (Tensor *input, Tensor *dPooled);
Tensor *encoderLayerBackward (Tensor *input, EncoderLayer *layer, Tensor *dOutput, ModelConfig *modelConfig);
Tensor *encoderStackBackward (Tensor **layerInputs, Transformer *transformer, int numLayers, Tensor *dOutput, ModelConfig *modelConfig);
