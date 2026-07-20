#pragma once
#include "../../include/tensor.h"

void multiplyBackwardA(Tensor *A, Tensor *B, Tensor *dC);
void multiplyBackwardB(Tensor *A, Tensor *B, Tensor *dC);
void layerNormBackward(Tensor *x, Tensor *dy);
void softmaxBackward(Tensor *scores, Tensor *A, Tensor *dA);
void addBiasBackward(Tensor *bias, Tensor *upstream);
void classificationHeadBackward(Tensor *pooled, Tensor *classW, Tensor *classB, Tensor *dLogits);
void crossEntropyBackward (Tensor *dLogits, Tensor *probs, int trueClass);
float crossEntropyLoss (Tensor *probs, int trueClass);