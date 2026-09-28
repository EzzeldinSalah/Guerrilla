#pragma once
#include "tensor.h"

void leakyReluBackward (Tensor *A, Tensor *dC, float alpha);
void multiplyBackwardA (Tensor *A, Tensor *B, Tensor *dC);
void multiplyBackwardAData (Tensor *A, Tensor *B, Tensor *dC, Tensor *dA);
void multiplyBackwardB (Tensor *A, Tensor *B, Tensor *dC);
void addBiasBackward (Tensor *bias, Tensor *upstream);
void layerNormBackward (Tensor *x, Tensor *dy);
void softmaxBackward (Tensor *scores, Tensor *A, Tensor *dA);
