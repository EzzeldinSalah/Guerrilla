#pragma once
#include "tensor.h"

void addBackward (Tensor *A, Tensor *B, Tensor *dC);
void transposeBackward (Tensor *A, Tensor *dC);
void scaleBackward (Tensor *A, Tensor *dC, float scale);
void reluBackward (Tensor *A, Tensor *dC);
void leakyReluBackward (Tensor *A, Tensor *dC, float alpha);
void multiplyBackwardA (Tensor *A, Tensor *B, Tensor *dC);
void multiplyBackwardB (Tensor *A, Tensor *B, Tensor *dC);
void addBiasBackward (Tensor *bias, Tensor *upstream);
void layerNormBackward (Tensor *x, Tensor *dy);
void softmaxBackward (Tensor *scores, Tensor *A, Tensor *dA);
