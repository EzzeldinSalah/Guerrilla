#pragma once
#include "tensor.h"

float crossEntropyLoss (Tensor *probs, int trueClass);
void crossEntropyBackward (Tensor *dLogits, Tensor *probs, int trueClass);
