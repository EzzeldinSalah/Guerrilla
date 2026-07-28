#pragma once
#include "tensor.h"

void singleHeadAttentionBackward (Tensor *Q, Tensor *K, Tensor *V, Tensor *dScores, Tensor *dQ, Tensor *dK, Tensor *dV, int d_k);
