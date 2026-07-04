#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/attention.h"
#include "../include/forward.h"

Tensor *feedForward(Tensor *x, Tensor *W1, Tensor *B1, Tensor *W2, Tensor *B2) {
    Tensor *theta = multiply(x, W1);
    Tensor *h = addBias(theta, B1);
    tensorFree(theta);

    Tensor *leakyH = leakyRelu(h, 0.01);
    tensorFree(h);

    Tensor *beta = multiply(leakyH, W2);
    tensorFree(leakyH);

    Tensor *FFN = addBias(beta, B2);
    tensorFree(beta);

    return FFN;
}

Tensor *encoderLayerForward(Tensor *input, EncoderLayer *layer, ModelConfig *modelConfig) {
    Tensor *Q = multiply(input, layer->W_Q), *K = multiply(input, layer->W_K), *V = multiply(input, layer->W_V);

    Tensor **heads = multiHeadAttention(Q, K, V, modelConfig);
    tensorFree(Q); tensorFree(K); tensorFree(V);

    Tensor *concatenatedHeads = tensorConcat(heads, modelConfig);
    for (int h = 0; h < modelConfig->heads; h++) tensorFree(heads[h]);
    free(heads);

    Tensor *attentionOut = multiply(concatenatedHeads, layer->W_O);
    tensorFree(concatenatedHeads);

    Tensor *firstResidualConnection = add(input, attentionOut);
    tensorFree(attentionOut); 

    Tensor *x = layerNormalization(firstResidualConnection);
    tensorFree(firstResidualConnection);

    Tensor *ffnOut = feedForward(x, layer->W1, layer->B1, layer->W2, layer->B2);

    Tensor *y = add(x, ffnOut);
    tensorFree(x); tensorFree(ffnOut);

    Tensor *output = layerNormalization(y);
    tensorFree(y);

    return output;
}