#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/attention.h"
#include "../include/forward.h"

Tensor *encoderLayerForward (Tensor *input, EncoderLayer *layer, ModelConfig *modelConfig) {
    Tensor *Q = multiply(input, layer->W_Q), *K = multiply(input, layer->W_K), *V = multiply(input, layer->W_V);

    Tensor **heads = multiHeadAttention(Q, K, V, modelConfig);
    tensorFree(Q), tensorFree(K), tensorFree(V);

    Tensor *concatenatedHeads = tensorConcat(heads, modelConfig);
    for (int h = 0; h < modelConfig->heads; h++) tensorFree(heads[h]);
    free(heads);

    Tensor *attentionOut = multiply(concatenatedHeads, layer->W_O);
    tensorFree(concatenatedHeads);

    Tensor *firstResidualConnection = add(input, attentionOut);
    tensorFree(attentionOut); 

    Tensor *x = layerNormalization(firstResidualConnection);
    tensorFree(firstResidualConnection);

    Tensor *theta = multiply(x, layer->W1);
    Tensor *h = add(theta, layer->B1);
    tensorFree(theta);

    Tensor *leakyH = leakyRelu(h, 0.01);
    tensorFree(h);

    Tensor *beta = multiply(leakyH, layer->W2);
    tensorFree(leakyH);

    Tensor *FFN = add(beta, layer->B2);
    tensorFree(beta);

    Tensor *y = add(x, FFN);
    tensorFree(x), tensorFree(FFN);

    Tensor *output = layerNormalization(y);
    tensorFree(y);

    return output;
}