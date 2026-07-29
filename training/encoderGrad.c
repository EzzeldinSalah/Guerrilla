#include <stdlib.h>
#include "tensor.h"
#include "tensorGrad.h"
#include "attention.h"
#include "attentionGrad.h"
#include "encoderGrad.h"

static void tensorZeroData (Tensor *tensor) {
    int totalSize = tensor->rows * tensor->cols;
    for (int i = 0; i < totalSize; i++) tensor->data[i] = 0.0f;
}

static Tensor *tensorCopyData (Tensor *tensor) {
    Tensor *copy = tensorCreate(tensor->rows, tensor->cols);

    int totalSize = tensor->rows * tensor->cols;
    for (int i = 0; i < totalSize; i++)
        copy->data[i] = tensor->data[i];

    return copy;
}

static Tensor *tensorCopyGrad (Tensor *tensor) {
    Tensor *copy = tensorCreate(tensor->rows, tensor->cols);

    int totalSize = tensor->rows * tensor->cols;
    for (int i = 0; i < totalSize; i++)
        copy->data[i] = tensor->grad ? tensor->grad[i] : 0.0f;

    return copy;
}

static void multiplyBackwardAData (Tensor *A, Tensor *B, Tensor *dC, Tensor *dA) {
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++) {
            float tempSum = 0.0f;
            for (int k = 0; k < dC->cols; k++)
                tempSum += dC->data[i * dC->cols + k] * B->data[j * B->cols + k];

            dA->data[i * dA->cols + j] += tempSum;
        }
}

static void tensorSliceBackward (Tensor *tensor, Tensor *sliceGrad, int colStart) {
    for (int i = 0; i < sliceGrad->rows; i++)
        for (int j = 0; j < sliceGrad->cols; j++)
            tensor->data[i * tensor->cols + colStart + j] += sliceGrad->data[i * sliceGrad->cols + j];
}

void classificationHeadBackward (Tensor *pooled, Tensor *classW, Tensor *classB, Tensor *dLogits) {
    addBiasBackward(classB, dLogits);   
    multiplyBackwardA(pooled, classW, dLogits), multiplyBackwardB(pooled, classW, dLogits);
}

Tensor *meanPoolBackward (Tensor *input, Tensor *dPooled) {
    Tensor *dInput = tensorCreate(input->rows, input->cols);
    tensorZeroData(dInput);

    for (int i = 0; i < input->rows; i++)
        for (int j = 0; j < input->cols; j++)
            dInput->data[i * input->cols + j] += dPooled->data[j] / (float)input->rows;

    return dInput;
}

Tensor *encoderLayerBackward (Tensor *input, EncoderLayer *layer, Tensor *dOutput, ModelConfig *modelConfig) {
    Tensor *Q = multiply(input, layer->W_Q), *K = multiply(input, layer->W_K), *V = multiply(input, layer->W_V);

    Tensor **heads = multiHeadAttention(Q, K, V, modelConfig);
    Tensor *concatenatedHeads = tensorConcat(heads, modelConfig);
    for (int h = 0; h < modelConfig->heads; h++) tensorFree(heads[h]);
    free(heads);

    Tensor *attentionOut = multiply(concatenatedHeads, layer->W_O);
    Tensor *firstResidualConnection = add(input, attentionOut);
    Tensor *x = layerNormalization(firstResidualConnection);

    Tensor *theta = multiply(x, layer->W1);
    Tensor *hidden = addBias(theta, layer->B1);
    Tensor *leakyHidden = leakyRelu(hidden, 0.01f);
    Tensor *beta = multiply(leakyHidden, layer->W2);
    Tensor *ffnOut = addBias(beta, layer->B2);

    Tensor *secondResidualConnection = add(x, ffnOut);
    layerNormBackward(secondResidualConnection, dOutput);

    Tensor *dSecondResidualConnection = tensorCopyGrad(secondResidualConnection);
    Tensor *dX = tensorCopyData(dSecondResidualConnection);
    Tensor *dFfnOut = tensorCopyData(dSecondResidualConnection);

    addBiasBackward(layer->B2, dFfnOut);
    multiplyBackwardB(leakyHidden, layer->W2, dFfnOut);

    Tensor *dLeakyHidden = tensorCreate(leakyHidden->rows, leakyHidden->cols);
    tensorZeroData(dLeakyHidden);
    multiplyBackwardAData(leakyHidden, layer->W2, dFfnOut, dLeakyHidden);

    leakyReluBackward(hidden, dLeakyHidden, 0.01f);
    Tensor *dHidden = tensorCopyGrad(hidden);

    addBiasBackward(layer->B1, dHidden);
    multiplyBackwardB(x, layer->W1, dHidden);
    multiplyBackwardAData(x, layer->W1, dHidden, dX);

    layerNormBackward(firstResidualConnection, dX);
    Tensor *dFirstResidualConnection = tensorCopyGrad(firstResidualConnection);
    Tensor *dInput = tensorCopyData(dFirstResidualConnection);
    Tensor *dAttentionOut = tensorCopyData(dFirstResidualConnection);

    multiplyBackwardB(concatenatedHeads, layer->W_O, dAttentionOut);

    Tensor *dConcatenatedHeads = tensorCreate(concatenatedHeads->rows, concatenatedHeads->cols);
    tensorZeroData(dConcatenatedHeads);
    multiplyBackwardAData(concatenatedHeads, layer->W_O, dAttentionOut, dConcatenatedHeads);

    Tensor *dQ = tensorCreate(Q->rows, Q->cols), *dK = tensorCreate(K->rows, K->cols), *dV = tensorCreate(V->rows, V->cols);
    tensorZeroData(dQ), tensorZeroData(dK), tensorZeroData(dV);

    int dHead = modelConfig->dModel / modelConfig->heads;
    for (int h = 0; h < modelConfig->heads; h++) {
        int colStart = h * dHead, colEnd = colStart + dHead;

        Tensor *q = tensorSlice(Q, colStart, colEnd), *k = tensorSlice(K, colStart, colEnd), *v = tensorSlice(V, colStart, colEnd);
        Tensor *dHeadOut = tensorSlice(dConcatenatedHeads, colStart, colEnd);
        Tensor *dQHead = tensorCreate(q->rows, q->cols), *dKHead = tensorCreate(k->rows, k->cols), *dVHead = tensorCreate(v->rows, v->cols);
        tensorZeroData(dQHead), tensorZeroData(dKHead), tensorZeroData(dVHead);

        singleHeadAttentionBackward(q, k, v, dHeadOut, dQHead, dKHead, dVHead, modelConfig->dk);
        tensorSliceBackward(dQ, dQHead, colStart), tensorSliceBackward(dK, dKHead, colStart), tensorSliceBackward(dV, dVHead, colStart);

        tensorFree(q), tensorFree(k), tensorFree(v), tensorFree(dHeadOut);
        tensorFree(dQHead), tensorFree(dKHead), tensorFree(dVHead);
    }

    multiplyBackwardB(input, layer->W_Q, dQ), multiplyBackwardAData(input, layer->W_Q, dQ, dInput);
    multiplyBackwardB(input, layer->W_K, dK), multiplyBackwardAData(input, layer->W_K, dK, dInput);
    multiplyBackwardB(input, layer->W_V, dV), multiplyBackwardAData(input, layer->W_V, dV, dInput);

    tensorFree(Q), tensorFree(K), tensorFree(V), tensorFree(concatenatedHeads), tensorFree(attentionOut);
    tensorFree(firstResidualConnection), tensorFree(x), tensorFree(theta), tensorFree(hidden), tensorFree(leakyHidden);
    tensorFree(beta), tensorFree(ffnOut), tensorFree(secondResidualConnection), tensorFree(dSecondResidualConnection);
    tensorFree(dX), tensorFree(dFfnOut), tensorFree(dLeakyHidden), tensorFree(dHidden), tensorFree(dFirstResidualConnection);
    tensorFree(dAttentionOut), tensorFree(dConcatenatedHeads), tensorFree(dQ), tensorFree(dK), tensorFree(dV);

    return dInput;
}

Tensor *encoderStackBackward (Tensor **layerInputs, Transformer *transformer, int numLayers, Tensor *dOutput, ModelConfig *modelConfig) {
    Tensor *currentGrad = tensorCopyData(dOutput);

    for (int i = numLayers - 1; i >= 0; i--) {
        Tensor *nextGrad = encoderLayerBackward(layerInputs[i], &transformer->layers[i], currentGrad, modelConfig);
        tensorFree(currentGrad);
        currentGrad = nextGrad;
    }

    return currentGrad;
}
