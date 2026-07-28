#include <math.h>
#include <stdlib.h>
#include "optimizer.h"

static AdamState adamStateCreate (Tensor *tensor) {
    AdamState state;
    state.m = tensorCreate(tensor->rows, tensor->cols);
    state.v = tensorCreate(tensor->rows, tensor->cols);

    int totalSize = tensor->rows * tensor->cols;
    for (int i = 0; i < totalSize; i++)
        state.m->data[i] = 0.0f, state.v->data[i] = 0.0f;

    return state;
}

static void adamStateFree (AdamState *state) {
    if (!state) return;

    tensorFree(state->m), tensorFree(state->v);
    state->m = NULL, state->v = NULL;
}

void zeroGrad (Tensor *tensor) {
    if (!tensor || !tensor->grad) return;

    int totalSize = tensor->rows * tensor->cols;
    for (int i = 0; i < totalSize; i++)
        tensor->grad[i] = 0.0f;
}

void sgd (Tensor *tensor, float learningRate) {
    if (!tensor || !tensor->grad) return;

    int totalSize = tensor->rows * tensor->cols;
    for (int i = 0; i < totalSize; i++)
        tensor->data[i] -= learningRate * tensor->grad[i];
}

void zeroTransformerGrad (Transformer *transformer, ModelConfig *modelConfig) {
    if (!transformer || !modelConfig) return;

    for (int i = 0; i < modelConfig->layers; i++) {
        zeroGrad(transformer->layers[i].W_Q), zeroGrad(transformer->layers[i].W_K);
        zeroGrad(transformer->layers[i].W_V), zeroGrad(transformer->layers[i].W_O);
        zeroGrad(transformer->layers[i].W1), zeroGrad(transformer->layers[i].W2);
        zeroGrad(transformer->layers[i].B1), zeroGrad(transformer->layers[i].B2);
    }

    zeroGrad(transformer->classW), zeroGrad(transformer->classB);
}

void sgdTransformer (Transformer *transformer, ModelConfig *modelConfig, float learningRate) {
    if (!transformer || !modelConfig) return;

    for (int i = 0; i < modelConfig->layers; i++) {
        sgd(transformer->layers[i].W_Q, learningRate), sgd(transformer->layers[i].W_K, learningRate);
        sgd(transformer->layers[i].W_V, learningRate), sgd(transformer->layers[i].W_O, learningRate);
        sgd(transformer->layers[i].W1, learningRate), sgd(transformer->layers[i].W2, learningRate);
        sgd(transformer->layers[i].B1, learningRate), sgd(transformer->layers[i].B2, learningRate);
    }

    sgd(transformer->classW, learningRate), sgd(transformer->classB, learningRate);
}

AdamOptimizer *adamCreate (Transformer *transformer, ModelConfig *modelConfig, float learningRate) {
    if (!transformer || !modelConfig) return NULL;

    AdamOptimizer *optimizer = malloc(sizeof(AdamOptimizer));
    if (!optimizer) return NULL;

    optimizer->layers = malloc(sizeof(AdamLayerState) * modelConfig->layers);
    if (!optimizer->layers) {
        free(optimizer);
        return NULL;
    }

    optimizer->learningRate = learningRate;
    optimizer->beta1 = 0.9f;
    optimizer->beta2 = 0.999f;
    optimizer->epsilon = 1e-8f;
    optimizer->timestep = 0;

    for (int i = 0; i < modelConfig->layers; i++) {
        optimizer->layers[i].W_Q = adamStateCreate(transformer->layers[i].W_Q);
        optimizer->layers[i].W_K = adamStateCreate(transformer->layers[i].W_K);
        optimizer->layers[i].W_V = adamStateCreate(transformer->layers[i].W_V);
        optimizer->layers[i].W_O = adamStateCreate(transformer->layers[i].W_O);
        optimizer->layers[i].W1 = adamStateCreate(transformer->layers[i].W1);
        optimizer->layers[i].W2 = adamStateCreate(transformer->layers[i].W2);
        optimizer->layers[i].B1 = adamStateCreate(transformer->layers[i].B1);
        optimizer->layers[i].B2 = adamStateCreate(transformer->layers[i].B2);
    }

    optimizer->classW = adamStateCreate(transformer->classW);
    optimizer->classB = adamStateCreate(transformer->classB);

    return optimizer;
}

void adamFree (AdamOptimizer *optimizer, ModelConfig *modelConfig) {
    if (!optimizer) return;

    if (optimizer->layers) {
        for (int i = 0; i < modelConfig->layers; i++) {
            adamStateFree(&optimizer->layers[i].W_Q), adamStateFree(&optimizer->layers[i].W_K);
            adamStateFree(&optimizer->layers[i].W_V), adamStateFree(&optimizer->layers[i].W_O);
            adamStateFree(&optimizer->layers[i].W1), adamStateFree(&optimizer->layers[i].W2);
            adamStateFree(&optimizer->layers[i].B1), adamStateFree(&optimizer->layers[i].B2);
        }
        free(optimizer->layers);
    }

    adamStateFree(&optimizer->classW), adamStateFree(&optimizer->classB);
    free(optimizer);
}

void adam (Tensor *tensor, AdamState *state, float learningRate, float beta1, float beta2, float epsilon, int timestep) {
    if (!tensor || !tensor->grad || !state || !state->m || !state->v) return;

    float beta1Correction = 1.0f - powf(beta1, (float)timestep);
    float beta2Correction = 1.0f - powf(beta2, (float)timestep);

    int totalSize = tensor->rows * tensor->cols;
    for (int i = 0; i < totalSize; i++) {
        float grad = tensor->grad[i];

        state->m->data[i] = beta1 * state->m->data[i] + (1.0f - beta1) * grad;
        state->v->data[i] = beta2 * state->v->data[i] + (1.0f - beta2) * grad * grad;

        float mHat = state->m->data[i] / beta1Correction, vHat = state->v->data[i] / beta2Correction;

        tensor->data[i] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
    }
}

void adamTransformer (Transformer *transformer, ModelConfig *modelConfig, AdamOptimizer *optimizer) {
    if (!transformer || !modelConfig || !optimizer) return;

    optimizer->timestep++;

    for (int i = 0; i < modelConfig->layers; i++) {
        adam(transformer->layers[i].W_Q, &optimizer->layers[i].W_Q, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
        adam(transformer->layers[i].W_K, &optimizer->layers[i].W_K, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
        adam(transformer->layers[i].W_V, &optimizer->layers[i].W_V, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
        adam(transformer->layers[i].W_O, &optimizer->layers[i].W_O, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
        adam(transformer->layers[i].W1, &optimizer->layers[i].W1, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
        adam(transformer->layers[i].W2, &optimizer->layers[i].W2, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
        adam(transformer->layers[i].B1, &optimizer->layers[i].B1, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
        adam(transformer->layers[i].B2, &optimizer->layers[i].B2, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
    }

    adam(transformer->classW, &optimizer->classW, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
    adam(transformer->classB, &optimizer->classB, optimizer->learningRate, optimizer->beta1, optimizer->beta2, optimizer->epsilon, optimizer->timestep);
}
