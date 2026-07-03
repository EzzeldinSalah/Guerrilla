#include <math.h>
#include<stdlib.h>
#include<stdio.h>
#include "../include/tensor.h"
#include "../include/attention.h"

ModelConfig modelConfig = {
    .seqLen = SEQ_LEN,
    .dModel = D_MODEL,
    .heads  = NUM_HEADS,
    .dk = D_MODEL / NUM_HEADS, 
    .layers = NUM_LAYERS
};

Tensor* tensorSlice (Tensor *tensor, int colStart, int colEnd) {
    int numCols = colEnd - colStart;
    Tensor *slice = tensorCreate(tensor->rows, numCols);

    for (int i = 0; i < tensor->rows; i++) {
        for (int j = colStart; j < colEnd; j++) {
            slice->data[i * numCols + (j - colStart)] = tensor->data[i * tensor->cols + j];
        }
    }
    return slice;
}

Tensor *attention (Tensor *query, Tensor *key, Tensor *value, int dk) {
    Tensor *kt = transpose(key);
    Tensor *E = multiply(query, kt);
    tensorFree(kt);
    
    Tensor *scaled = scale(E, 1.0f / sqrt(dk));
    tensorFree(E);

    Tensor *softed = softmax(scaled);
    tensorFree(scaled);
    
    Tensor *scores = multiply(softed, value);
    tensorFree(softed);

    return scores;
}

Tensor **multiHeadAttention (Tensor *query, Tensor *key, Tensor *value, ModelConfig *modelConfig) {
    Tensor **heads = malloc(sizeof(Tensor*) * modelConfig->heads);
    int dHead = modelConfig->dModel / modelConfig->heads;

    for (int h = 0; h < modelConfig->heads; h++) {
        int colStart = h * dHead, colEnd = colStart + dHead;
        Tensor *q = tensorSlice(query, colStart, colEnd);
        Tensor *k = tensorSlice(key, colStart, colEnd);
        Tensor *v = tensorSlice(value, colStart, colEnd);
        heads[h] = attention(q, k, v, modelConfig->dk);
        tensorFree(q); tensorFree(k); tensorFree(v);
    }

    return heads;
}


Tensor *tensorConcat (Tensor **heads, ModelConfig *modelConfig) {
    if (!heads[0]) {
        for (int h = 0; h < modelConfig->heads; h++)
            tensorFree(heads[h]);
        free(heads);
        return NULL;
    }
    
    Tensor *result = tensorCreate(heads[0]->rows, heads[0]->cols * modelConfig->heads);

    for (int h = 0; h < modelConfig->heads; h++) {
        if (!heads[h]) {
            printf("WARNING: head number %d is a NULL, the program will continue and skip concatenating it", h + 1);
            continue;
        }

        int hCols = heads[h]->cols;
        for (int i = 0; i < result->rows; i++)
		    for (int j = 0; j < hCols; j++)
                result->data[i * result->cols +  h * heads[h]->cols + j] = heads[h]->data[i * heads[h]->cols + j];
    }                                                                                                                          

    return result;
}