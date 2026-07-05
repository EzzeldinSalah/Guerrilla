#include <stdio.h>
#include <stdlib.h>
#include "../include/tensor.h"
#include "../include/attention.h"
#include "../include/encoder.h"

void mainTest () {
    float weights1[] = {
        1.0f, 2.0f, 3.0f,
        1.0f, 2.0f, 3.0f,
        1.0f, 2.0f, 3.0f
    };


    float weights2[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };

    int rows1 = 3, cols1 = 3, rows2 = 3, cols2 = 3;

    Tensor *matrix1 = tensorCreate(rows1, cols1);
    fill(matrix1, weights1, rows1 * cols1);
    printf("Matrix 1 -> ");
    tensorPrint(matrix1);

    Tensor *matrix2 = tensorCreate(rows2, cols2);
    fill(matrix2, weights2, rows2 * cols2);
    printf("Matrix 2 -> ");
    tensorPrint(matrix2);

    Tensor *matrix3 = add(matrix1, matrix2);
    printf("Matrix 3: Add Matrix 1 + Matrix 2 -> ");
    tensorPrint(matrix3);

    Tensor *matrix4 = multiply(matrix1, matrix2);
    printf("Matrix 4: Multiply Matrix 1 * Matrix 2 -> ");
    tensorPrint(matrix4);

    Tensor *matrix5 = transpose(matrix4);
    printf("Matrix 5: Transpose Matrix 4 -> ");
    tensorPrint(matrix5);
    
    Tensor *matrix6 = scale(matrix5, 0.5f);
    printf("Matrix 6: Scale Matrix 5 -> ");
    tensorPrint(matrix6);

    printf("Matrix 7: Softmax of matrix 1 -> ");
    Tensor *matrix7 = softmax(matrix1);
    tensorPrint(matrix7);

    printf("Matrix 8: Layer Normalization of matrix 1 -> ");
    Tensor *matrix8 = layerNormalization(matrix1);
    tensorPrint(matrix8);

    float query[] = {
        10.0f, 2.0f, 3.0f,
        4.0f, 15.0f, 6.0f,
        7.0f, 80.0f, 9.0f
    };
    float key[] = {
        113.0f, 2.0f, 31.0f,
        414.0f, 5.0f, 16.0f,
        7.0f, 1.0f, 9.5f
    };
    float value[] = {
        111.0f, 2.0f, 312.0f,
        15.0f, 5.0f, 31.0f,
        75.0f, 18.0f, 9.0f
    };

	int rows = 3, cols = 64;
	Tensor *q = tensorCreate(rows, cols);
	for (int i = 0; i < rows * cols; i++) q->data[i] = (float)rand() / RAND_MAX;
	printf("Q:\n");
    tensorPrint(q);
    Tensor *q_ = layerNormalization(q);
    printf("Q after layer normalization - ");
    tensorPrint(q_);

	Tensor *k = tensorCreate(rows, cols);
	for (int i = 0; i < rows * cols; i++) k->data[i] = (float)rand() / RAND_MAX;
    printf("K\n");
    tensorPrint(k);
    Tensor *k_ = layerNormalization(k);
    printf("k after layer normalization - ");
    tensorPrint(k_);

    Tensor *v = tensorCreate(rows, cols);
	for (int i = 0; i < rows * cols; i++) v->data[i] = (float)rand() / RAND_MAX;
    printf("V:\n");
    tensorPrint(v);
    Tensor *v_ = layerNormalization(v);
    printf("V after layer normalization - ");
    tensorPrint(v_);
	
	extern ModelConfig modelConfig;
    Tensor **heads = multiHeadAttention(q_, k_, v_, &modelConfig);
    printf("Attention Scores:\n");
    
    for (int h = 0; h < modelConfig.heads; h++) {
        printf("Head %d:\n", h + 1);
        tensorPrint(heads[h]);        
    }

    Tensor *concatenatedTensor = tensorConcat(heads, &modelConfig);
    printf("Concatenated Heads -> ");
    tensorPrint(concatenatedTensor);

    for (int h = 0; h < modelConfig.heads; h++) tensorFree(heads[h]);
    free(heads);


    EncoderLayer layer;
    layer.W_Q = tensorCreate(D_MODEL, D_MODEL);
    layer.W_K = tensorCreate(D_MODEL, D_MODEL);
    layer.W_V = tensorCreate(D_MODEL, D_MODEL);
    layer.W_O = tensorCreate(D_MODEL, D_MODEL);
    layer.W1  = tensorCreate(D_MODEL, D_MODEL * 4);
    layer.W2  = tensorCreate(D_MODEL * 4, D_MODEL);
    layer.B1  = tensorCreate(1, D_MODEL * 4);
    layer.B2  = tensorCreate(1, D_MODEL);

    for (int i = 0; i < D_MODEL * D_MODEL; i++) {
        layer.W_Q->data[i] = (float)rand() / RAND_MAX;
        layer.W_K->data[i] = (float)rand() / RAND_MAX;
        layer.W_V->data[i] = (float)rand() / RAND_MAX;
        layer.W_O->data[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < D_MODEL * D_MODEL * modelConfig.heads; i++) layer.W1->data[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < D_MODEL * D_MODEL * modelConfig.heads; i++) layer.W2->data[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < D_MODEL * modelConfig.heads; i++) layer.B1->data[i] = 0.0f;
    for (int i = 0; i < D_MODEL; i++) layer.B2->data[i] = 0.0f;

    Tensor *encoderOut = encoderLayerForward(q_, &layer, &modelConfig);
    printf("Encoder output ->\n");
    tensorPrint(encoderOut);

    tensorFree(encoderOut);
    tensorFree(layer.W_Q); tensorFree(layer.W_K); tensorFree(layer.W_V); tensorFree(layer.W_O);
    tensorFree(layer.W1); tensorFree(layer.W2); tensorFree(layer.B1); tensorFree(layer.B2);

    tensorFree(matrix1); tensorFree(matrix2); tensorFree(matrix3); tensorFree(matrix4);
    tensorFree(matrix5); tensorFree(matrix6); tensorFree(matrix7); tensorFree(matrix8);
    tensorFree(q); tensorFree(k); tensorFree(v);
    tensorFree(q_); tensorFree(k_); tensorFree(v_);
    tensorFree(concatenatedTensor);
}