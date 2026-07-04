#pragma once

typedef struct {
    int rows;
    int cols;
    float *data;
} Tensor;

Tensor *tensorCreate (int rows, int cols);
void    tensorFree (Tensor *m);
void    fill (Tensor *m, float *weights, int len);
Tensor *add (Tensor *matrix1, Tensor *matrix2);
Tensor *addBias(Tensor *matrix, Tensor *bias);
Tensor *multiply (Tensor *matrix1, Tensor *matrix2);
Tensor *transpose (Tensor *matrix);
Tensor *scale (Tensor *matrix, float scale);
Tensor *softmax (Tensor *matrix);
Tensor *leakyRelu (Tensor *matrix, float alpha);
Tensor *relu (Tensor *matrix);
Tensor *layerNormalization (Tensor *matrix);
void tensorPrint (Tensor *matrix);
