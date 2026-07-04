#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "../include/tensor.h"

Tensor *tensorCreate (int rows, int cols) {
	Tensor *newMatrix = malloc(sizeof(Tensor));
	if (newMatrix == NULL) return NULL;

	newMatrix->rows = rows, newMatrix->cols = cols;
	newMatrix->data = malloc(rows * cols * sizeof(float));

	if (newMatrix->data == NULL) {
		free(newMatrix);
		return NULL;
	}

	return newMatrix;
}

void tensorFree (Tensor *matrix) {
	if (matrix == NULL) return;

	free(matrix->data);
	free(matrix);
}

void fill (Tensor *matrix, float *weights, int totalSize) {
	int totalCapacity = matrix->rows * matrix->cols;

	if (totalCapacity != totalSize) {
		printf("the total size of weights doesn't fit the matrix capacity\n");
		return;
	}

	for (int i = 0; i < matrix->rows; i++)
		for (int j = 0; j < matrix->cols; j++)
			matrix->data[i * matrix->cols + j] = weights[i * matrix->cols + j];
}


Tensor *add (Tensor *matrix1, Tensor *matrix2) {
	if (matrix1->rows != matrix2->rows || matrix1->cols != matrix2->cols) {
		printf("To add two matrices, they must have the exact same dimensions\n");
		return NULL;
	}

	Tensor *MatrixSum = tensorCreate(matrix1->rows, matrix1->cols);

	for (int i = 0; i < matrix1->rows; i++)
		for (int j = 0; j < matrix1->cols; j++)
			MatrixSum->data[i * matrix1->cols + j] = matrix1->data[i * matrix1->cols + j] + matrix2->data[i * matrix1->cols + j];
	
	return MatrixSum;
}

Tensor *addBias(Tensor *matrix, Tensor *bias) {
    if (bias->rows != 1 || matrix->cols != bias->cols) {
        printf("Bias must be shape (1, %d) to match matrix columns !\n", matrix->cols);
        return NULL;
    }

    Tensor *result = tensorCreate(matrix->rows, matrix->cols);

    for (int i = 0; i < matrix->rows; i++) 
        for (int j = 0; j < matrix->cols; j++)
            result->data[i * matrix->cols + j] = matrix->data[i * matrix->cols + j] + bias->data[j];

    return result;
}

Tensor *multiply (Tensor *matrix1, Tensor *matrix2) {
	if (matrix1->cols != matrix2->rows) {
		printf("To multiply two matrices, matrix1->cols has to equal matrix2->rows\n");
		return NULL;
	}

	Tensor *matrixDot = tensorCreate(matrix1->rows, matrix2->cols);
	
	for (int i = 0; i < matrix1->rows; i++) {
		for (int j = 0; j < matrix2->cols; j++) {
			matrixDot->data[i * matrixDot->cols + j] = 0;
			for (int k = 0; k < matrix2->rows; k++)
				matrixDot->data[i * matrixDot->cols + j] +=  matrix1->data[i * matrix1->cols + k] * matrix2->data[k * matrix2->cols + j];
		}
	}

	return matrixDot;
}

Tensor *transpose (Tensor *matrix) {
	Tensor *transposed = tensorCreate(matrix->cols, matrix->rows);

	for (int i = 0; i < transposed->cols; i++)
		for (int j = 0; j < transposed->rows; j++)
			transposed->data[j * transposed->cols + i] = matrix->data[i * matrix->cols + j];

	return transposed;
}

Tensor *scale (Tensor *matrix, float scale) {
	Tensor *scaled = tensorCreate(matrix->rows, matrix->cols);
	
	for (int i = 0; i < matrix->rows; i++)
		for (int j = 0; j < matrix->cols; j++)
			scaled->data[i * scaled->cols + j] = scale * matrix->data[i * matrix->cols + j];

	return scaled;
}

Tensor *softmax (Tensor *matrix) {
	Tensor *activated = tensorCreate(matrix->rows, matrix->cols);
	
	float maxVal;
	for (int i = 0; i < matrix->rows; i++) {
		maxVal = matrix->data[i * matrix->cols];
		for (int j = 0; j < matrix->cols; j++)
			if (matrix->data[i * matrix->cols + j] > maxVal) maxVal = matrix->data[i * matrix->cols + j];
		
		float dominator = 0;
		for (int j = 0; j < matrix->cols; j++)
			dominator += exp(matrix->data[i * matrix->cols + j] - maxVal);

		for (int j = 0; j < matrix->cols; j++)
			activated->data[i * activated->cols + j] = exp(matrix->data[i * matrix->cols + j] - maxVal) / dominator;
	}
	
	return activated;
}

Tensor *leakyRelu (Tensor *matrix, float alpha) {
	Tensor *activated = tensorCreate(matrix->rows, matrix->cols);

	for (int i = 0; i < matrix->rows; i++)
		for (int j = 0; j < matrix->cols; j++) {
			float value = matrix->data[i * matrix->cols + j];
			activated->data[i * activated->cols + j] = value > 0 ? value : alpha * value;
		}

	return activated;
}

Tensor *relu (Tensor *matrix) {
	Tensor *activated = tensorCreate(matrix->rows, matrix->cols);

	for (int i = 0; i < matrix->rows; i++)
		for (int j = 0; j < matrix->cols; j++) 
			activated->data[i * activated->cols + j] = matrix->data[i * matrix->cols + j] > 0 ?
				matrix->data[i * matrix->cols + j] : 0;
		
	return activated;
}

Tensor *layerNormalization (Tensor *matrix) {
	Tensor *normalized = tensorCreate(matrix->rows, matrix->cols);

	for (int i = 0; i < matrix->rows; i++) {
    	float mean = 0, variance = 0;

		for (int j = 0; j < matrix->cols; j++)
			mean += matrix->data[i * matrix->cols + j];
		mean /= matrix->cols;

		for (int j = 0; j < matrix->cols; j++)
			variance += pow(matrix->data[i * matrix->cols + j] - mean, 2);
		variance /= matrix->cols;

		float std = sqrt(variance + 1e-5f); // 1e-5f prevents division by zero, and it's called epsilon

		for (int j = 0; j < matrix->cols; j++)
			normalized->data[i * matrix->cols + j] = (matrix->data[i * matrix->cols + j] - mean) / std;
	}

	return normalized;
}
	
void tensorPrint (Tensor *matrix) {
	printf("Matrix Dimensions: %d x %d\n", matrix->rows, matrix->cols);
	for (int i = 0; i < matrix->rows; i++) {
		for (int j = 0; j < matrix->cols; j++)
			printf("\t%f", matrix->data[i * matrix->cols + j]);
		printf("\n");
	}
}


/* the point of MotherMatrix was to avoid calling matrix_free manually on every matrix, sounds convenient */
/* Cost: holds everything until the end actually makes this harder, because I need to free intermediates as I go
or I'll blow my memory LOL */

// typedef struct {
// 	Matrix **matrices;
// 	size_t capacity;
// 	size_t size;
// } MotherMatrix;

// MotherMatrix *mother_create () {
// 	MotherMatrix *mother = malloc(sizeof(MotherMatrix));
// 	if (mother == NULL) return NULL;

// 	mother->capacity = 10;
// 	mother->size = 0;
// 	mother->matrices = malloc(mother->capacity * sizeof(Matrix*));

// 	if (mother->matrices == NULL) {
// 		free(mother);
// 		return NULL;
// 	}

// 	return mother;
// }