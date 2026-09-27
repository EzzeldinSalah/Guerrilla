#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#if !defined(FORCE_SCALAR) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif
#include "tensor.h"

Tensor *tensorCreate (int rows, int cols) {
	Tensor *newMatrix = malloc(sizeof(Tensor));
	if (newMatrix == NULL) return NULL;

	newMatrix->rows = rows, newMatrix->cols = cols;
	newMatrix->data = malloc(rows * cols * sizeof(float));
	newMatrix->grad = NULL;

	if (newMatrix->data == NULL) {
		free(newMatrix);
		return NULL;
	}

	return newMatrix;
}

void tensorFree (Tensor *matrix) {
	if (!matrix) return;

	free(matrix->data);
	if (matrix->grad) free(matrix->grad);
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

Tensor *addBias (Tensor *matrix, Tensor *bias) {
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
	for (int i = 0; i < matrixDot->rows * matrixDot->cols; i++) matrixDot->data[i] = 0.0f;

	const int TILE = 32;
	for (int ii = 0; ii < matrix1->rows; ii += TILE) {
		for (int kk = 0; kk < matrix1->cols; kk += TILE) {
			for (int jj = 0; jj < matrix2->cols; jj += TILE) {

				int iEnd = (ii + TILE < matrix1->rows) ? ii + TILE : matrix1->rows;
				int kEnd = (kk + TILE < matrix1->cols) ? kk + TILE : matrix1->cols;
				int jEnd = (jj + TILE < matrix2->cols) ? jj + TILE : matrix2->cols;

#if !defined(FORCE_SCALAR) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
				int i = ii;
				for (; i <= iEnd - 4; i += 4) {
					int j = jj;
					for (; j <= jEnd - 4; j += 4) {
						float32x4_t c0 = vld1q_f32(&matrixDot->data[(i + 0) * matrixDot->cols + j]);
						float32x4_t c1 = vld1q_f32(&matrixDot->data[(i + 1) * matrixDot->cols + j]);
						float32x4_t c2 = vld1q_f32(&matrixDot->data[(i + 2) * matrixDot->cols + j]);
						float32x4_t c3 = vld1q_f32(&matrixDot->data[(i + 3) * matrixDot->cols + j]);

						for (int k = kk; k < kEnd; k++) {
							float32x4_t vb = vld1q_f32(&matrix2->data[k * matrix2->cols + j]);
							c0 = vfmaq_n_f32(c0, vb, matrix1->data[(i + 0) * matrix1->cols + k]);
							c1 = vfmaq_n_f32(c1, vb, matrix1->data[(i + 1) * matrix1->cols + k]);
							c2 = vfmaq_n_f32(c2, vb, matrix1->data[(i + 2) * matrix1->cols + k]);
							c3 = vfmaq_n_f32(c3, vb, matrix1->data[(i + 3) * matrix1->cols + k]);
						}

						vst1q_f32(&matrixDot->data[(i + 0) * matrixDot->cols + j], c0);
						vst1q_f32(&matrixDot->data[(i + 1) * matrixDot->cols + j], c1);
						vst1q_f32(&matrixDot->data[(i + 2) * matrixDot->cols + j], c2);
						vst1q_f32(&matrixDot->data[(i + 3) * matrixDot->cols + j], c3);
					}
					// Remainder loop for j — not dead code. Only skipped when jEnd - jj is a multiple of 4.
					for (; j < jEnd; j++) {
						for (int r_idx = 0; r_idx < 4; r_idx++) {
							float sum = 0.0f;
							for (int k = kk; k < kEnd; k++)
								sum += matrix1->data[(i + r_idx) * matrix1->cols + k] * matrix2->data[k * matrix2->cols + j];
							matrixDot->data[(i + r_idx) * matrixDot->cols + j] += sum;
						}
					}
				}
				// Remainder loop for i — not dead code. Only skipped when iEnd - ii is a multiple of 4.
				for (; i < iEnd; i++) {
					for (int k = kk; k < kEnd; k++) {
						float r = matrix1->data[i * matrix1->cols + k];
						for (int j = jj; j < jEnd; j++)
							matrixDot->data[i * matrixDot->cols + j] += r * matrix2->data[k * matrix2->cols + j];
					}
				}
#else
				for (int i = ii; i < iEnd; i++) {
					for (int k = kk; k < kEnd; k++) {
						float r = matrix1->data[i * matrix1->cols + k];
						for (int j = jj; j < jEnd; j++)
							matrixDot->data[i * matrixDot->cols + j] += r * matrix2->data[k * matrix2->cols + j];
					}
				}
#endif

			}
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
	
	for (int i = 0; i < matrix->rows; i++) {
#if !defined(FORCE_SCALAR) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
		float maxVal = matrix->data[i * matrix->cols];
		float32x4_t vmax = vdupq_n_f32(maxVal);
		int j = 0;
		for (; j <= matrix->cols - 4; j += 4) {
			float32x4_t vx = vld1q_f32(&matrix->data[i * matrix->cols + j]);
			vmax = vmaxq_f32(vmax, vx);
		}
		maxVal = vmaxvq_f32(vmax);
		// Remainder loop — not dead code. Only skipped when matrix->cols is a multiple of 4.
		for (; j < matrix->cols; j++)
			if (matrix->data[i * matrix->cols + j] > maxVal) maxVal = matrix->data[i * matrix->cols + j];

		float dominator = 0.0f;
		for (j = 0; j < matrix->cols; j++) {
			float e = (float)exp(matrix->data[i * matrix->cols + j] - maxVal);
			activated->data[i * activated->cols + j] = e;
			dominator += e;
		}

		float invDominator = 1.0f / dominator;
		float32x4_t vinv = vdupq_n_f32(invDominator);
		j = 0;
		for (; j <= matrix->cols - 4; j += 4) {
			float32x4_t ve = vld1q_f32(&activated->data[i * activated->cols + j]);
			vst1q_f32(&activated->data[i * activated->cols + j], vmulq_f32(ve, vinv));
		}
		for (; j < matrix->cols; j++)
			activated->data[i * activated->cols + j] *= invDominator;
#else
		float maxVal = matrix->data[i * matrix->cols];
		for (int j = 0; j < matrix->cols; j++)
			if (matrix->data[i * matrix->cols + j] > maxVal) maxVal = matrix->data[i * matrix->cols + j];
		
		float dominator = 0;
		for (int j = 0; j < matrix->cols; j++)
			dominator += exp(matrix->data[i * matrix->cols + j] - maxVal);

		for (int j = 0; j < matrix->cols; j++)
			activated->data[i * activated->cols + j] = exp(matrix->data[i * matrix->cols + j] - maxVal) / dominator;
#endif
	}
	
	return activated;
}

Tensor *leakyRelu (Tensor *matrix, float alpha) {
	Tensor *activated = tensorCreate(matrix->rows, matrix->cols);

	for (int i = 0; i < matrix->rows; i++) {
#if !defined(FORCE_SCALAR) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
		int j = 0;
		for (; j <= matrix->cols - 4; j += 4) {
			float32x4_t vx = vld1q_f32(&matrix->data[i * matrix->cols + j]);
			float32x4_t vscaled = vmulq_n_f32(vx, alpha);
			float32x4_t vout = vmaxq_f32(vx, vscaled);
			vst1q_f32(&activated->data[i * activated->cols + j], vout);
		}
		/* Remainder loop (not dead code): only skipped when the dimension is a multiple of 4 
		(true for current D_MODEL/d_ff, not guaranteed generally) */
		for (; j < matrix->cols; j++) {
			float value = matrix->data[i * matrix->cols + j];
			activated->data[i * activated->cols + j] = value > 0 ? value : alpha * value;
		}
#else
		for (int j = 0; j < matrix->cols; j++) {
			float value = matrix->data[i * matrix->cols + j];
			activated->data[i * activated->cols + j] = value > 0 ? value : alpha * value;
		}
#endif
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

void tensorRequiresGrad (Tensor *tensor) {
	if (!tensor->grad)
		tensor->grad = calloc(tensor->rows * tensor->cols, sizeof(float));
}