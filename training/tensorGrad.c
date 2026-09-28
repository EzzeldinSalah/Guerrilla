#include <math.h>
#include <stdio.h>
#if !defined(FORCE_SCALAR) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#endif
#include "tensorGrad.h"

void leakyReluBackward (Tensor *A, Tensor *dC, float alpha) {
    if (A->rows != dC->rows || A->cols != dC->cols) {
        printf("leakyReluBackward: shape mismatch (A: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, dC->rows, dC->cols);

        return;
    }

    if (!A->grad) tensorRequiresGrad(A);

    for (int i = 0; i < A->rows; i++) {
#if !defined(FORCE_SCALAR) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
        int j = 0;
        float32x4_t valpha = vdupq_n_f32(alpha);
        for (; j <= A->cols - 4; j += 4) {
            float32x4_t va = vld1q_f32(&A->data[i * A->cols + j]);
            float32x4_t vdc = vld1q_f32(&dC->data[i * dC->cols + j]);
            uint32x4_t mask = vcgtq_f32(va, vdupq_n_f32(0.0f));
            float32x4_t vscaled = vmulq_f32(vdc, valpha);
            float32x4_t vgrad = vbslq_f32(mask, vdc, vscaled);
            float32x4_t vprev = vld1q_f32(&A->grad[i * A->cols + j]);
            vst1q_f32(&A->grad[i * A->cols + j], vaddq_f32(vprev, vgrad));
        }
        // Remainder loop (Only skipped when the dimension is a multiple of 4
        // (true for current D_MODEL/d_ff, but it's not guaranteed generally)
        for (; j < A->cols; j++)
            A->grad[i * A->cols + j] += A->data[i * A->cols + j] > 0 ? dC->data[i * dC->cols + j] : alpha * dC->data[i * dC->cols + j];
#else
        for (int j = 0; j < A->cols; j++)
            A->grad[i * A->cols + j] += A->data[i * A->cols + j] > 0 ? dC->data[i * dC->cols + j] : alpha * dC->data[i * dC->cols + j];
#endif
    }
}

void layerNormBackward (Tensor *x, Tensor *dy) {
    if (!x || !dy) return;
    if (!x->grad) tensorRequiresGrad(x);

    for (int i = 0; i < x->rows; i++) {
#if !defined(FORCE_SCALAR) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
        float32x4_t vsum = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j <= x->cols - 4; j += 4) {
            float32x4_t vx = vld1q_f32(&x->data[i * x->cols + j]);
            vsum = vaddq_f32(vsum, vx);
        }
        float mean = vaddvq_f32(vsum);
        // Remainder loop — not dead code. Only skipped when the dimension is a
        // multiple of 4 (true for current D_MODEL/d_ff, not guaranteed generally).
        for (; j < x->cols; j++)
            mean += x->data[i * x->cols + j];
        mean /= x->cols;

        float32x4_t vmean = vdupq_n_f32(mean);
        float32x4_t vvar = vdupq_n_f32(0.0f);
        j = 0;
        for (; j <= x->cols - 4; j += 4) {
            float32x4_t vx = vld1q_f32(&x->data[i * x->cols + j]);
            float32x4_t vdiff = vsubq_f32(vx, vmean);
            vvar = vfmaq_f32(vvar, vdiff, vdiff);
        }
        float variance = vaddvq_f32(vvar);
        for (; j < x->cols; j++) {
            float diff = x->data[i * x->cols + j] - mean;
            variance += diff * diff;
        }
        variance /= x->cols;

        float invStd = 1.0f / sqrtf(variance + 1e-5f);
        float32x4_t vinvStd = vdupq_n_f32(invStd);

        float32x4_t vmeanDy = vdupq_n_f32(0.0f);
        float32x4_t vmeanDyY = vdupq_n_f32(0.0f);
        j = 0;
        for (; j <= x->cols - 4; j += 4) {
            float32x4_t vx = vld1q_f32(&x->data[i * x->cols + j]);
            float32x4_t vdy = vld1q_f32(&dy->data[i * x->cols + j]);
            float32x4_t vy = vmulq_f32(vsubq_f32(vx, vmean), vinvStd);
            vmeanDy = vaddq_f32(vmeanDy, vdy);
            vmeanDyY = vfmaq_f32(vmeanDyY, vdy, vy);
        }
        float meanDy = vaddvq_f32(vmeanDy);
        float meanDyY = vaddvq_f32(vmeanDyY);
        for (; j < x->cols; j++) {
            float xi = x->data[i * x->cols + j], yi = (xi - mean) * invStd;
            float dyi = dy->data[i * x->cols + j];
            meanDy += dyi, meanDyY += dyi * yi;
        }
        meanDy /= x->cols, meanDyY /= x->cols;

        float32x4_t vmDy = vdupq_n_f32(meanDy);
        float32x4_t vmDyY = vdupq_n_f32(meanDyY);
        j = 0;
        for (; j <= x->cols - 4; j += 4) {
            float32x4_t vx = vld1q_f32(&x->data[i * x->cols + j]);
            float32x4_t vdy = vld1q_f32(&dy->data[i * x->cols + j]);
            float32x4_t vy = vmulq_f32(vsubq_f32(vx, vmean), vinvStd);
            float32x4_t vterm = vsubq_f32(vdy, vmDy);
            vterm = vmlsq_f32(vterm, vy, vmDyY);
            float32x4_t vgrad = vmulq_f32(vterm, vinvStd);
            float32x4_t vcurr = vld1q_f32(&x->grad[i * x->cols + j]);
            vst1q_f32(&x->grad[i * x->cols + j], vaddq_f32(vcurr, vgrad));
        }
        for (; j < x->cols; j++) {
            float xi = x->data[i * x->cols + j], yi = (xi - mean) * invStd;
            float dyi = dy->data[i * x->cols + j];
            float grad_val = (dyi - meanDy - yi * meanDyY) * invStd;
            x->grad[i * x->cols + j] += grad_val;
        }
#else
        float mean = 0.0f, variance = 0.0f;
        for (int j = 0; j < x->cols; j++)
            mean += x->data[i * x->cols + j];
        mean /= x->cols;

        for (int j = 0; j < x->cols; j++) {
            float diff = x->data[i * x->cols + j] - mean;
            variance += diff * diff;
        }
        variance /= x->cols;

        float invStd = 1.0f / sqrtf(variance + 1e-5f);

        float meanDy = 0.0f, meanDyY = 0.0f;
        for (int j = 0; j < x->cols; j++) {
            float xi = x->data[i * x->cols + j], yi = (xi - mean) * invStd;
            float dyi = dy->data[i * x->cols + j];

            meanDy += dyi, meanDyY += dyi * yi;
        }
        meanDy /= x->cols, meanDyY /= x->cols;

        for (int j = 0; j < x->cols; j++) {
            float xi = x->data[i * x->cols + j], yi = (xi - mean) * invStd;
            float dyi = dy->data[i * x->cols + j];

            float grad_val = (dyi - meanDy - yi * meanDyY) * invStd;
            x->grad[i * x->cols + j] += grad_val;
        }
#endif
    }
}

void softmaxBackward (Tensor *scores, Tensor *A, Tensor *dA) {
    if (!scores || !A || !dA) return;

    if (scores->rows != A->rows || scores->cols != A->cols ||
        A->rows != dA->rows || A->cols != dA->cols) {
        printf("softmaxBackward: shape mismatch (scores: %dx%d, A: %dx%d, dA: %dx%d)\n",
               scores->rows, scores->cols, A->rows, A->cols, dA->rows, dA->cols);

        return;
    }
    
    if (!scores->grad) tensorRequiresGrad(scores);


    for (int i = 0; i <  A->rows; i++) {
        int offset = i * A->cols;

        float dot = 0.0f;
        for (int j = 0; j < A->cols; j++)
            dot += A->data[offset + j] * dA->data[offset + j];

        for (int j = 0; j < A->cols; j++) {
            float a = A->data[offset + j], da = dA->data[offset + j];
            scores->grad[offset + j] += a * (da - dot);
        }
    }
}


static void matmulBackwardATile (Tensor *A, Tensor *B, Tensor *dC, float *dest) {
    const int TILE = 32;
    for (int ii = 0; ii < A->rows; ii += TILE) {
        for (int jj = 0; jj < A->cols; jj += TILE) {
            for (int kk = 0; kk < dC->cols; kk += TILE) {

                int iEnd = (ii + TILE < A->rows) ? ii + TILE : A->rows;
                int jEnd = (jj + TILE < A->cols) ? jj + TILE : A->cols;
                int kEnd = (kk + TILE < dC->cols) ? kk + TILE : dC->cols;

#if !defined(FORCE_SCALAR) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                int i = ii;
                for (; i <= iEnd - 4; i += 4) {
                    int j = jj;
                    for (; j <= jEnd - 4; j += 4) {
                        float32x4_t s00 = vdupq_n_f32(0.0f), s01 = vdupq_n_f32(0.0f), s02 = vdupq_n_f32(0.0f), s03 = vdupq_n_f32(0.0f);
                        float32x4_t s10 = vdupq_n_f32(0.0f), s11 = vdupq_n_f32(0.0f), s12 = vdupq_n_f32(0.0f), s13 = vdupq_n_f32(0.0f);
                        float32x4_t s20 = vdupq_n_f32(0.0f), s21 = vdupq_n_f32(0.0f), s22 = vdupq_n_f32(0.0f), s23 = vdupq_n_f32(0.0f);
                        float32x4_t s30 = vdupq_n_f32(0.0f), s31 = vdupq_n_f32(0.0f), s32 = vdupq_n_f32(0.0f), s33 = vdupq_n_f32(0.0f);

                        int k = kk;
                        for (; k <= kEnd - 4; k += 4) {
                            float32x4_t dc0 = vld1q_f32(&dC->data[(i + 0) * dC->cols + k]);
                            float32x4_t dc1 = vld1q_f32(&dC->data[(i + 1) * dC->cols + k]);
                            float32x4_t dc2 = vld1q_f32(&dC->data[(i + 2) * dC->cols + k]);
                            float32x4_t dc3 = vld1q_f32(&dC->data[(i + 3) * dC->cols + k]);

                            float32x4_t b0 = vld1q_f32(&B->data[(j + 0) * B->cols + k]);
                            float32x4_t b1 = vld1q_f32(&B->data[(j + 1) * B->cols + k]);
                            float32x4_t b2 = vld1q_f32(&B->data[(j + 2) * B->cols + k]);
                            float32x4_t b3 = vld1q_f32(&B->data[(j + 3) * B->cols + k]);

                            s00 = vfmaq_f32(s00, dc0, b0); s01 = vfmaq_f32(s01, dc0, b1);
                            s02 = vfmaq_f32(s02, dc0, b2); s03 = vfmaq_f32(s03, dc0, b3);

                            s10 = vfmaq_f32(s10, dc1, b0); s11 = vfmaq_f32(s11, dc1, b1);
                            s12 = vfmaq_f32(s12, dc1, b2); s13 = vfmaq_f32(s13, dc1, b3);

                            s20 = vfmaq_f32(s20, dc2, b0); s21 = vfmaq_f32(s21, dc2, b1);
                            s22 = vfmaq_f32(s22, dc2, b2); s23 = vfmaq_f32(s23, dc2, b3);

                            s30 = vfmaq_f32(s30, dc3, b0); s31 = vfmaq_f32(s31, dc3, b1);
                            s32 = vfmaq_f32(s32, dc3, b2); s33 = vfmaq_f32(s33, dc3, b3);
                        }

                        dest[(i + 0) * A->cols + (j + 0)] += vaddvq_f32(s00);
                        dest[(i + 0) * A->cols + (j + 1)] += vaddvq_f32(s01);
                        dest[(i + 0) * A->cols + (j + 2)] += vaddvq_f32(s02);
                        dest[(i + 0) * A->cols + (j + 3)] += vaddvq_f32(s03);

                        dest[(i + 1) * A->cols + (j + 0)] += vaddvq_f32(s10);
                        dest[(i + 1) * A->cols + (j + 1)] += vaddvq_f32(s11);
                        dest[(i + 1) * A->cols + (j + 2)] += vaddvq_f32(s12);
                        dest[(i + 1) * A->cols + (j + 3)] += vaddvq_f32(s13);

                        dest[(i + 2) * A->cols + (j + 0)] += vaddvq_f32(s20);
                        dest[(i + 2) * A->cols + (j + 1)] += vaddvq_f32(s21);
                        dest[(i + 2) * A->cols + (j + 2)] += vaddvq_f32(s22);
                        dest[(i + 2) * A->cols + (j + 3)] += vaddvq_f32(s23);

                        dest[(i + 3) * A->cols + (j + 0)] += vaddvq_f32(s30);
                        dest[(i + 3) * A->cols + (j + 1)] += vaddvq_f32(s31);
                        dest[(i + 3) * A->cols + (j + 2)] += vaddvq_f32(s32);
                        dest[(i + 3) * A->cols + (j + 3)] += vaddvq_f32(s33);

                        // Remainder loop for k (Only skipped when kEnd - kk % 4 == 0)
                        for (; k < kEnd; k++) {
                            for (int r = 0; r < 4; r++)
                                for (int c = 0; c < 4; c++)
                                    dest[(i + r) * A->cols + (j + c)] += dC->data[(i + r) * dC->cols + k] * B->data[(j + c) * B->cols + k];
                        }
                    }
                    // Remainder loop for j (Only skipped when jEnd - jj % 4 == 0)
                    for (; j < jEnd; j++) {
                        for (int r = 0; r < 4; r++) {
                            float tempSum = 0.0f;
                            for (int k = kk; k < kEnd; k++)
                                tempSum += dC->data[(i + r) * dC->cols + k] * B->data[j * B->cols + k];
                            dest[(i + r) * A->cols + j] += tempSum;
                        }
                    }
                }
                // Remainder loop for i (Only skipped when iEnd - ii % 4 == 0)
                for (; i < iEnd; i++) {
                    for (int j = jj; j < jEnd; j++) {
                        float tempSum = 0.0f;
                        for (int k = kk; k < kEnd; k++)
                            tempSum += dC->data[i * dC->cols + k] * B->data[j * B->cols + k];
                        dest[i * A->cols + j] += tempSum;
                    }
                }
#else
                for (int i = ii; i < iEnd; i++) {
                    for (int j = jj; j < jEnd; j++) {
                        float tempSum = 0.0f;
                        for (int k = kk; k < kEnd; k++)
                            tempSum += dC->data[i * dC->cols + k] * B->data[j * B->cols + k];
                        dest[i * A->cols + j] += tempSum;
                    }
                }
#endif

            }
        }
    }
}

void multiplyBackwardA (Tensor *A, Tensor *B, Tensor *dC) {
    if (A->rows != dC->rows || B->cols != dC->cols || A->cols != B->rows) {
        printf("multiplyBackwardA: shape mismatch (A: %dx%d, B: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols, dC->rows, dC->cols);
        return;
    }

    if (!A->grad) tensorRequiresGrad(A);
    matmulBackwardATile(A, B, dC, A->grad);
}

void multiplyBackwardAData (Tensor *A, Tensor *B, Tensor *dC, Tensor *dA) {
    if (A->rows != dC->rows || B->cols != dC->cols || A->cols != B->rows ||
        A->rows != dA->rows || A->cols != dA->cols) {
        printf("multiplyBackwardAData: shape mismatch (A: %dx%d, B: %dx%d, dC: %dx%d, dA: %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols, dC->rows, dC->cols, dA->rows, dA->cols);
        return;
    }

    matmulBackwardATile(A, B, dC, dA->data);
}

void multiplyBackwardB (Tensor *A, Tensor *B, Tensor *dC) {
    if (A->rows != dC->rows || B->cols != dC->cols || A->cols != B->rows) {
        printf("multiplyBackwardB: shape mismatch (A: %dx%d, B: %dx%d, dC: %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols, dC->rows, dC->cols);
        return;
    }

    if (!B->grad) tensorRequiresGrad(B);

    const int TILE = 32;
    for (int ii = 0; ii < A->rows; ii += TILE) {
        for (int kk = 0; kk < A->cols; kk += TILE) {
            for (int jj = 0; jj < B->cols; jj += TILE) {

                int iEnd = (ii + TILE < A->rows) ? ii + TILE : A->rows;
                int kEnd = (kk + TILE < A->cols) ? kk + TILE : A->cols;
                int jEnd = (jj + TILE < B->cols) ? jj + TILE : B->cols;

#if !defined(FORCE_SCALAR) && defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
                int k = kk;
                for (; k <= kEnd - 4; k += 4) {
                    int j = jj;
                    for (; j <= jEnd - 4; j += 4) {
                        float32x4_t c0 = vld1q_f32(&B->grad[(k + 0) * B->cols + j]);
                        float32x4_t c1 = vld1q_f32(&B->grad[(k + 1) * B->cols + j]);
                        float32x4_t c2 = vld1q_f32(&B->grad[(k + 2) * B->cols + j]);
                        float32x4_t c3 = vld1q_f32(&B->grad[(k + 3) * B->cols + j]);

                        for (int i = ii; i < iEnd; i++) {
                            float32x4_t vdc = vld1q_f32(&dC->data[i * dC->cols + j]);
                            c0 = vfmaq_n_f32(c0, vdc, A->data[i * A->cols + (k + 0)]);
                            c1 = vfmaq_n_f32(c1, vdc, A->data[i * A->cols + (k + 1)]);
                            c2 = vfmaq_n_f32(c2, vdc, A->data[i * A->cols + (k + 2)]);
                            c3 = vfmaq_n_f32(c3, vdc, A->data[i * A->cols + (k + 3)]);
                        }

                        vst1q_f32(&B->grad[(k + 0) * B->cols + j], c0);
                        vst1q_f32(&B->grad[(k + 1) * B->cols + j], c1);
                        vst1q_f32(&B->grad[(k + 2) * B->cols + j], c2);
                        vst1q_f32(&B->grad[(k + 3) * B->cols + j], c3);
                    }
                    // Remainder for j (Only skipped when jEnd - jj is a multiple of 4)
                    for (; j < jEnd; j++) {
                        for (int r_idx = 0; r_idx < 4; r_idx++) {
                            float sum = 0.0f;
                            for (int i = ii; i < iEnd; i++)
                                sum += A->data[i * A->cols + (k + r_idx)] * dC->data[i * dC->cols + j];
                            B->grad[(k + r_idx) * B->cols + j] += sum;
                        }
                    }
                }
                // Remainder for k (Only skipped when kEnd - kk is a multiple of 4)
                for (; k < kEnd; k++) {
                    for (int i = ii; i < iEnd; i++) {
                        float r = A->data[i * A->cols + k];
                        for (int j = jj; j < jEnd; j++)
                            B->grad[k * B->cols + j] += r * dC->data[i * dC->cols + j];
                    }
                }
#else
                for (int i = ii; i < iEnd; i++) {
                    for (int k = kk; k < kEnd; k++) {
                        float r = A->data[i * A->cols + k];
                        for (int j = jj; j < jEnd; j++)
                            B->grad[k * B->cols + j] += r * dC->data[i * dC->cols + j];
                    }
                }
#endif

            }
        }
    }
}

void addBiasBackward (Tensor *bias, Tensor *upstream) {
    if (bias->rows != 1 || bias->cols != upstream->cols) {
        printf("addBiasBackward: shape mismatch (bias: %dx%d, upstream: %dx%d)\n",
               bias->rows, bias->cols, upstream->rows, upstream->cols);
        return;
    }

    if (!bias->grad) tensorRequiresGrad(bias);

    for (int i = 0; i < upstream->rows; i++)
        for (int j = 0; j < upstream->cols; j++)
            bias->grad[j] += upstream->data[i * upstream->cols + j];
}