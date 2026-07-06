#include "../../include/tensor.h"
#include "../include/testUtils.h"

void tensorTest() {
    float weights1[] = {
        1,2,3,
        1,2,3,
        1,2,3
    };

    float weights2[] = {
        1,2,3,
        4,5,6,
        7,8,9
    };

    Tensor *A = tensorCreate(3,3), *B = tensorCreate(3,3);
    
    fill(A, weights1, 9), fill(B, weights2, 9);
    tensorPrint(A), tensorPrint(B);

    Tensor *C = add(A, B);
    tensorPrint(C);

    Tensor *D = multiply(A, B);
    tensorPrint(D);

    Tensor *E = transpose(D);
    tensorPrint(E);

    Tensor *F = scale(E, 0.5f);
    tensorPrint(F);

    Tensor *G = softmax(A);
    tensorPrint(G);

    Tensor *H = layerNormalization(A);
    tensorPrint(H);

    tensorFree(A), tensorFree(B), tensorFree(C), tensorFree(D),
    tensorFree(E), tensorFree(F), tensorFree(G), tensorFree(H);
}