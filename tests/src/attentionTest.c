#include <stdlib.h>
#include <stdio.h>

#include "../../include/tensor.h"
#include "../../include/attention.h"
#include "../include/testUtils.h"

extern ModelConfig modelConfig;

void attentionTest() {
    Tensor *Q = tensorCreate(3, D_MODEL),  *K = tensorCreate(3, D_MODEL), *V = tensorCreate(3, D_MODEL);
    randomFill(Q), randomFill(K), randomFill(V);

    Tensor **heads = multiHeadAttention(Q, K, V, &modelConfig);
    for (int i = 0; i < modelConfig.heads; i++) {
        printf("Head %d\n", i + 1);
        tensorPrint(heads[i]);
    }

    Tensor *concat = tensorConcat(heads, &modelConfig);
    printf("Concatenated Heads:\n");
    tensorPrint(concat);

    for (int i = 0; i < modelConfig.heads; i++) tensorFree(heads[i]);
    free(heads);

    tensorFree(Q), tensorFree(K), tensorFree(V), tensorFree(concat);
}