#include <stdlib.h>
#include "../include/testUtils.h"

void randomFill(Tensor *tensor) {
    for (int i = 0; i < tensor->rows * tensor->cols; i++)
        tensor->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
}