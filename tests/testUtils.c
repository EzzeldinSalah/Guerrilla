#include <stdlib.h>
#include "tests.h"

void randomDataFill(Tensor *tensor) {
    for (int i = 0; i < tensor->rows * tensor->cols; i++)
        tensor->data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
}
