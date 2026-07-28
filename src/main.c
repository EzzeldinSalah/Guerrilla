#include <stdio.h>
#include "tensor.h"
#include "../tests/tests.h"

int main() {
    tensorTest();
    attentionTest();
    encoderTest();
    backwardTest();
    return 0;
}