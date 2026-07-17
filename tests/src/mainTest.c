#include "../include/tensorTest.h"
#include "../include/attentionTest.h"
#include "../include/encoderTest.h"
#include "../include/backwardTest.h"

void mainTest() {
    tensorTest();
    attentionTest();
    encoderTest();
    backwardTest();
}