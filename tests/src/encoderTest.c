#include "../../include/tensor.h"
#include "../../include/attention.h"
#include "../../include/encoder.h"
#include "../include/testUtils.h"

extern ModelConfig modelConfig;

void encoderTest() {
    Tensor *input = tensorCreate(3, D_MODEL);
    randomFill(input);

    EncoderLayer layer = {
        .W_Q = tensorCreate(D_MODEL, D_MODEL),
        .W_K = tensorCreate(D_MODEL, D_MODEL),
        .W_V = tensorCreate(D_MODEL, D_MODEL),
        .W_O = tensorCreate(D_MODEL, D_MODEL),
        .W1 = tensorCreate(D_MODEL, D_MODEL * NUM_HEADS),
        .W2 = tensorCreate(D_MODEL * NUM_HEADS, D_MODEL),
        .B1 = tensorCreate(1, D_MODEL * NUM_HEADS),
        .B2 = tensorCreate(1, D_MODEL)
    };


    randomFill(layer.W_Q), randomFill(layer.W_K);
    randomFill(layer.W_V), randomFill(layer.W_O);
    randomFill(layer.W1), randomFill(layer.W2);
    randomFill(layer.B1), randomFill(layer.B2);

    Tensor *output = encoderLayerForward(input, &layer, &modelConfig);
    tensorPrint(output);

    tensorFree(input), tensorFree(output);

    tensorFree(layer.W_Q), tensorFree(layer.W_K), tensorFree(layer.W_V), tensorFree(layer.W_O);
    tensorFree(layer.W1), tensorFree(layer.W2);
    tensorFree(layer.B1), tensorFree(layer.B2);
}