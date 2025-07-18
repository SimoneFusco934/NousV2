#ifndef LOSS_FUNCTIONS
#define LOSS_FUNCTIONS

void crossEntropyLoss(float* predicted_probabilities, float* true_labels, int size, float* output_relative_derivatives);
void meanSquaredError(float* input, float* true_labels, int size, float* output_relative_derivatives);

#endif
