#ifndef ACTIVATIONFUNCTIONS_HPP
#define ACTIVATIONFUNCTIONS_HPP

float relu(float value);
float drelu(float value);

void softmax(float* z, int size);
float dsoftmax(float* a, int index, int j);

float heNormalInitialization(int n_input_neuron);
float xavierInitialization(int n_input_neuron, int n_output_neuron);

#endif
