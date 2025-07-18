#include <random>

float relu(float value){
	return value > 0.0f ? value : 0.0f;
}

float drelu(float value){
	return value > 0.0f ? 1.0f : 0.0f;
}

void softmax(float* z, int size){

	float sum = 0.0f;

	for(int j = 0; j < size; j++){
		sum += exp(z[j]);
	}

	for(int i = 0; i < size; i++){
		z[i] = exp(z[i]) / sum;
	}
}

float dsoftmax(float *a, int index, int j){
	if(index == j){
		return (a[index] * (1 - a[index]));
	}else{
		return (-(a[index] * a[j]));
	}
}

float heNormalInitialization(int n_input_neuron){

	std::random_device rd;
	std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / n_input_neuron));

	return dist(gen);
}

float xavierInitialization(int n_input_neuron, int n_output_neuron){

				/*
	std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, std::sqrt(6.0f / (n_input_neuron + n_output_neuron)));

	return dist(gen);
	*/

	std::random_device rd;
  std::mt19937 gen(rd());
  // Glorot Uniform initialization with uniform distribution
  float limit = std::sqrt(6.0f / (n_input_neuron + n_output_neuron));
  std::uniform_real_distribution<float> dist(-limit, limit); // Uniform distribution
  return dist(gen);
}
