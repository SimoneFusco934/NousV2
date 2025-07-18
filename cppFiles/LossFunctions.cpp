#include "../hppFiles/LossFunctions.hpp"

void crossEntropyLoss(float* predicted_probabilities, float* true_labels, int size, float* output_relative_derivatives){

	//Clean before writing
	for(int i = 0; i < size; i++){
		output_relative_derivatives[i] = 0.0f;
	}

	if(size == 1){

		//Loss = -(y*log(p) + (1-y)*log(1-p))

		//Derivative of binary cross entropy loss
		output_relative_derivatives[0] = (predicted_probabilities[0] - true_labels[0]) / (predicted_probabilities[0] * (1 - predicted_probabilities[0]));
	}else{

		//Loss = -(sum for i = 1 -> C of yi*log(pi))

		//Derivative of multi-class cross entropy loss
		for(int i = 0; i < size; i++){
			output_relative_derivatives[i] = -(true_labels[i]/predicted_probabilities[i]);
		}

	}	
}

void meanSquaredError(float* input, float* true_labels, int size, float* output_relative_derivatives){

	//Clean before writing
  for(int i = 0; i < size; i++){
    output_relative_derivatives[i] = 0.0f;
  }

	//Loss = 1/n * sum for i = 1 -> N of (yi - pi)²
	for(int i = 0; i < size; i++){
		output_relative_derivatives[i] = -((1/size) * 2 * (true_labels[i] - input[i]));
	}
}
