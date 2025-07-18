#include <iostream>
#include <omp.h>
#include "../hppFiles/Dense.hpp"
#include "../hppFiles/ActivationFunctions.hpp"

Dense::Dense(int n_neurons, int input_shape[3], std::string activation_function, std::string weight_initialization_technique, std::string bias_initialization_technique){
	
	//Save layer variables
	this->n_neurons = n_neurons;
	this->activation_function = activation_function;
	for(int i = 0; i < 3; i++){
		this->input_shape[i] = input_shape[i];
	}
	this->output_shape[0] = n_neurons;
	this->output_shape[1] = 1;
	this->output_shape[2] = 1;

	//Initialize output array
	output = new float[n_neurons]();

	//Initalize weight and dweight arrays
	weight = new float*[n_neurons];
	dweight = new float*[n_neurons];

	for(int i = 0; i < n_neurons; i++){
		weight[i] = new float[input_shape[0] * input_shape[1] * input_shape[2]];
		dweight[i] = new float[input_shape[0] * input_shape[1] * input_shape[2]]();

		if(weight_initialization_technique == "he normal"){
			for(int j = 0; j < input_shape[0] * input_shape[1] * input_shape[2]; j++){
				weight[i][j] = heNormalInitialization(input_shape[0] * input_shape[1] * input_shape[2]);
			}
		}else if(weight_initialization_technique == "xavier"){
    	for(int j = 0; j < input_shape[0] * input_shape[1] * input_shape[2]; j++){
				weight[i][j] = xavierInitialization(input_shape[0] * input_shape[1] * input_shape[2], n_neurons);
      }
		}
	}

	//Initialize bias and dbias arrays
	bias = new float[n_neurons];
	dbias = new float[n_neurons]();

	if(bias_initialization_technique == "zero"){
		for(int i = 0; i < n_neurons; i++){
			bias[i] = 0.0f;
		}
	}
}

void Dense::forwardPropagate(){

	#pragma omp parallel for
	for(int i = 0; i < n_neurons; i++){

		//Clean before writing
		output[i] = 0.0f;
		
		#pragma omp simd
		for(int j = 0; j < input_shape[0] * input_shape[1] * input_shape[2]; j++){
			output[i] += input[j] * weight[i][j];
		}

		//Add bias
		output[i] += bias[i];

		//Pass to activation function
		if(activation_function == "relu"){
			output[i] = relu(output[i]);
		}
	}

	if(activation_function == "softmax"){
		softmax(output, n_neurons);
	}
}

void Dense::backPropagate(){

	float loss_derivative_with_respect_to_z;
	
	//Clean before writing
	if(output_relative_derivatives != nullptr){
		#pragma omp simd
		for(int i = 0; i < input_shape[0] * input_shape[1] * input_shape[2]; i++){
			output_relative_derivatives[i] = 0.0f;
		}
	}

	//#pragma omp parallel for //private(loss_derivative_with_respect_to_z)
	for(int i = 0; i < n_neurons; i++){

		//Clean before writing
		loss_derivative_with_respect_to_z = 0.0f;

		//Calculate loss derivative with respect to z
		if(activation_function == "relu"){
			loss_derivative_with_respect_to_z = input_relative_derivatives[i] * drelu(output[i]);
		}else if(activation_function == "softmax"){
			#pragma omp parallel for reduction(+:loss_derivative_with_respect_to_z)
			for(int j = 0; j < n_neurons; j++){
				loss_derivative_with_respect_to_z += dsoftmax(output, i, j) * input_relative_derivatives[j];
			}
		}

		dbias[i] += loss_derivative_with_respect_to_z;

		//#pragma omp simd
		for(int j = 0; j < input_shape[0] * input_shape[1] * input_shape[2]; j++){

			dweight[i][j] += loss_derivative_with_respect_to_z * input[j];

      if(output_relative_derivatives != nullptr){
        output_relative_derivatives[j] += loss_derivative_with_respect_to_z * weight[i][j];
      }
    }
	}
}

void Dense::update(){

	float weight_scale = learning_rate / batch_size;
	
	//Updates weight
	#pragma omp parallel for
	for(int i = 0; i < n_neurons; i++){
		#pragma omp simd
		for(int j = 0; j < input_shape[0] * input_shape[1] * input_shape[2]; j++){
			weight[i][j] -= dweight[i][j] * weight_scale;

			dweight[i][j] = 0.0f;
		}
	}

	//Updates bias
	#pragma omp parallel for
	for(int i = 0; i < n_neurons; i++){
		bias[i] -= dbias[i] * weight_scale;

		dbias[i] = 0.0f;
	}
}

void Dense::saveToFile(std::ofstream& write_to_file){

	//Write layer variables
	write_to_file << "Dense" << std::endl << n_neurons << std::endl << activation_function << std::endl;
		
	//Write weight
	for(int i = 0; i < n_neurons; i++){
		for(int j = 0; j < input_shape[0] * input_shape[1] * input_shape[2]; j++){
			write_to_file << weight[i][j] << std::endl;
		}
	}

	//Write bias
	for(int i = 0; i < n_neurons; i++){
		write_to_file << bias[i] << std::endl;
	}
};

void Dense::restoreFromFile(std::ifstream& read_from_file){
	
	//Read weight
  for(int i = 0; i < n_neurons; i++){
    for(int j = 0; j < input_shape[0] * input_shape[1] * input_shape[2]; j++){
      read_from_file >> weight[i][j];
    }
  }

  //Read bias
  for(int i = 0; i < n_neurons; i++){
    read_from_file >> bias[i];
  }
}

Dense::~Dense(){

	//Delete weight and dweight
	for(int i = 0; i < n_neurons; i++){
		delete[] weight[i];
		delete[] dweight[i];
	}

	delete[] weight;
	delete[] dweight;

	//Delete bias and dbias
	delete[] bias;
	delete[] dbias;
}
