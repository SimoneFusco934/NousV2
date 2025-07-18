#include <iostream>
#include <omp.h>
#include "../hppFiles/Convolutional.hpp"
#include "../hppFiles/ActivationFunctions.hpp"

//weight_initialization_technique == "he normal" || "xavier", bias_initialization_technique == "zero", activation_function == "relu" || "softmax"
Convolutional::Convolutional(int n_kernel, int kernel_size, int input_shape[3], std::string activation_function, std::string weight_initialization_technique, std::string bias_initialization_technique){

	//Save layer variables
  this->n_kernel = n_kernel;
  this->kernel_size = kernel_size;
	this->activation_function = activation_function;
	for(int i = 0; i < 3; i++){
		this->input_shape[i] = input_shape[i];
  }
  this->output_shape[0] = n_kernel;
  this->output_shape[1] = input_shape[1] - kernel_size + 1;
  this->output_shape[2] = input_shape[2] - kernel_size + 1;

  //Initialize output array
  output = new float[output_shape[0] * output_shape[1] * output_shape[2]]();

  //Initialize kernel and dkernel arrays
  kernel = new float***[n_kernel];
	dkernel = new float***[n_kernel];

  for(int i = 0; i < n_kernel; i++){
  	kernel[i] = new float**[input_shape[0]];
		dkernel[i] = new float**[input_shape[0]];
    for(int j = 0; j < input_shape[0]; j++){
    	kernel[i][j] = new float*[kernel_size];
			dkernel[i][j] = new float*[kernel_size];
      for(int k = 0; k < kernel_size; k++){
      	kernel[i][j][k] = new float[kernel_size];
				//Initialize dkernel to 0
				dkernel[i][j][k] = new float[kernel_size]();

				if(weight_initialization_technique == "he normal"){
					for(int m = 0; m  < kernel_size; m++){
						kernel[i][j][k][m] = heNormalInitialization(input_shape[0] * kernel_size * kernel_size);
					}
				}else if(weight_initialization_technique == "xavier"){
					for(int m = 0; m  < kernel_size; m++){
            kernel[i][j][k][m] = xavierInitialization(input_shape[0], n_kernel);
          }
				}
      }
    }
  }

  //Initialize bias and dbias arrays
  bias = new float**[n_kernel];
	dbias = new float**[n_kernel];
    	
  for(int i = 0; i < n_kernel; i++){
  	bias[i] = new float*[output_shape[1]];
		dbias[i] = new float*[output_shape[1]];
    for(int j = 0; j < output_shape[1]; j++){
    	bias[i][j] = new float[output_shape[2]];
			//Initialize dbias to 0
			dbias[i][j] = new float[output_shape[2]]();

			if(bias_initialization_technique == "zero"){
				for(int k = 0; k < output_shape[2]; k++){
					bias[i][j][k] = 0.0f;
				}
			}
    }
  }
}

void Convolutional::forwardPropagate(){
	
	int output_index = 0;
	int input_index = 0;

	int output_shape_1_2 = output_shape[1] * output_shape[2];
	int output_shape_1 = output_shape[1];
	int output_shape_2 = output_shape[2];

	int input_shape_1_2 = input_shape[1] * input_shape[2];
	int input_shape_0 = input_shape[0];
	int input_shape_1 = input_shape[1];
	int input_shape_2 = input_shape[2];

	#pragma omp parallel for collapse(3)
	for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){
		for(int indexY = 0; indexY < output_shape_1; indexY++){ 
    	for(int indexX = 0; indexX < output_shape_2; indexX++){

				//Calculate output index
				output_index = kernelSet * output_shape_1_2 + indexY * output_shape_2 + indexX;

				//Clean before writing
				output[output_index] = 0.0f;


				#pragma omp simd
      	for(int kernelId = 0; kernelId < input_shape_0; kernelId++){

					//Calculate input index
					input_index = kernelId * input_shape_1_2 + indexY * input_shape_2 + indexX;

        	for(int y = 0; y < kernel_size; y++){
          	for(int x = 0; x < kernel_size; x++){
            	output[output_index] += input[input_index + y * input_shape_2 + x] * kernel[kernelSet][kernelId][y][x];
            }
          }
        }
			
				//Add bias
        output[output_index] += bias[kernelSet][indexY][indexX];
        
				//Pass to activation function
				if(activation_function == "relu"){
					output[output_index] = relu(output[output_index]);
				}
      }
    }
  }

	if(activation_function == "softmax"){
		softmax(output, output_shape[0] * output_shape[1] * output_shape[2]);
	}
}

void Convolutional::backPropagate(){

	int output_index = 0;
	int input_index = 0;

	int output_shape_0 = output_shape[0];
	int output_shape_1 = output_shape[1];
  int output_shape_2 = output_shape[2];
	int output_shape_1_2 = output_shape_1 * output_shape_2;
	int output_shape_0_1_2 = output_shape_0 * output_shape_1_2;

  int input_shape_0 = input_shape[0];
  int input_shape_1 = input_shape[1];
  int input_shape_2 = input_shape[2];
	int input_shape_1_2 = input_shape_1 * input_shape_2;
	int input_shape_0_1_2 = input_shape_0 * input_shape_1_2;
	
	float loss_derivative_with_respect_to_z = 0.0f;

	//Clean before writing
  if(output_relative_derivatives != nullptr){
		#pragma omp parallel for
    for(int i = 0; i < input_shape_0_1_2; i++){
      output_relative_derivatives[i] = 0.0f;
    }
  }

	#pragma omp parallel for collapse(3) private (loss_derivative_with_respect_to_z)	
	for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){
		for(int indexY = 0; indexY < output_shape_1; indexY++){
			for(int indexX = 0; indexX < output_shape_2; indexX++){

				//Calculate output index
				output_index = kernelSet * output_shape_1_2 + indexY * output_shape_2 + indexX;

				//Clean before writing
				loss_derivative_with_respect_to_z = 0.0f;


				if(activation_function == "relu"){
					loss_derivative_with_respect_to_z = input_relative_derivatives[output_index] * drelu(output[output_index]);		
				}else if(activation_function == "softmax"){
					for(int j = 0; j < output_shape_0_1_2; j++){
						loss_derivative_with_respect_to_z += dsoftmax(output, output_index, j);
					}
				}

				dbias[kernelSet][indexY][indexX] += loss_derivative_with_respect_to_z;

				for(int kernelId = 0; kernelId < input_shape_0; kernelId++){

					input_index = kernelId * input_shape_1_2 + indexY * input_shape_2 + indexX;

					#pragma omp simd
					for(int y = 0; y < kernel_size; y++){
						for(int x = 0; x < kernel_size; x++){
							dkernel[kernelSet][kernelId][y][x] += loss_derivative_with_respect_to_z * input[input_index + y * input_shape_2 + x];

							if(output_relative_derivatives != nullptr){
								output_relative_derivatives[input_index + y * input_shape_2 + x] += loss_derivative_with_respect_to_z * kernel[kernelSet][kernelId][y][x];
							}
						}
					}
				}
			}
		}
	}
	
}

void Convolutional::update(){

	float weight_scale = learning_rate / batch_size;

	//Updates kernel
	#pragma omp parallel for collapse(4)
	for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){
		for(int kernelId = 0; kernelId < input_shape[0]; kernelId++){
			for(int y = 0; y < kernel_size; y++){
				for(int x = 0; x < kernel_size; x++){

					kernel[kernelSet][kernelId][y][x] -= dkernel[kernelSet][kernelId][y][x] * weight_scale;

					//Momentum
					//
				}
			}
		}
	}

	#pragma omp parallel for collapse(3)
  for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){
    for(int kernelId = 0; kernelId < input_shape[0]; kernelId++){
      for(int y = 0; y < kernel_size; y++){
        for(int x = 0; x < kernel_size; x++){

          dkernel[kernelSet][kernelId][y][x] = 0.0f;

          //Momentum
          //
        }
      }
    }
  }

	//Updates bias
	#pragma omp parallel for collapse(3)
	for(int indexY = 0; indexY < output_shape[1]; indexY++){
    for(int indexX = 0; indexX < output_shape[2]; indexX++){
      for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){

				bias[kernelSet][indexY][indexX] -= dbias[kernelSet][indexY][indexX] * weight_scale;

				//Momentum
				//
			}
		}
	}

	#pragma omp parallel for collapse(2)
  for(int indexY = 0; indexY < output_shape[1]; indexY++){
    for(int indexX = 0; indexX < output_shape[2]; indexX++){
      for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){

        dbias[kernelSet][indexY][indexX] = 0.0f;

        //Momentum
        //
      }
    }
  }

}

void Convolutional::saveToFile(std::ofstream& write_to_file){

	//Write layer variables
	write_to_file << "Convolutional" << std::endl << n_kernel << std::endl << kernel_size << std::endl << activation_function << std::endl;

	//Write kernel
	for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){
		for(int kernelId = 0; kernelId < input_shape[0]; kernelId++){
			for(int y = 0; y < kernel_size; y++){
				for(int x = 0; x < kernel_size; x++){
					write_to_file << kernel[kernelSet][kernelId][y][x] << std::endl;
				}
			}
		}
	}

	//Write bias
	for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){
		for(int indexY = 0; indexY < output_shape[1]; indexY++){
			for(int indexX = 0; indexX < output_shape[2]; indexX++){
				write_to_file << bias[kernelSet][indexY][indexX] << std::endl;
			}
		}
	}	
};

void Convolutional::restoreFromFile(std::ifstream& read_from_file){

	//Read kernel
  for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){
    for(int kernelId = 0; kernelId < input_shape[0]; kernelId++){
      for(int y = 0; y < kernel_size; y++){
        for(int x = 0; x < kernel_size; x++){
          read_from_file >> kernel[kernelSet][kernelId][y][x];
        }
      }
    }
  }

  //Read bias
  for(int kernelSet = 0; kernelSet < n_kernel; kernelSet++){
    for(int indexY = 0; indexY < output_shape[1]; indexY++){
      for(int indexX = 0; indexX < output_shape[2]; indexX++){
        read_from_file >> bias[kernelSet][indexY][indexX];
      }
    }
  }
}

Convolutional::~Convolutional(){

  //Delete kernel and dkernel
  for(int i = 0; i < n_kernel; i++){
  	for(int j = 0; j < input_shape[0]; j++){
    	for(int k = 0; k < kernel_size; k++){
      	delete[] kernel[i][j][k];
				delete[] dkernel[i][j][k];
      }
     	delete[] kernel[i][j];
			delete[] dkernel[i][j];
    }
    delete[] kernel[i];
		delete[] dkernel[i];
  }
  delete[] kernel;
	delete[] dkernel;

  //Delete bias and dbias
  for(int i = 0; i < n_kernel; i++){
  	for(int j = 0; j < output_shape[1]; j++){
    	delete[] bias[i][j];
			delete[] dbias[i][j];
    }
    delete[] bias[i];
		delete[] dbias[i];
  }
  delete[] bias;
	delete[] dbias;
}
