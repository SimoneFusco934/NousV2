#include <iostream>
#include "../hppFiles/MaxPooling.hpp"

MaxPooling::MaxPooling(int pooling_size, int input_shape[3]){

	//Save layer variables
	this->pooling_size = pooling_size;	
	for(int i = 0; i < 3; i++){
		this->input_shape[i] = input_shape[i];
	}
	this->output_shape[0] = input_shape[0];
	this->output_shape[1] = input_shape[1] / pooling_size;
	this->output_shape[2] = input_shape[2] / pooling_size;

	//Initilaize output array
	output = new float[output_shape[0] * output_shape[1] * output_shape[2]]();
}

void MaxPooling::forwardPropagate(){

	float max = 0.0f;

	for(int neuron_n = 0; neuron_n < output_shape[0]; neuron_n++){
		for(int indexY = 0; indexY < input_shape[1] - pooling_size + 1; indexY += pooling_size){
			for(int indexX = 0; indexX < input_shape[2] - pooling_size + 1; indexX += pooling_size){

				for(int y = 0; y < pooling_size; y++){
					for(int x = 0; x < pooling_size; x++){
						if(input[neuron_n * input_shape[1] * input_shape[2] + (indexY + y) * input_shape[2] + (indexX + x)] > max){
							max = input[neuron_n * input_shape[1] * input_shape[2] + (indexY + y) * input_shape[2] + (indexX + x)];
						}
					}
				}

				output[neuron_n * output_shape[1] * output_shape[2] + (indexY / pooling_size) * output_shape[2] + (indexX / pooling_size)] = max;

				max = 0.0f;
			}
		}
	}
}

void MaxPooling::backPropagate(){

	float max = 0.0f;

	//Clean before writing
	if(output_relative_derivatives != nullptr){
		for(int i = 0; i < input_shape[0] * input_shape[1] * input_shape[2]; i++){
			output_relative_derivatives[i] = 0.0f;
		}

		for(int neuron_n = 0; neuron_n < input_shape[0]; neuron_n++){
    	for(int indexY = 0; indexY < input_shape[1] - pooling_size + 1; indexY += pooling_size){
      	for(int indexX = 0; indexX < input_shape[2] - pooling_size + 1; indexX += pooling_size){
				
					max = output[neuron_n * output_shape[1] * output_shape[2] + (indexY / pooling_size) * output_shape[2] + (indexX / pooling_size)];

					for(int y = 0; y < pooling_size; y++){
  	        for(int x = 0; x < pooling_size; x++){
    	        if(input[neuron_n * input_shape[1] * input_shape[2] + (indexY + y) * input_shape[2] + (indexX + x)] == max){
								output_relative_derivatives[neuron_n * input_shape[1] * input_shape[2] + (indexY + y) * input_shape[2] + (indexX + x)] = input_relative_derivatives[neuron_n * output_shape[1] * output_shape[2] + (indexY / pooling_size) * output_shape[2] + (indexX / pooling_size)];
        	    }else{
								output_relative_derivatives[neuron_n * input_shape[1] * input_shape[2] + (indexY + y) * input_shape[2] + (indexX + x)] = 0.0f;
							}
	          }
  	      }

					max = 0.0f;
				}
			}
		}
	}
}

void MaxPooling::update(){};

void MaxPooling::saveToFile(std::ofstream& write_to_file){

	//Write layer variables
	write_to_file << "MaxPooling" << std::endl << pooling_size << std::endl;

};

void MaxPooling::restoreFromFile(std::ifstream& read_from_file){}

MaxPooling::~MaxPooling(){};
