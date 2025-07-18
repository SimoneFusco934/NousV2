#include <iostream>
#include "../hppFiles/Layer.hpp"

//Default layer variables
std::string Layer::optimizer = "none";
int Layer::batch_size = 1;
int Layer::epochs = 1;
float Layer::learning_rate = 0.1f;

void Layer::link(float* input, float* input_relative_derivatives){
  this->input = input;
  this->input_relative_derivatives = input_relative_derivatives;
}

void Layer::initialize(bool isFirstLayer){

  //if(optimizer == "momentum"){
  //  //Initialize velocity arrays
  //}

  if(!isFirstLayer){
    output_relative_derivatives = new float[input_shape[0] * input_shape[1] * input_shape[2]]();
  }else{
    output_relative_derivatives = nullptr;
  }
}

Layer::~Layer(){

	//Delete output
  delete[] output;

	//If it exists, delete output_relative_derivatives
	if(output_relative_derivatives != nullptr){
    delete[] output_relative_derivatives;
  }
}
