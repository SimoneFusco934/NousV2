#include <iostream>
#include <vector>
#include "../hppFiles/Model.hpp"
#include "../hppFiles/Convolutional.hpp"
#include "../hppFiles/Dense.hpp"
#include "../hppFiles/MaxPooling.hpp"
#include "../hppFiles/ReadData.hpp"
#include "../hppFiles/LossFunctions.hpp"

void Model::initModelInput(){
  //Right now the model supports 3d input/output data. Will become more flexible in a future update.
  for(int i = 0; i < 3; i++){
    model_input_shape[i] = (int) data_dimension[i];
  }

  model_input = new float[model_input_shape[0] * model_input_shape[1] * model_input_shape[2]]();
};

void Model::initModelOutput(int z, int y, int x){
	model_output_shape[0] = z;
	model_output_shape[1] = y;
	model_output_shape[2] = x;

	model_output_relative_derivatives = new float[model_output_shape[0] * model_output_shape[1] * model_output_shape[2]];
}

void Model::addLayerConvolutional(int n_kernel, int kernel_size, std::string activation_function, std::string weight_initialization_technique, std::string bias_initialization_technique){
	if(layers.size() == 0){
  		layers.push_back(new Convolutional(n_kernel, kernel_size, model_input_shape, activation_function, weight_initialization_technique, bias_initialization_technique));
  	}else{
    	layers.push_back(new Convolutional(n_kernel, kernel_size, layers[layers.size() - 1]->output_shape, activation_function, weight_initialization_technique, bias_initialization_technique));
  	}
}

void Model::addLayerDense(int n_neurons, std::string activation_function, std::string weight_initialization_technique, std::string bias_initialization_technique){
	if(layers.size() == 0){
    	layers.push_back(new Dense(n_neurons, model_input_shape, activation_function, weight_initialization_technique, bias_initialization_technique));
  	}else{
    	layers.push_back(new Dense(n_neurons, layers[layers.size() - 1]->output_shape, activation_function, weight_initialization_technique, bias_initialization_technique));
  	}					
}

void Model::addLayerMaxPooling(int pooling_size){
	if(layers.size() == 0){
    	layers.push_back(new MaxPooling(pooling_size, model_input_shape));
  	}else{
    	layers.push_back(new MaxPooling(pooling_size, layers[layers.size() - 1]->output_shape));
  	}
}

void Model::setUp(){
	
	//Initialize layers
	layers[0]->initialize(true);

	for(int i = 1; i < layers.size(); i++){
		layers[i]->initialize(false);
	}

	//Link layers
	initModelOutput(layers[layers.size() - 1]->output_shape[0], layers[layers.size() - 1]->output_shape[1], layers[layers.size() - 1]->output_shape[2]);

	if(layers.size() == 1){
		layers[0]->link(model_input, model_output_relative_derivatives);
	}else{

		layers[0]->link(model_input, layers[1]->output_relative_derivatives);

		for(int i = 1; i < layers.size() - 1; i++){
			layers[i]->link(layers[i-1]->output, layers[i+1]->output_relative_derivatives);
		}

		layers[layers.size() - 1]->link(layers[layers.size() - 2]->output, model_output_relative_derivatives);
	}
}

void Model::setTrainFiles(std::string train_data_file_path, std::string train_labels_file_path){
	
	train_data_file.open(train_data_file_path, std::ios::binary);
	train_labels_file.open(train_labels_file_path, std::ios::binary);

	if(!train_data_file){
		std::cerr << "Train data file could not be opened." << std::endl;
	}	

	if(!train_labels_file){
		std::cerr << "Train labels file could not be opened." << std::endl;
	}

  initializeFile(train_data_file, data_type, data_n_dimension, train_size, data_dimension);
  initializeFile(train_labels_file, label_type, label_n_dimension, train_size, label_dimension);

  if(model_input == nullptr){
    initModelInput();
  }
}

void Model::setTestFiles(std::string test_data_file_path, std::string test_labels_file_path){

  test_data_file.open(test_data_file_path, std::ios::binary);
  test_labels_file.open(test_labels_file_path, std::ios::binary);

  if(!test_data_file){
    std::cerr << "Test data file could not be opened." << std::endl;
  }

  if(!test_labels_file){
    std::cerr << "Test labels file could not be opened." << std::endl;
  }

  initializeFile(test_data_file, data_type, data_n_dimension, test_size, data_dimension);
  initializeFile(test_labels_file, label_type, label_n_dimension, test_size, label_dimension);

  if(model_input == nullptr){
    initModelInput();
  }
}


void Model::nextData(std::ifstream& data_file, std::ifstream& label_file, float* input, int& label){	
	readImage(data_file, data_dimension, data_n_dimension, data_type, input, data_augumentation);
	readLabel(label_file, label);
}

void Model::setDataAugumentation(bool flag){
	data_augumentation = flag;
}	

void Model::setHyperparameters(float learning_rate, int batch_size, int epochs){
	Layer::learning_rate = learning_rate;
	Layer::batch_size = batch_size;
	Layer::epochs = epochs;
}

void Model::train(){
	
	std::cout << "training.. " << std::endl;

	int index_pred = 0;
	float max = 0.0f;

	int n_success = 0;
	float accuracy = 0.0f;

	int label;
  int show_results_after_n_batches = 50;

	float true_label_array[model_output_shape[0] * model_output_shape[1] * model_output_shape[2]];

	for(int epoch_n = 1; epoch_n <= Layer::epochs; epoch_n++){
		for(int batch_n = 1; batch_n <= (train_size / Layer::batch_size); batch_n++){
			for(int data_point = 1; data_point <= Layer::batch_size; data_point++){

				//Setting input
				nextData(train_data_file, train_labels_file, model_input, label);
        
        /*debug
        std::cout << "Printing image and label..." << std::endl;

        for(int i = 0; i < 28; i++){
          for(int j = 0; j < 28; j++){
            if(model_input[i*28+j] > 0.5f){
              std::cout << "#";
            }else{
              std::cout << " ";
            }
          }
          std::cout << std::endl;
        }

        std::cout << "Label: " << label << std::endl;*/
        ////////////////////////////////////////////////

				for(int i = 0; i < 10; i++){
					if(i == label){
						true_label_array[i] = 1.0f;
					}else{
						true_label_array[i] = 0.0f;
					}
				}
        
				//Forward propagation
				for(int i = 0; i < layers.size(); i++){
					layers[i]->forwardPropagate();
				}	

				//Calculating model prediction
				for(int i = 0; i < 10; i++){
					if(layers[layers.size() - 1]->output[i] > max){
						index_pred = i;
						max = layers[layers.size() - 1]->output[i];
					}
				}

				max = 0.0f;

				//Check if model prediction is correct
				if(index_pred == label){
					n_success++;
				}

				//Loss function
				crossEntropyLoss(layers[layers.size() - 1]->output, true_label_array, 10, model_output_relative_derivatives);

				//Back propagation
				for(int i = layers.size() - 1; i >= 0; i--){
					layers[i]->backPropagate();
				}
          
			}

			//Update
			for(int i = 0; i < layers.size(); i++){
				layers[i]->update();
			}

			if(batch_n != 0 && batch_n % show_results_after_n_batches == 0){
				std::cout << "Batch n. " << batch_n << ", n_success: " << n_success << " accuracy: " << (n_success * 100.0) / (Layer::batch_size * show_results_after_n_batches) << "%" << std::endl;
				n_success = 0;
			}
      
		}

		std::cout << "Epoch n. " << epoch_n << " terminated." << std::endl << std::endl;
		
		//reset file pointer
		train_data_file.seekg(4 + (data_n_dimension * 4), std::ios::beg);
		train_labels_file.seekg(4 + (label_n_dimension * 4), std::ios::beg);

	}
  
}

void Model::test(){

	std::cout << std::endl << std::endl << "Testing..." << std::endl;

	int index_pred = 0;
  float max = 0.0f;

  int n_success = 0;
  float accuracy = 0.0f;

  int label;

  float true_label_array[model_output_shape[0] * model_output_shape[1] * model_output_shape[2]];

	for(int data_point = 1; data_point <= test_size; data_point++){

		//Setting input
  	nextData(test_data_file, test_labels_file, model_input, label); 

    for(int i = 0; i < 10; i++){
    	if(i == label){
      	true_label_array[i] = 1.0f;
      }else{
      	true_label_array[i] = 0.0f;
      }
   	}

    //Forward propagation
    for(int i = 0; i < layers.size(); i++){
    	layers[i]->forwardPropagate();
    }

    //Calculating model prediction
    for(int i = 0; i < 10; i++){
    	if(layers[layers.size() - 1]->output[i] > max){
      	index_pred = i;
        max = layers[layers.size() - 1]->output[i];
      }
    }

    max = 0.0f;

    if(index_pred == label){
    	n_success++;
    }

	}

	std::cout << "Test n_success: " << n_success << ", test accuracy: " << (n_success * 100.0) / 10000.0 << "%" << std::endl << std::endl;

	std::cout << "Do you want to save the model? [y/n]" << std::endl;

	char answer;

	std::cin >> answer;

	if(answer == 'y' || answer == 'Y'){

		std::string file_name;

		std::cout << "Enter file name: " << std::endl;

		std::cin >> file_name;

		saveToFile(file_name);
	}

}

void Model::saveToFile(std::string write_to_file_path){

	write_to_file.open(write_to_file_path);

	if(!write_to_file){
    std::cerr << "Error opening the file: " << write_to_file_path << "." << std::endl;
		return;
  }

  write_to_file << data_n_dimension << std::endl;

  for(int i = 0; i < (int) data_n_dimension; i++){
    write_to_file << data_dimension[i] << std::endl;
  }

	for(int i = 0; i < layers.size(); i++){
  	layers[i]->saveToFile(write_to_file);
  }

	write_to_file.close();

}

void Model::restoreSavedModel(std::string read_from_file_path){

	read_from_file.open(read_from_file_path);

	if(!read_from_file){
		std::cerr << "Error opening the file: " << read_from_file_path << "." << std::endl;
		return;
	}
  
  //restore data_n_dimension and data_dimension 
  read_from_file >> data_n_dimension;

  data_dimension = new int[(int) data_n_dimension];

  for(int i = 0; i < (int) data_n_dimension; i++){
    read_from_file >> data_dimension[i];
  }

  //call setModelInputShape()
  initModelInput();
		
	std::string nextLayer;

	while(read_from_file >> nextLayer){

		if(nextLayer == "Dense"){

			int n_neurons;
			std::string activation_function;

			read_from_file >> n_neurons;
			read_from_file >> activation_function;

			addLayerDense(n_neurons, activation_function);

		}else if(nextLayer == "Convolutional"){

			int n_kernel;
			int kernel_size;
			std::string activation_function;

			read_from_file >> n_kernel;
			read_from_file >> kernel_size;
      read_from_file >> activation_function;

      addLayerConvolutional(n_kernel, kernel_size, activation_function);

		}else if(nextLayer == "MaxPooling"){

			int pooling_size;

			read_from_file >> pooling_size;

			addLayerMaxPooling(pooling_size);
		}

		layers[layers.size() - 1]->restoreFromFile(read_from_file);

	}

	setUp();

	read_from_file.close();
}

std::string Model::predict(float* input){

	int index_pred = 0;
  float max = 0.0f;

	//Copy input in model_input
	for(int i = 0; i < model_input_shape[0] * model_input_shape[1] * model_input_shape[2]; i++){
		model_input[i] = input[i];
	}

	//Forward propagation
  for(int i = 0; i < layers.size(); i++){
  	layers[i]->forwardPropagate();
  }

  //Calculating model prediction
  for(int i = 0; i < 10; i++){
  	if(layers[layers.size() - 1]->output[i] > max){
    	index_pred = i;
      max = layers[layers.size() - 1]->output[i];
   	}
  }

	return std::to_string(index_pred);
}

Model::~Model(){

	this->train_data_file.close();
	this->train_labels_file.close();
	this->test_data_file.close();
	this->test_labels_file.close();

  delete[] model_input;
	delete[] model_output_relative_derivatives;

	for(int i = 0; i < layers.size(); i++){
  	delete layers[i];
  }

  if(data_dimension != nullptr){
    delete[] data_dimension;
  }

  if(label_dimension != nullptr){
    delete[] label_dimension;
  }
};
