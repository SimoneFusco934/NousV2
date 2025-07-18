#ifndef MODEL_HPP
#define MODEL_HPP

#include <fstream>
#include <vector>
#include "Layer.hpp"

class Model {

	private:

	  uint8_t data_type; //data type will be always automatically converted to float. It is saved just to pass to readData()
    uint8_t label_type;
		uint8_t data_n_dimension; //needs to be saved to file
    uint8_t label_n_dimension;
		int train_size;	
	  int test_size;
		int* data_dimension = nullptr; //needs to be saved to file
    int* label_dimension = nullptr;

		bool data_augumentation = false;

    std::vector<Layer*> layers;
	  float* model_input = nullptr;
    int model_input_shape[3]; //the same as data_dimension
		float* model_output_relative_derivatives;
		int model_output_shape[3];

		std::ifstream train_data_file;
		std::ifstream train_labels_file;
		std::ifstream test_data_file;
		std::ifstream test_labels_file;

		std::ofstream write_to_file;
		std::ifstream read_from_file;

    void initModelInput();

		void initModelOutput(int z, int y, int x);

		void nextData(std::ifstream& data_file, std::ifstream& label_file, float* input, int& label);


	public:

    void addLayerConvolutional(int n_kernel, int kernel_size, std::string activation_function = "none", std::string weight_initialization_technique = "none", std::string bias_initialization_technique = "none");

    void addLayerDense(int n_neurons, std::string activation_function = "none", std::string weight_initialization_technique = "none", std::string bias_initialization_technique = "none");

		void addLayerMaxPooling(int pooling_size);

		void saveToFile(std::string write_to_file_path);

		void restoreSavedModel(std::string read_from_file_path);

		std::string predict(float* input);

		void setUp(); //Needs to be called afted adding layers

		void setTrainFiles(std::string train_data_file_path, std::string train_labels_file_path);

		void setTestFiles(std::string test_data_file_path, std::string test_labels_file_path);

		void setDataAugumentation(bool flag);

		void setHyperparameters(float learning_rate, int batch_size, int epochs);

		void train(); //Last to be called

		void test();

    ~Model();
};

#endif
