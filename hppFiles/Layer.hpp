#ifndef LAYER_HPP
#define LAYER_HPP

#include <fstream>

class Layer {

	protected:
		
		//Common layer variables
		int input_shape[3];
		int output_shape[3];
    float* input;
    float* input_relative_derivatives;
    float* output;
    float* output_relative_derivatives;

		//Static layer parameters
		static std::string optimizer;
		static int batch_size;
    static int epochs;
    static float learning_rate;

		//Common layer functions
		void initialize(bool isFirstLayer);
		void link(float* input, float* input_relative_derivatives);
		
	public: 

		friend class Model;

		//Layer-specific functions
    virtual void forwardPropagate() = 0;
		virtual void backPropagate() = 0;
		virtual void update() = 0;
		virtual void saveToFile(std::ofstream& write_to_file) = 0;
		virtual void restoreFromFile(std::ifstream& read_from_file) = 0;

		virtual ~Layer();

};

#endif

