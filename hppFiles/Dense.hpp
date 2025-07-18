#ifndef DENSE_HPP
#define DENSE_HPP

#include "Layer.hpp"

class Dense : public Layer {

	private:

		int n_neurons;
		std::string activation_function;

		float** weight;
		float* bias;

		float** dweight;
		float* dbias;

	public: 

		Dense(int n_neurons, int input_shape[3], std::string activation_function, std::string weight_initialization_technique, std::string bias_initialization_technique);

    void forwardPropagate() override;

    void backPropagate() override;

    void update() override;

		void saveToFile(std::ofstream& write_to_file) override;

		void restoreFromFile(std::ifstream& read_from_file) override;

    ~Dense() override;
};

#endif
