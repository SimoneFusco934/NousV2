#ifndef CONVOLUTIONAL_HPP
#define CONVOLUTIONAL_HPP

#include "Layer.hpp"

class Convolutional : public Layer {

	private:

    int n_kernel;
    int kernel_size;
    std::string activation_function;
  
    float**** kernel; //kernel_shape = [n_kernels][inputShape][size_kernel][size_kernel]
    float*** bias; //bias_shape = [n_kernels][input_shape[1] - kernel_size + 1][input_shape[2] - kernel_size + 1]

		float**** dkernel;
		float*** dbias;
    
  public:

    Convolutional(int n_kernel, int kernel_size, int input_shape[3], std::string activation_function, std::string weight_initialization_technique, std::string bias_initialization_technique);

    void forwardPropagate() override;

		void backPropagate() override;

		void update() override;

		void saveToFile(std::ofstream& write_to_file) override;

		void restoreFromFile(std::ifstream& read_from_file) override;

    ~Convolutional() override;

};

#endif
