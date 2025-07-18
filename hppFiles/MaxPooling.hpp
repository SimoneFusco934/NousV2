#ifndef MAXPOOLING_HPP
#define MAXPOOLING_HPP

#include "Layer.hpp"

class MaxPooling : public Layer {

	private:

		int pooling_size;

	public:

		MaxPooling(int pooling_size, int input_shape[3]);

    void forwardPropagate() override;

    void backPropagate() override;

    void update() override;

		void saveToFile(std::ofstream& write_to_file) override;	

		void restoreFromFile(std::ifstream& read_from_file) override;

    ~MaxPooling() override;	
};

#endif
