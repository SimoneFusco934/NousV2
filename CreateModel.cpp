#include "./hppFiles/Model.hpp"

int main(){

  Model m;

	m.setTrainFiles("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte");
  m.setTestFiles("./t10k-images.idx3-ubyte", "./t10k-labels.idx1-ubyte");
		
	m.setDataAugumentation(true);

	m.setHyperparameters(0.1f, 20, 3);

	m.addLayerConvolutional(32, 3, "relu", "he normal", "zero");
	m.addLayerMaxPooling(2);
	m.addLayerConvolutional(64, 3, "relu", "he normal", "zero");
	m.addLayerMaxPooling(2);
	m.addLayerDense(128, "relu", "he normal", "zero");
	m.addLayerDense(10, "softmax", "xavier", "zero");

	m.setUp();

	m.train();
	
	m.test();
}