#ifndef READ_DATA
#define READ_DATA

#include <fstream>

int reverseInt(int i);
void initializeFile(std::ifstream &file, uint8_t &type, uint8_t &n_dimension, int &size, int* &dimension);
void readImage(std::ifstream &file, int* dimension, uint8_t n_dimension, uint8_t data_type, float *input, bool augumentation);
void readLabel(std::ifstream &file, int& label);

void rotateImage(float *image, int n_rows, int n_cols);
void addNoise(float *image, int n_rows, int n_cols, float noise_level = 0.05f);
void translateImage(float *image, int n_rows, int n_cols, int translationCoefficient = 3);
void scaleImage(float *image, int n_rows, int n_cols, float scale_coefficient = 0.15f);



#endif
