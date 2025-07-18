#include <iostream>
#include <random>
#include <cmath>

#include "../hppFiles/ReadData.hpp"

//Reverses the byte order of an integer (from little indian to big indian or from big indian to little indian)
int reverseInt(int i){
  unsigned char c1, c2, c3, c4;
  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

//Initializes idx files
void initializeFile(std::ifstream &file, uint8_t &type, uint8_t &n_dimension, int &size, int* &dimension){

	uint8_t zero_padding_0;
	uint8_t zero_padding_1;

	file.read((char*)&zero_padding_0, 1);
	file.read((char*)&zero_padding_1, 1);

  //First two bytes should always be 0 for idx files
	if(zero_padding_0 != 0 || zero_padding_1 != 0){
		std::cerr << "Idx file format wrong (missing 0 paddings)" << std::endl;
	}

	file.read((char*)&type, 1);
	file.read((char*)&n_dimension, 1);

	file.read((char*)&size, 4);
	size = reverseInt(size);
	
  if(dimension == nullptr){
	  dimension = new int[(int) n_dimension];
    dimension[0] = 1;

	  for(int i = 1; i < (int) n_dimension; i++){
		  file.read((char*)&dimension[i], 4); 
		  dimension[i] = reverseInt(dimension[i]);
	  }
  }else{
    file.seekg(4 * ((int) (n_dimension - 1)), std::ios::cur);
  }
}

//Reads the next image
void readImage(std::ifstream &file, int* dimension, uint8_t n_dimension, uint8_t type, float *input, bool augumentation){

	int dim = 1;

	for(int i = 0; i < (int) n_dimension; i++){
		dim *= (int) dimension[i];
	}

	switch(type){
		case 8:
			for(int i = 0; i < 784; i++){
				uint8_t temp = 0;
        file.read((char*)&temp, 1);
        float num = temp / 255.0f ;
        input[i] = num;
			}
			break;
		case 9:
			break;
		case 11:
			break;
    case 12:
			break;
    case 13:
			break;
    case 14:
			break;
		default:
			break;

	}

	//Data augumentation (for images only)
  if((int) n_dimension == 3 && augumentation){
    rotateImage(input, dimension[1], dimension[2]);
		translateImage(input, dimension[1], dimension[2]);
		scaleImage(input, dimension[1], dimension[2]);
		addNoise(input, dimension[1], dimension[2]);
  }
}

void readLabel(std::ifstream &file, int& label){
  unsigned char temp;
  file.read((char*)&temp, 1);
  label = static_cast<int>(temp);
}





















//Rotates input image by a small random angle
void rotateImage(float *image, int n_rows, int n_cols) {

  // Random angle between -10 and 10 degrees
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-10, 10);
  int angle = dist(gen);

  float radians = angle * (M_PI / 180.0f);
  float cos_angle = cos(radians);
  float sin_angle = sin(radians);

  //Compute the center of the image
  float center_x = n_cols / 2.0f;
  float center_y = n_rows / 2.0f;

  float rotated_image[n_rows * n_cols] = {0};

  // Rotate each pixel
  for(size_t y = 0; y < n_rows; ++y) {
    for (size_t x = 0; x < n_cols; ++x) {
      //Calculates pixel coordinates relative to center of image
      float rel_x = x - center_x;
      float rel_y = y - center_y;

      //Apply rotation
      float new_x = cos_angle * rel_x - sin_angle * rel_y + center_x;
      float new_y = sin_angle * rel_x + cos_angle * rel_y + center_y;

      //Check if the new pixel coordinates are within bounds
      if (new_x >= 0 && new_x < n_cols && new_y >= 0 && new_y < n_rows) {
        int new_x_int = (int) new_x;
        int new_y_int = (int) new_y;
        rotated_image[new_y_int * n_cols + new_x_int] = image[y * n_cols + x];
      }
    }
  }

  //Copy the rotated image back to the original image
  for(int i = 0; i < n_rows * n_cols; i++){
    image[i] = rotated_image[i];
  }
}

void translateImage(float *image, int n_rows, int n_cols, int translationCoefficient){

  // Random angle between -10 and 10 degrees
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-translationCoefficient, translationCoefficient);

  //Generate random translations along x and y axis
  int translate_x = dist(gen);  // Random horizontal translation
  int translate_y = dist(gen);  // Random vertical translation

  float translated_image[n_rows * n_cols] = {0};

  for(int y = 0; y < n_rows; ++y) {
    for(int x = 0; x < n_cols; ++x) {
      //Calculate new pixel positions after translation
      int new_x = x + translate_x;
      int new_y = y + translate_y;

      //Check if the new position is within bounds
      if(new_x >= 0 && new_x < n_cols && new_y >= 0 && new_y < n_rows) {
        //Place the pixel in the new position
        translated_image[new_y * n_cols + new_x] = image[y * n_cols + x];
      }
    }
  }

  //Copy the translated image back to the original image
  for(int i = 0; i < n_rows * n_cols; i++){
    image[i] = translated_image[i];
  }
}

// Function to add random noise to an image
void addNoise(float *image, int n_rows, int n_cols, float noise_level) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, noise_level);

    for(int i = 0; i < n_rows * n_cols; ++i) {
        float noise = dist(gen);
        float new_pixel = image[i] + noise;
        image[i] = std::min(std::max(new_pixel, 0.0f), 1.0f);  // Ensure pixel values are in range [0.0, 1.0]
    }
}

void scaleImage(float *image, int n_rows, int n_cols, float scale_coefficient){

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(1.0f - scale_coefficient, 1.0f + scale_coefficient);

  float scale_rnd_coeff = dist(gen);

  float scaledImage[n_rows * n_cols];

  if(scale_rnd_coeff > 1.0f){
    for(int y = 0; y < n_rows; ++y) {
      for(int x = 0; x < n_cols; ++x) {
        //Find the corresponding position in the original image (before scaling)
        float orig_x = x / scale_rnd_coeff;
        float orig_y = y / scale_rnd_coeff;

        // Calculate the surrounding four pixels (for bilinear interpolation)
        int x1 = static_cast<int>(orig_x);
        int y1 = static_cast<int>(orig_y);
        int x2 = std::min(x1 + 1, n_cols - 1); // To prevent out of bounds
        int y2 = std::min(y1 + 1, n_rows - 1); // To prevent out of bounds

        // Calculate the weights for interpolation
        float dx = orig_x - x1;
        float dy = orig_y - y1;

        // Perform bilinear interpolation
        float top_left = image[y1 * n_cols + x1];
        float top_right = image[y1 * n_cols + x2];
        float bottom_left = image[y2 * n_cols + x1];
        float bottom_right = image[y2 * n_cols + x2];

        //Interpolate between the points
        float top = top_left + dx * (top_right - top_left);
        float bottom = bottom_left + dx * (bottom_right - bottom_left);
        scaledImage[y * n_cols + x] = std::min(std::max(top + dy * (bottom - top), 0.0f), 1.0f);
      }
    }
  }else if(scale_rnd_coeff < 1.0f){
		for(int y = 0; y < n_rows; ++y) {
      for(int x = 0; x < n_cols; ++x) {
        //Find the corresponding position in the original image (before scaling)
        float orig_x = x / scale_rnd_coeff;
        float orig_y = y / scale_rnd_coeff;

        //Calculate the surrounding block of pixels (for downsampling)
        int x1 = static_cast<int>(orig_x);
        int y1 = static_cast<int>(orig_y);
        int x2 = std::min(x1 + 1, n_cols - 1); // To prevent out of bounds
        int y2 = std::min(y1 + 1, n_rows - 1); // To prevent out of bounds

        //Average the block of pixels (for downsampling)
        float block_sum = 0.0f;
        int count = 0;

        for(int dy = y1; dy <= y2; ++dy) {
          for(int dx = x1; dx <= x2; ++dx) {
            block_sum += image[dy * n_cols + dx];
            count++;
          }
        }

        //Assign the averaged value to the corresponding pixel in the scaled image
        scaledImage[y * n_cols + x] = std::min(std::max(0.0f, block_sum / count), 1.0f);
      }
    }
  }

  if(scale_rnd_coeff != 1.0f){
    //Copy the translated image back to the original image
    for(int i = 0; i < n_rows * n_cols; i++){
      image[i] = scaledImage[i];
    }
  }
}















