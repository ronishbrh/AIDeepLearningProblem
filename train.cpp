#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

// Neural Network Parameters
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;
const int HIDDEN_SIZE = 2;
const int OUTPUT_SIZE = 10;
const float LEARNING_RATE = 0.01f;
const int EPOCHS = 20;

// Activation function (Sigmoid)
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Derivative of sigmoid
float sigmoid_derivative(float x) { return x * (1 - x); }

// Initialize weights with random values
void initialize_weights(std::vector<std::vector<float>> &weights, int rows,
                        int cols) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  weights.clear();
  weights.resize(rows, std::vector<float>(cols));
  for (auto &row : weights)
    for (auto &w : row)
      w = dist(rng);
}

void custom_train(unsigned int number_data,
                  std::vector<std::vector<float>> &training_data,
                  unsigned int input_size,
                  std::vector<std::vector<float>> &output_data,
                  unsigned int output_size,
                  std::vector<std::vector<std::vector<float>>> &weights,
                  std::vector<std::vector<float>> &biases,
                  unsigned int hidden_layers_number,
                  unsigned int hidden_layer_size) {

  for (int t = 0; t < number_data; t++) {
    auto output = training_data[t];

    std::vector<std::vector<float>> outputs(
        hidden_layers_number, std::vector<float>(hidden_layer_size));
    outputs.push_back(std::vector<float>(output_size));

    // feed forward
    for (int i = 0; i < hidden_layers_number; i++) {
      for (int j = 0; j < hidden_layer_size; j++) {
        int k;
        if (i == 0)
          k = input_size;
        else
          k = hidden_layer_size;

        outputs[i][j] = biases[i][j];
        while (k--)
          outputs[i][j] += weights[i][j][k] * output[k];
        outputs[i][j] = sigmoid(outputs[i][j]);
      }
      output = outputs[i];
    }

    for (int j = 0; j < output_size; j++) {
      int k = hidden_layer_size;
      outputs[hidden_layers_number][j] = biases[hidden_layers_number][j];
      while (k--)
        outputs[hidden_layers_number][j] +=
            weights[hidden_layers_number][j][k] * output[k];
      outputs[hidden_layers_number][j] =
          sigmoid(outputs[hidden_layers_number][j]);
    }

    // Compute loss
    for (int i = 0; i < output_size; i++) {
      outputs[hidden_layers_number][i] =
          sigmoid_derivative(outputs[hidden_layers_number][i]) *
          (output_data[t][i] - outputs[hidden_layers_number][i]);
      biases[hidden_layers_number][i] +=
          LEARNING_RATE * outputs[hidden_layers_number][i];
    }

    for (int i = hidden_layers_number; i > 0; i--) {
      for (int j = 0; j < hidden_layer_size; j++) {
        float sum = 0;
        int k;
        if (i == hidden_layers_number)
          k = output_size;
        else
          k = hidden_layer_size;
        while (k--) {
          sum += weights[i][k][j] * outputs[i][k];
          weights[i][k][j] += LEARNING_RATE * outputs[i - 1][j] * outputs[i][k];
        }
        outputs[i - 1][j] = sum * sigmoid_derivative(outputs[i - 1][j]);
        biases[i - 1][j] += LEARNING_RATE * outputs[i - 1][j];
      }
    }

    for (int j = 0; j < input_size; j++) {
      int k = hidden_layer_size;
      while (k--) {
        weights[0][k][j] += LEARNING_RATE * training_data[t][j] * outputs[0][k];
      }
    }
  }
}

std::vector<float> custom_test(std::vector<float> testing_data, unsigned int input_size, unsigned int output_size, std::vector<std::vector<std::vector<float>>> &weights, std::vector<std::vector<float>> biases, unsigned int hidden_layers_number, unsigned int hidden_layer_size) {

  auto output1 = testing_data;
  std::vector<float> output2(hidden_layer_size, 0);

  // feed forward
  for (int i = 0; i < hidden_layers_number; i++) {
    for (int j = 0; j < hidden_layer_size; j++) {
      int k;
      if (i == 0)
        k = input_size;
      else
        k = hidden_layer_size;

      output2[j] = biases[i][j];
      while (k--)
        output2[j] += weights[i][j][k] * output1[k];
      output2[j] = sigmoid(output2[j]);
    }
    output1 = output2;
  }

  output2.resize(output_size);
  for (int j = 0; j < output_size; j++) {
    int k = hidden_layer_size;
    output2[j] = biases[hidden_layers_number][j];
    while (k--)
      output2[j] += weights[hidden_layers_number][j][k] * output1[k];
    output2[j] = sigmoid(output2[j]);
  }

  return output2;
}

void load_from_csv(const std::string &name,
                   std::vector<std::vector<float>> &labels,
                   std::vector<std::vector<float>> &data) {
  std::ifstream file(name); // Open the CSV file
  std::string line;

  // Skip the header line
  std::getline(file, line);

  // Read each line (representing one image's data)
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value;
    std::vector<float> pixels; // Vector to store pixel values for one image

    // Read the label (digit) but we don't need to store it for now
    std::getline(ss, value, ','); // Read the label (digit)
    int n = std::stoi(value);
    labels.push_back(std::vector<float>(10, 0.0f));
    labels[labels.size() - 1][n] = 1.0f;

    // Read the pixel values (784 pixels for a 28x28 image)
    while (std::getline(ss, value, ',')) {
      pixels.push_back(std::stoi(value) /
                       255.0f); // Convert string to int and store
    }

    // Add the pixel vector for the current image to the images vector
    data.push_back(pixels);
  }
  file.close(); // Close the file
}

void print_vec_result(std::vector<float> &v) {
  for (float e : v)
    std::cout << ((e > 0.5) ? 1 : 0) << " ";
  std::cout << std::endl;
}

void save_wts_biases(const std::string &file_name,
                     std::vector<std::vector<std::vector<float>>> &weights,
                     std::vector<std::vector<float>> &biases) {
  std::ofstream file(file_name, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening the file" << std::endl;
    return;
  }

  int layer_number = weights.size();
  file.write(reinterpret_cast<char *>(&layer_number), sizeof(int));
  int input_size = weights[0][0].size();
  file.write(reinterpret_cast<char *>(&input_size), sizeof(int));

  for (int i = 0; i < layer_number; i++) {
    int layer_size = weights[i].size();
    file.write(reinterpret_cast<char *>(&layer_size), sizeof(int));

    int prev_layer_size = weights[i][0].size();

    for (int j = 0; j < layer_size; j++) {
      file.write(reinterpret_cast<char *>(weights[i][j].data()),
                 sizeof(float) * prev_layer_size);
    }
    file.write(reinterpret_cast<char *>(biases[i].data()),
               sizeof(float) * layer_size);
  }

  file.close();
}

void read_wts_biases(const std::string &file_name,
                     std::vector<std::vector<std::vector<float>>> &weights,
                     std::vector<std::vector<float>> &biases) {
  std::ifstream file(file_name, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening the file" << std::endl;
    return;
  }

  int layer_number;
  file.read(reinterpret_cast<char *>(&layer_number), sizeof(int));

  int input_size;
  file.read(reinterpret_cast<char *>(&input_size), sizeof(int));
  int prev_layer_size = input_size;

  int layer_size;
  for (int i = 0; i < layer_number; i++) {

    file.read(reinterpret_cast<char *>(&layer_size), sizeof(int));
    weights.push_back(std::vector<std::vector<float>>(0));
    biases.push_back(std::vector<float>(layer_size, 0));
    for (int j = 0; j < layer_size; j++) {
      weights[i].push_back(std::vector<float>(prev_layer_size, 0));
      file.read(reinterpret_cast<char *>(weights[i][j].data()),
                sizeof(float) * prev_layer_size);
    }
    file.read(reinterpret_cast<char *>(biases[i].data()),
              sizeof(float) * layer_size);
    prev_layer_size = layer_size;
  }

  file.close();
}

bool read_images_from_emnist(const std::string &filename, std::vector<std::vector<float>> &images){
	using namespace std;
	ifstream file(filename);
	if(file.is_open()){
		int magic_number;
		file.read(reinterpret_cast<char*>(&magic_number), 4);
		magic_number = __builtin_bswap32(magic_number);

		//cout << magic_number << endl;
		assert(magic_number == 0x803);

		int number_of_data;
		file.read(reinterpret_cast<char*>(&number_of_data), 4);
		number_of_data = __builtin_bswap32(number_of_data);
		cout << number_of_data << endl;

		int height;
		file.read(reinterpret_cast<char*>(&height), 4);
		height = __builtin_bswap32(height);
		cout << height << endl;
		assert(height == 28);

		int width;
		file.read(reinterpret_cast<char*>(&width), 4);
		width = __builtin_bswap32(width);
		cout << width << endl;
		assert(width == 28);

		for(int i = 0; i<number_of_data; i++){
			images.push_back(vector<float>(height*width));
			for(int j = 0; j<width*height; j++){
				unsigned char pixel;
				file.read(reinterpret_cast<char*>(&pixel), 1);
				images[i][j/width + (j%width)*height] = pixel/255.0f;
			}
		}

		file.close();
		return true;
	} else {
		cout << "Couldn't open the file." << endl;
		return false;
	}

}

bool read_labels_from_emnist(const std::string &filename, std::vector<std::vector<float>> &labels){
	using namespace std;
	ifstream file(filename);
	if(file.is_open()){
		int magic_number;
		file.read(reinterpret_cast<char*>(&magic_number), 4);
		magic_number = __builtin_bswap32(magic_number);

		assert(magic_number == 0x801);

		int number_of_data;
		file.read(reinterpret_cast<char*>(&number_of_data), 4);
		number_of_data = __builtin_bswap32(number_of_data);

		for(int i = 0; i<number_of_data; i++){
			unsigned char num;
			file.read(reinterpret_cast<char*>(&num), 1);
			labels.push_back({0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f});
			labels[i][num] = 1.0f;
		}

		file.close();
		return true;
	} else {
		cout << "Couldn't open the file." << endl;
		return false;
	}

}

int main() {
  std::vector<std::vector<float>> training_data;
  std::vector<std::vector<float>> labels;
  std::vector<std::vector<float>> biases(2, std::vector(16, 0.0f));
  biases.push_back(std::vector<float>(10, 0.0f));
  std::vector<std::vector<std::vector<float>>> weights;

  std::vector<std::vector<float>> i;
  initialize_weights(i, 16, 784);
  weights.push_back(i);
  initialize_weights(i, 16, 16);
  weights.push_back(i);
  initialize_weights(i, 10, 16);
  weights.push_back(i);

  load_from_csv("dataset/train.csv", labels, training_data);
  read_labels_from_emnist("dataset/gzip/emnist-digits-train-labels-idx1-ubyte", labels);
  read_images_from_emnist("dataset/gzip/emnist-digits-train-images-idx3-ubyte", training_data);

  for (int i = 0; i < 50; i++)
    custom_train(labels.size(), training_data, 784, labels, 10, weights, biases,
                 2, 16);

  save_wts_biases("wts_biases.bin", weights, biases);

  std::vector<std::vector<std::vector<float>>> weights2;
  std::vector<std::vector<float>> biases2;

  read_wts_biases("wts_biases.bin", weights2, biases2);

  assert(weights.size() == weights2.size());
  assert(biases.size() == biases2.size());

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < biases[i].size(); j++) {
      //std::cout << biases[i][j] << "\t" << biases2[i][j] << std::endl;
      assert(biases[i][j] == biases2[i][j]);
    }
  }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < weights[i].size(); j++)
      for (int k = 0; k < weights[i][j].size(); k++)
        assert(weights[i][j][k] == weights2[i][j][k]);
  }

  return 0;
}
