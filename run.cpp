#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Activation function (Sigmoid)
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Derivative of sigmoid
float sigmoid_derivative(float x) { return x * (1 - x); }

bool read_images_from_emnist(const std::string &filename,
                             std::vector<std::vector<float>> &images) {
  using namespace std;
  ifstream file(filename);
  if (file.is_open()) {
    int magic_number;
    file.read(reinterpret_cast<char *>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);

    // cout << magic_number << endl;
    assert(magic_number == 0x803);

    int number_of_data;
    file.read(reinterpret_cast<char *>(&number_of_data), 4);
    number_of_data = __builtin_bswap32(number_of_data);
    cout << number_of_data << endl;

    int height;
    file.read(reinterpret_cast<char *>(&height), 4);
    height = __builtin_bswap32(height);
    cout << height << endl;
    assert(height == 28);

    int width;
    file.read(reinterpret_cast<char *>(&width), 4);
    width = __builtin_bswap32(width);
    cout << width << endl;
    assert(width == 28);

    for (int i = 0; i < number_of_data; i++) {
      images.push_back(vector<float>(height * width));
      for (int j = 0; j < width * height; j++) {
        unsigned char pixel;
        file.read(reinterpret_cast<char *>(&pixel), 1);
        images[i][j / width + (j % width) * height] = pixel / 255.0f;
      }
    }

    file.close();
    return true;
  } else {
    cout << "Couldn't open the file." << endl;
    return false;
  }
}

bool read_labels_from_emnist(const std::string &filename,
                             std::vector<std::vector<float>> &labels) {
  using namespace std;
  ifstream file(filename);
  if (file.is_open()) {
    int magic_number;
    file.read(reinterpret_cast<char *>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);

    assert(magic_number == 0x801);

    int number_of_data;
    file.read(reinterpret_cast<char *>(&number_of_data), 4);
    number_of_data = __builtin_bswap32(number_of_data);

    for (int i = 0; i < number_of_data; i++) {
      unsigned char num;
      file.read(reinterpret_cast<char *>(&num), 1);
      labels.push_back(
          {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
      labels[i][num] = 1.0f;
    }

    file.close();
    return true;
  } else {
    cout << "Couldn't open the file." << endl;
    return false;
  }
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

std::vector<float>
custom_test(std::vector<float> testing_data, unsigned int input_size,
            unsigned int output_size,
            std::vector<std::vector<std::vector<float>>> &weights,
            std::vector<std::vector<float>> biases,
            unsigned int hidden_layers_number, unsigned int hidden_layer_size) {

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

void print_vec_result(std::vector<float> &v) {
  for (float e : v)
    std::cout << ((e > 0.5) ? 1 : 0) << " ";
  std::cout << std::endl;
}

int main() {
  std::vector<std::vector<std::vector<float>>> weights;
  std::vector<std::vector<float>> biases;

  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> labels;

  read_wts_biases("wts_biases.bin", weights, biases);

  read_images_from_emnist("dataset/gzip/emnist-digits-test-images-idx3-ubyte", images);
  read_labels_from_emnist("dataset/gzip/emnist-digits-test-labels-idx1-ubyte", labels);

  int wrong = 0;

  for (int i = 0; i < labels.size(); i++) {
    auto result = custom_test(images[i], 784, 10, weights, biases, 2, 16);
    for (int j = 0; j < 10; j++) {
      if (((result[j] > 0.5f) ? 1.0f : 0.0f) != labels[i][j]) {
        wrong++;
        std::cout << "For image :" << i << std::endl;
        print_vec_result(labels[i]);
        print_vec_result(result);
		continue;
      }
    }
  }

  std::cout << "Total images: " << images.size() << std::endl;
  std::cout << "Total wrong: " << wrong << std::endl;

  return 0;
}
