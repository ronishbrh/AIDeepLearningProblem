#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <intrin.h>
#include <opencv2/opencv.hpp>
#include <algorithm>

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
    magic_number = _byteswap_ulong(magic_number);

    // cout << magic_number << endl;
    assert(magic_number == 0x803);

    int number_of_data;
    file.read(reinterpret_cast<char *>(&number_of_data), 4);
    number_of_data = _byteswap_ulong(number_of_data);
    cout << number_of_data << endl;

    int height;
    file.read(reinterpret_cast<char *>(&height), 4);
    height = _byteswap_ulong(height);
    cout << height << endl;
    assert(height == 28);

    int width;
    file.read(reinterpret_cast<char *>(&width), 4);
    width = _byteswap_ulong(width);
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
    magic_number = _byteswap_ulong(magic_number);

    assert(magic_number == 0x801);

    int number_of_data;
    file.read(reinterpret_cast<char *>(&number_of_data), 4);
    number_of_data = _byteswap_ulong(number_of_data);

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



std::vector<float> prepare_input_image(const std::string& image_path) {
  cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
      std::cerr << "Error: Could not load image from " << image_path << std::endl;
      return {};
  }

  //pixel values to the range [0.0, 1.0]
  img.convertTo(img, CV_32F, 1.0 / 255.0);

  std::vector<float> input_data(784); 
  for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
          input_data[i * 28 + j] = img.at<float>(i, j);
      }
  }

  return input_data;
}

int main() {
  std::vector<std::vector<std::vector<float>>> weights;
  std::vector<std::vector<float>> biases;

  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> labels;

  read_wts_biases("wts_biases.bin", weights, biases);

  //read_images_from_emnist("dataset/gzip/emnist-digits-test-images-idx3-ubyte", images);
  //read_labels_from_emnist("dataset/gzip/emnist-digits-test-labels-idx1-ubyte", labels);

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

//taking input start........yaha bata
  cv::VideoCapture cap(1); 
  if (!cap.isOpened()) {
      std::cerr << "Error: Cannot open the webcam!" << std::endl;
      return -1;
  }

  cv::Mat frame, gray, binary;
  cv::Mat croppedImage;  


  while (true) {
      cap >> frame; //take each webcam's frame
      if (frame.empty()) {
          std::cerr << "Error: Blank frame grabbed!" << std::endl;
          break;
      }

      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

      cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

      cv::Rect boundingBox;
      for (size_t i = 0; i < contours.size(); i++) {
          cv::Rect tempBoundingBox = cv::boundingRect(contours[i]);
          if (tempBoundingBox.area() > boundingBox.area()) {
              boundingBox = tempBoundingBox;  //the largest bounding box
          }
      }

      if (boundingBox.area() > 0) { 
          croppedImage = frame(boundingBox); //crop garyo
      }


      cv::imshow("Cropped Digit", croppedImage);

      int key = cv::waitKey(1);

      if (key == 13) {  //Enter key
          if (!croppedImage.empty()) {
              std::string outputFilename = "cropped_digit.png"; 
              bool isSaved = cv::imwrite(outputFilename, croppedImage); 
              if (isSaved) {
                  std::cout << "Cropped image saved as " << outputFilename << std::endl;

                  cv::Mat grayscaleCropped;
                  cv::cvtColor(croppedImage, grayscaleCropped, cv::COLOR_BGR2GRAY);
                  cv::imshow("Grayscale Image", grayscaleCropped);


                  key = cv::waitKey(0);

                  //CLAHE(Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
                  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                  clahe->setClipLimit(3.0); 
                  cv::Mat contrastEnhanced;
                  clahe->apply(grayscaleCropped, contrastEnhanced);
                  cv::imshow("CLAHE Enhanced Image", contrastEnhanced);

    
                  key = cv::waitKey(0);

                  //binary thresholding to get a black digit on a white background
                  cv::Mat thresholdedImage;
                  cv::threshold(contrastEnhanced, thresholdedImage, 100, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
                  cv::imshow("Thresholded Image (white on black)", thresholdedImage);

      
                  key = cv::waitKey(0);


                  cv::Mat downscaledImage;
                  cv::resize(thresholdedImage, downscaledImage, cv::Size(28, 28), 0, 0, cv::INTER_AREA);
                  cv::imshow("Downscaled Image", downscaledImage);


                  key = cv::waitKey(0);

                  //save the downscaled image
                  std::string downscaledFilename = "downscaled_digit.png";
                  bool isDownscaledSaved = cv::imwrite(downscaledFilename, downscaledImage);
                  if (isDownscaledSaved) {
                      std::cout << "Downscaled image saved as " << downscaledFilename << std::endl;
                  } else {
                      std::cerr << "Error: Could not save the downscaled image!" << std::endl;
                  }


                  std::vector<float> input_data = prepare_input_image(downscaledFilename);

                  auto result = custom_test(input_data, 784, 10, weights, biases, 2, 16);


                  int predicted_digit = std::distance(result.begin(), std::max_element(result.begin(), result.end()));
                  std::cout << "Predicted digit: " << predicted_digit << std::endl;

                  key = cv::waitKey(0);

                  cv::Mat image = cv::Mat::zeros(500, 500, CV_8UC3);
                  std::string text = "Predicted digit: " + std::to_string(predicted_digit);
    
                  int font = cv::FONT_HERSHEY_SIMPLEX;
                  double fontScale = 1.5;
                  int thickness = 2;
                  cv::Scalar color(255, 255, 255); 
                  
                  cv::Point position(50, 250);
                  
                  cv::putText(image, text, position, font, fontScale, color, thickness);

                  cv::imshow("Predicted Digit", image);

                  key = cv::waitKey(0);
              } else {
                  std::cerr << "Error: Could not save the cropped image!" << std::endl;
              }
          } else {
              std::cerr << "No cropped image to save!" << std::endl;
          }

          if (key == 27)//ESC to exit webcam ko window
          break;
      }
      if (key == 27)
          break;
      
  }

  cap.release(); 
  cv::destroyAllWindows();  

  return 0;
}
