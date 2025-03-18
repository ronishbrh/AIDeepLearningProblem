#include <opencv2/opencv.hpp>
#include <iostream>
#include "run.h"  // Include the header file for custom_test
#include <algorithm>

// Function to prepare the input image
std::vector<float> prepare_input_image(const std::string& image_path) {
    // Load the image in grayscale (1 channel)
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return {};
    }

    // Normalize pixel values to the range [0.0, 1.0]
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // Flatten the image into a 1D vector
    std::vector<float> input_data(784); // 28x28 image
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input_data[i * 28 + j] = img.at<float>(i, j);
        }
    }

    return input_data;
}

int main() {
    // Initialize webcam
    cv::VideoCapture cap(1); // Open the default webcam
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the webcam!" << std::endl;
        return -1;
    }

    cv::Mat frame, gray, binary;
    cv::Mat croppedImage;  // Variable to store the cropped image

    // Declare the variables for weights and biases
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> biases;

    // Load weights and biases from the file
    read_wts_biases("wts_biases.bin", weights, biases);

    while (true) {
        cap >> frame; // Capture a frame from the webcam
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed!" << std::endl;
            break;
        }

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Apply thresholding to get a binary image
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Find the bounding box for the largest contour
        cv::Rect boundingBox;
        for (size_t i = 0; i < contours.size(); i++) {
            cv::Rect tempBoundingBox = cv::boundingRect(contours[i]);
            if (tempBoundingBox.area() > boundingBox.area()) {
                boundingBox = tempBoundingBox;  // Keep the largest bounding box
            }
        }

        if (boundingBox.area() > 0) { // Ensure a valid bounding box is found
            // Crop the digit region (without the boundary)
            croppedImage = frame(boundingBox);  // Just extract the region of interest
        }

        // Show the cropped digit in a separate window
        cv::imshow("Cropped Digit", croppedImage);

        // Wait for a key press
        int key = cv::waitKey(1);

        // Press 'Enter' (key code 13) to save the cropped image
        if (key == 13) {  // Enter key
            if (!croppedImage.empty()) {
                std::string outputFilename = "cropped_digit.png"; // Define the output filename
                cv::Mat croppedImageWithoutBoundary = croppedImage.clone();  // Create a copy of the cropped image
                bool isSaved = cv::imwrite(outputFilename, croppedImageWithoutBoundary); // Save the cropped image without boundary
                if (isSaved) {
                    std::cout << "Cropped image saved as " << outputFilename << std::endl;

                    // Convert the cropped image to grayscale (if it's in color)
                    cv::Mat grayscaleCropped;
                    cv::cvtColor(croppedImageWithoutBoundary, grayscaleCropped, cv::COLOR_BGR2GRAY);

                    // Show the grayscale image
                    cv::imshow("Grayscale Image", grayscaleCropped);

                    // Wait for Enter to go to the next step
                    key = cv::waitKey(0);

                    // Apply CLAHE for contrast enhancement
                    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                    clahe->setClipLimit(2.0); // You can adjust this value
                    cv::Mat contrastEnhanced;
                    clahe->apply(grayscaleCropped, contrastEnhanced);

                    // Show the contrast-enhanced image
                    cv::imshow("CLAHE Enhanced Image", contrastEnhanced);

                    // Wait for Enter to go to the next step
                    key = cv::waitKey(0);

                    // Apply binary thresholding to get a black digit on a white background
                    cv::Mat thresholdedImage;
                    cv::threshold(contrastEnhanced, thresholdedImage, 100, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

                    // Show the thresholded image
                    cv::imshow("Thresholded Image (white on black)", thresholdedImage);

                    // Wait for Enter to go to the next step
                    key = cv::waitKey(0);

                    // Now downscale the thresholded image to 28x28
                    cv::Mat downscaledImage;
                    cv::resize(thresholdedImage, downscaledImage, cv::Size(28, 28), 0, 0, cv::INTER_AREA);

                    // Show the downscaled image
                    cv::imshow("Downscaled Image", downscaledImage);

                    // Wait for Enter to save the image
                    key = cv::waitKey(0);

                    // Save the downscaled image as another file
                    std::string downscaledFilename = "downscaled_digit.png";
                    bool isDownscaledSaved = cv::imwrite(downscaledFilename, downscaledImage);
                    if (isDownscaledSaved) {
                        std::cout << "Downscaled image saved as " << downscaledFilename << std::endl;
                    } else {
                        std::cerr << "Error: Could not save the downscaled image!" << std::endl;
                    }

                    // Prepare input data for custom_test
                    std::vector<float> input_data = prepare_input_image(downscaledFilename);

                    // Perform custom test to get the predicted digit
                    auto result = custom_test(input_data, 784, 10, weights, biases, 2, 16);

                    // Find the predicted digit
                    int predicted_digit = std::distance(result.begin(), std::max_element(result.begin(), result.end()));
                    std::cout << "Predicted digit: " << predicted_digit << std::endl;
                } else {
                    std::cerr << "Error: Could not save the cropped image!" << std::endl;
                }
            } else {
                std::cerr << "No cropped image to save!" << std::endl;
            }
        }

        // Press 'ESC' to exit
        if (key == 27)
            break;
    }

    cap.release();  // Release the webcam
    cv::destroyAllWindows();  // Close all OpenCV windows

    return 0;
}

#ifndef RUN_H
#define RUN_H

#include <vector>

// Function prototype for custom_test
std::vector<float> custom_test(const std::vector<float>& input_data, int input_size, int output_size,
                                const std::vector<std::vector<std::vector<float>>>& weights,
                                const std::vector<std::vector<float>>& biases,
                                int hidden_layers_number, int hidden_layer_size);

// Function prototype for reading weights and biases from file
void read_wts_biases(const std::string &file_name,
                     std::vector<std::vector<std::vector<float>>> &weights,
                     std::vector<std::vector<float>> &biases);

#endif // RUN_H
