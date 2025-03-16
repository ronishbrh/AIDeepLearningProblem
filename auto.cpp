#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load the image
    Mat img = imread("digit.png", IMREAD_COLOR);
    if (img.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Step 1: Convert the image to grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Step 2: Apply binary thresholding
    Mat thresh;
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV + THRESH_OTSU); // OTSU for automatic thresholding

    // Step 3: Find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Step 4: Find the bounding box for the largest contour (digit)
    Rect boundingBox;
    for (size_t i = 0; i < contours.size(); i++) {
        Rect currentBoundingBox = boundingRect(contours[i]);
        
        // Filter out small contours (noise), assuming the digit is the largest contour
        if (currentBoundingBox.area() > 100) { // You can adjust the area threshold as needed
            boundingBox = currentBoundingBox;
        }
    }

    // Step 5: Crop the image using the bounding box
    Mat croppedImage = img(boundingBox);

    // Show the cropped image
    imshow("Cropped Digit", croppedImage);
    waitKey(0);

    return 0;
}

