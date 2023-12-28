// Include all file structure:
#include "other.hpp"
#include "camera_calibration.hpp"

// Include other dependencies
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>

void camCalibration() {
    // Suppresing OpenCV parallel libs errors
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // Number of calibration images
    int numberOfImages = 25;

    // Checkerboard dimensions
    int CHECKERBOARD[2]{ 31, 23 }; 

    // Creating vectors to store points for each checkerboard image
    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;

    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++) {
        for (int j = 0; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j, i, 0));
    }

    // Vector to store the pixel coordinates of detected checkerboard corners 
    std::vector<cv::Point2f> corner_pts;
    bool success;

    // Define a variable to store the size of the images
    cv::Size imageSize;

    std::cout << "\n Week 5 Portfolio - Exercise 1 (runtime: ~2 min):\n For these images, corners are detected for 3 out of 25 images. \n Distortion coefficients is calculated at the end ...\n" << std::endl;
    // Loop over all calibration images // chessboard_image_
    for (int i = 1; i <= numberOfImages; i++) {
         cv::Mat img = cv::imread("../../../../resources/images/camera_calib" + std::to_string(i) + ".jpg");
        //cv::Mat img = cv::imread("../../../../resources/images4/chessboard_image_" + std::to_string(i) + ".jpg");
        // Check if the image is successfully loaded
        if (img.empty()) {
            std::cerr << " Failed to load image: camera_calib" << i << ".jpg" << std::endl;
            continue; // Skip this iteration
        }

        // Convert image to grayscale
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        // Store the size of the first image (assuming all images are the same size)
        if (i == 1) {
            imageSize = gray.size();
        }

        // Find the chess board corners
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        // Check for successful corner detection
        if (!success) {
            std::cerr << " Chessboard corners not found for image: camera_calib" << i << ".jpg" << std::endl;
            continue; // Skip this image
        }
        else {
            std::cout << " Chessboard corners successfully found for image: camera_calib" << i << ".jpg" << std::endl;
            // Additional processing here
        }

        // Refining pixel coordinates for given 2d points.
        cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
        cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

        // Storing the 3D points and 2D image points for current image
        objpoints.push_back(objp);
        imgpoints.push_back(corner_pts);
    }

    // Camera matrix
    cv::Mat cameraMatrix, distCoeffs, R, T;

    // Performing camera calibration
    double calibrationError = cv::calibrateCamera(objpoints, imgpoints, imageSize, cameraMatrix, distCoeffs, R, T);

    // Check for successful calibration
    if (calibrationError == -1) {
        std::cerr << " Camera calibration failed." << std::endl;
        return; // Exit the function
    }

    std::cout << " Camera matrix:" << std::endl << cameraMatrix << std::endl;
    std::cout << " Distortion coefficients:" << std::endl << distCoeffs << std::endl;
}
