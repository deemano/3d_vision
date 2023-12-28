// Include all file structure:
#include "other.hpp"
#include "w5_ex2.hpp"
#include "camera_calibration.hpp"

// Include other dependencies
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <Eigen/Core>
#include <vector>
#include <cstdlib> // For system()
#include <chrono>
#include <thread>

// Forward declaration
void exercise2();

// Main Loop
int main() {
    try {
        // Exercise 1 - Execute camera calibration 
        camCalibration(); // << Uncomment line to execute

        // Exercise 2 - Perspective-n-Point PnP. 
        // Here the data gets exported for plotting & generate video via plottingToVideo.py script
        exercise2();

        // Optional - Play the plotting video
        system("start ../../../../camera_motion_cam_Zaxis_perspective.mp4");
    } 
    // exception handling
    catch (const std::exception& e) {
        std::cerr << " An error occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// Exercise 2 Function
void exercise2() {
    std::cout << "\n Week 5 Portfolio - Exercise 2 (runtime: ~1 min):\n";
    std::this_thread::sleep_for(std::chrono::seconds(10)); // short pause to separate exercises
    // Load data using predefined paths from AppConfig
    // This function also does 2D points Normalize and Re-Homogenize + Homogenize 3D points
    LoadedData data = loadPnPData(AppConfig::PATH_TO_K, AppConfig::PATH_TO_POINTS_3D, AppConfig::PATH_TO_POINTS_2D);

    // For Debug - Re-verifying the data alignment
    std::cout << "\n 3D Points in Homogeneous Coordinates:" << std::endl;
    std::cout << data.points_3d_h << std::endl << std::endl;

    /* // Print the 2D points in normalized homogeneous coordinates for each pose (this takes a bit of resources+time)
    std::cout << "\n 2D Points in Normalized Homogeneous Coordinates for each pose:" << std::endl;
    for (const auto& points_2d_h : data.normalized_points_2d_h) {
        std::cout << points_2d_h << std::endl << std::endl;
    } // */

    // Prepare vectors for DLT
    std::vector<Eigen::Vector4d> points_3D_h(data.points_3d_h.rows());
    for (int j = 0; j < data.points_3d_h.rows(); ++j) {
        points_3D_h[j] = data.points_3d_h.row(j);
    }

    std::vector<Eigen::MatrixXd> all_projection_matrices;   // Define a vector to store all projection matrices
    std::vector<Eigen::Vector2d> observed_points_2D;        // To store flattened 2D points

    // Flatten all the 2D points into a single vector for error calculation
    for (const Eigen::MatrixXd& pose_points_2d : data.points_2d_per_pose) {
        for (int i = 0; i < pose_points_2d.rows(); ++i) {
            Eigen::Vector2d point(pose_points_2d(i, 0), pose_points_2d(i, 1));
            observed_points_2D.push_back(point);
        }
    }

    // Now iterate over each camera pose to process the 2D points
    for (size_t i = 0; i < data.points_2d_per_pose.size(); ++i) {
        std::vector<Eigen::Vector3d> points_2D_n(data.normalized_points_2d_h[i].rows());
        for (int j = 0; j < data.normalized_points_2d_h[i].rows(); ++j) {
            points_2D_n[j] = data.normalized_points_2d_h[i].row(j);
        }

        // Call DLT for each pose using the same 3D points and the new 2D points
        Eigen::MatrixXd P = DLT(points_3D_h, points_2D_n);
        all_projection_matrices.push_back(P); // Store the projection matrix for each pose

        // Print out the projection matrix for each camera pose
        std::cout << "\n Projection matrix P for pose " << i + 1 << ":\n" << P;

        // Extract the camera pose from P
        auto [R, t] = extractCameraPose(P, true);
    }
    // Call the Debug print function, if necessary
    //PrintDebugInfo(points_3D_h, all_projection_matrices, observed_points_2D, data.K);


    // Calculate total and average reprojection errors
    auto [total_error, avg_error] = calculateTotalAndAverageReprojectionError(
        data.K, all_projection_matrices, points_3D_h, observed_points_2D);

    // Print the errors to verify
    for (int i = 0; i < 25; i++) {std::cout << " _";}
    std::cout << "\n\n Total reprojection error: " << total_error << std::endl;
    std::cout << " Average reprojection error per pose: " << avg_error << std::endl;
}