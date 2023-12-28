#ifndef W5_EX2_HPP
#define W5_EX2_HPP

#include <string>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// Define data paths
namespace AppConfig {
    const std::string PATH_TO_K = "../../../../resources/pnp/K.txt";
    const std::string PATH_TO_POINTS_3D = "../../../../resources/pnp/points_3d.txt";
    const std::string PATH_TO_POINTS_2D = "../../../../resources/pnp/points_2d.txt";
}

// Function declaration for readMatrix
Eigen::MatrixXd readMatrix(const std::string& filename, int& rows, int& cols);

// Define the LoadedData struct to hold all necessary matrices for PnP problem
struct LoadedData {
    Eigen::MatrixXd K;
    Eigen::MatrixXd K_inverse;
    Eigen::MatrixXd points_3d;
    Eigen::MatrixXd points_3d_h;                            // to get access to homogenious data
    std::vector<Eigen::MatrixXd> points_2d_per_pose;        // A vector of matrices, one for each camera pose
    std::vector<Eigen::MatrixXd> normalized_points_2d_h;    // to get access normalized data
};

// Functions declaration
LoadedData loadPnPData(const std::string & path_to_K, const std::string & path_to_points_3d, const std::string & path_to_points_2d);
Eigen::MatrixXd convert3DToHomogeneous(const Eigen::MatrixXd& points_3d);
Eigen::MatrixXd convertToHomogeneous2D(const Eigen::MatrixXd& points);
Eigen::MatrixXd normalize2DPoints(const Eigen::MatrixXd& points_2d, const Eigen::MatrixXd& K_inverse);
Eigen::MatrixXd DLT(const std::vector<Eigen::Vector4d>& points_3D_h, const std::vector<Eigen::Vector3d>& points_2D_n);
std::pair<Eigen::Matrix3d, Eigen::Vector3d> extractCameraPose(const Eigen::MatrixXd& P, bool debug);
std::pair<double, double> calculateTotalAndAverageReprojectionError(
    const Eigen::Matrix3d& K,
    const std::vector<Eigen::MatrixXd>& all_projection_matrices,
    const std::vector<Eigen::Vector4d>& points_3D_h,
    const std::vector<Eigen::Vector2d>& points_2D);
void PrintDebugInfo(
    const std::vector<Eigen::Vector4d>& points_3D_h,
    const std::vector<Eigen::MatrixXd>& all_projection_matrices,
    const std::vector<Eigen::Vector2d>& observed_points_2D,
    const Eigen::Matrix3d& K);

#endif // OTHER_HPP