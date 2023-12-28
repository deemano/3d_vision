#include "w5_ex2.hpp"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Geometry> 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdlib.h>  // For _fullpath dirs

// Global file streams (could also be passed as parameters if preferred)
std::ofstream rotation_file("../../../../resources/pnp/rotations.txt", std::ios::app);
std::ofstream translation_file("../../../../resources/pnp/translations.txt", std::ios::app);

// Data Loading & Process
// Read matrix from file and convert to Eigen format
Eigen::MatrixXd readMatrix(const std::string& filename, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(" Unable to open file: " + filename);
    }
    // Vector to store matrix entries
    std::vector<double> matrixEntries;
    std::string line;
    double val;
    // Variables
    rows = 0;
    cols = -1;

    // Read file line by line
    while (std::getline(file, line)) {
        // Replace all commas in the line with spaces
        std::replace(line.begin(), line.end(), ',', ' ');
        // Stream to process each line
        std::istringstream lineStream(line);
        int temp_cols = 0;

        // Read each value in the line
        while (lineStream >> val) {
            matrixEntries.push_back(val);
            ++temp_cols;
        }
        // Set columns # based on the first line, or check consistency
        if (cols == -1) {
            cols = temp_cols;
        }
        else if (temp_cols != cols) {
            throw std::runtime_error(" Inconsistent number of columns in file: " + filename);
        }
        // Increment the row count
        ++rows;
    }
    // Handling if empty or invalid format
    if (rows == 0 || cols == 0) {
        throw std::runtime_error(" Empty or invalid format in file: " + filename);
    }

    // Create and populate Eigen matrix
    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = matrixEntries[i * cols + j];
        }
    }
    // Return the Eigen matrix
    return matrix;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// 1. Loading Data Functions :
LoadedData loadPnPData(const std::string& path_to_K, const std::string& path_to_points_3d, const std::string& path_to_points_2d) {
    LoadedData data;
    int rows, cols, temp = 0;

    // 4DEBUG <<<<
    // Load Camera Matrix (K)
    data.K = readMatrix(path_to_K, rows, cols);
    std::cout << "\n Identified camera matrix K with " << rows << " rows and " << cols << " cols." << std::endl;

    // Error checking right after reading the camera matrix
    if (rows != 3 || cols != 3) {
        throw std::runtime_error(" Camera matrix K must be 3x3, but read a matrix with " +
            std::to_string(rows) + " rows and " + std::to_string(cols) + " cols.");
    }

    // Calculate K_inverse
    data.K_inverse = data.K.inverse();

    // Load 3D Points
    data.points_3d = readMatrix(path_to_points_3d, rows, cols);
    std::cout << "\n Identified 3D points with " << rows << " rows and " << cols << " cols." << std::endl;
    // Convert the 3D points to homogeneous coordinates and store them
    data.points_3d_h = convert3DToHomogeneous(data.points_3d); // Store the homogeneous 3D points

    // Load 2D Points and organize by camera pose
    std::ifstream file(path_to_points_2d);
    if (!file.is_open()) {
        throw std::runtime_error(" Unable to open file: " + path_to_points_2d);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::vector<double> pointEntries;
        double val;

        // Read each value in the line and store in pointEntries
        while (lineStream >> val) {
            pointEntries.push_back(val);
        }

        // Each camera pose has the same number of 2D points (n)
        int num_points = pointEntries.size() / 2; // Since each point has u and v coordinates
        Eigen::MatrixXd points_2d(num_points, 2); // One row for each point, two columns for u and v

        for (int i = 0; i < num_points; ++i) {
            points_2d(i, 0) = pointEntries[2 * i];     // u coordinate
            points_2d(i, 1) = pointEntries[2 * i + 1]; // v coordinate
        }

        data.points_2d_per_pose.push_back(points_2d); // observed 2D points collection for Reprojection Error
    }

    // Normalize 2D points for each camera pose
    for (Eigen::MatrixXd& points_2d : data.points_2d_per_pose) {
        //std::cout << "Number of conversions " << temp++ << '\n'; // to check # of conversions =210
        points_2d = normalize2DPoints(points_2d, data.K_inverse);
        // Homogenization of normalized 2D points
        Eigen::MatrixXd points_2d_h = convertToHomogeneous2D(points_2d);
        data.normalized_points_2d_h.push_back(points_2d_h); // Store the processed points
    }

    std::cout << "\n Identified and normalized 2D points for " << data.points_2d_per_pose.size() << " camera poses." << std::endl;

    return data;
}

// 2. Convert and Normalization Functions :
Eigen::MatrixXd normalize2DPoints(const Eigen::MatrixXd& points_2d, const Eigen::MatrixXd& K_inverse) {
    Eigen::MatrixXd normalized_points(points_2d.rows(), points_2d.cols());
    for (int i = 0; i < points_2d.rows(); ++i) {
        Eigen::Vector3d point_homogeneous(points_2d(i, 0), points_2d(i, 1), 1.0); // Convert to homogeneous coordinates
        Eigen::Vector3d normalized_point = K_inverse * point_homogeneous; // Normalize
        normalized_points.row(i) << normalized_point(0) / normalized_point(2), normalized_point(1) / normalized_point(2); // Convert back to Cartesian coordinates
    }
    return normalized_points;
}

// Convert a set of 3D points to homogeneous coordinates
Eigen::MatrixXd convert3DToHomogeneous(const Eigen::MatrixXd& points_3d) {
    // Initialize a new matrix with an additional column than the input matrix for the homogeneous coordinate
    Eigen::MatrixXd homogeneous_points(points_3d.rows(), points_3d.cols() + 1);

    // Copy the original 3D points into the new matrix
    homogeneous_points.block(0, 0, points_3d.rows(), points_3d.cols()) = points_3d;

    // Set the last column to 1 for all points, which represents the homogeneous coordinate
    homogeneous_points.col(points_3d.cols()).setOnes();

    // Print the resulting homogeneous matrix
    //std::cout << "3D Points in Homogeneous Coordinates:\n" << homogeneous_points << "\n\n";

    return homogeneous_points;
}

// Convert a set of 2D points to homogeneous coordinates
Eigen::MatrixXd convertToHomogeneous2D(const Eigen::MatrixXd& points) {
    // Initialize a new matrix with 3 columns for the 2D points in homogeneous coordinates
    Eigen::MatrixXd homogeneous_points(points.rows(), 3);

    // Fill in the u and v coordinates and set the third coordinate to 1
    homogeneous_points << points, Eigen::VectorXd::Ones(points.rows());

    // Print the resulting homogeneous matrix
    //std::cout << "2D Points in Homogeneous Coordinates:\n" << homogeneous_points << "\n\n";

    return homogeneous_points;
}

// 3. DLT Function with SVD decomposition and Matrices Reshaping in 3x4
Eigen::MatrixXd DLT(const std::vector<Eigen::Vector4d>& points_3D_h, const std::vector<Eigen::Vector3d>& points_2D_n) {
    // Ensure the points are aligned
    assert(points_3D_h.size() == points_2D_n.size());

    // Each correspondence gives us two equations, so the matrix A has the number of rows double the number of points
    Eigen::MatrixXd A(2 * points_3D_h.size(), 12);

    // Constructing a system of linear equations in matrix form
    for (size_t i = 0; i < points_3D_h.size(); ++i) {
        const Eigen::Vector4d& X = points_3D_h[i]; // Homogeneous 3D point
        const Eigen::Vector3d& x = points_2D_n[i]; // Normalized 2D point

        A.row(2 * i) << X.transpose(), Eigen::Vector4d::Zero().transpose(), -x(0) * X.transpose();
        A.row(2 * i + 1) << Eigen::Vector4d::Zero().transpose(), X.transpose(), -x(1) * X.transpose();
    }

    // Perform Singular Value Decomposition (SVD) on A
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(11); // Solution is the last column of V

    // Reshape the solution into a 3x4 matrix
    Eigen::MatrixXd P(3, 4);
    P << h(0), h(1), h(2), h(3),
        h(4), h(5), h(6), h(7),
        h(8), h(9), h(10), h(11);

    return P;
}

// 4. Decomposition Function:
std::pair<Eigen::Matrix3d, Eigen::Vector3d> extractCameraPose(const Eigen::MatrixXd& P, bool debug = false) {
    // Sanity check: Make sure that P is 3x4
    assert(P.rows() == 3 && P.cols() == 4);

    // Extract A and t from P
    Eigen::Matrix3d A = P.block<3, 3>(0, 0); // The rotation component of P
    Eigen::Vector3d t = P.col(3);            // The translation component of P

    // Ensure that the z-component of the translation is positive
    if (t(2) < 0) {
        A = -A; // Invert A to ensure positive z-component
        t = -t; // Invert t as well
    }

    // Perform SVD on A to find the closest R in SO(3)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = U * V.transpose();

    // Enforce a determinant of +1 for R
    if (R.determinant() < 0) {
        V.col(2) *= -1; // Change the sign of the third column of V
        R = U * V.transpose();
    }

    // Calculate the scale factor alpha
    double alpha = A.norm() / R.norm();

    // Scale R by alpha
    R *= alpha;

    // Conditional debug print based on the debug flag
    if (debug) {
        std::cout << "\n Alpha scale factor: " << alpha << std::endl;
        std::cout << " Rotation matrix R (scaled by alpha): \n" << R << std::endl;
        std::cout << " Translation vector t: \n" << t << std::endl;
    }

    // Write rotation and translation to files
    rotation_file << R << std::endl << std::endl;
    translation_file << t.transpose() << std::endl;

    // Return the rotation and translation as a pair
    return { R, t };
}

// 5. Reprojection and Error Calculation Function :
std::pair<double, double> calculateTotalAndAverageReprojectionError(
    const Eigen::Matrix3d& K,
    const std::vector<Eigen::MatrixXd>& all_projection_matrices,
    const std::vector<Eigen::Vector4d>& points_3D_h,
    const std::vector<Eigen::Vector2d>& points_2D) {
    double total_reprojection_error = 0.0;
    int num_poses = all_projection_matrices.size(); // Number of camera poses

    // Iterates over each camera pose
    for (const auto& P : all_projection_matrices) {
        // Extract R and t from the current projection matrix P
        auto [R, t] = extractCameraPose(P, false);

        int num_points = points_3D_h.size();
        double pose_error = 0.0;

        // Calculating reprojection error for each point
        for (int i = 0; i < num_points; ++i) {
            // Project each 3D point into the 2D image plane using K, R, and t
            Eigen::Vector3d projected_point_h = K * (R * points_3D_h[i].head<3>() + t);
            Eigen::Vector2d projected_point = projected_point_h.hnormalized(); // Convert from homogeneous to Cartesian coordinates

            // Calculate the reprojection error for this point
            double error = (projected_point - points_2D[i]).norm();
            pose_error += error;
        }

        // Accumulate the total error
        total_reprojection_error += pose_error;
    }

    // Calculate the average reprojection error per pose
    double average_reprojection_error = total_reprojection_error / num_poses;

    // Return both total and average reprojection errors
    return { total_reprojection_error, average_reprojection_error };
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// 6. DEBUG function - I used this because alpha seems too high, way higher that square root of 3 (1).
void PrintDebugInfo(
    const std::vector<Eigen::Vector4d>& points_3D_h,
    const std::vector<Eigen::MatrixXd>& all_projection_matrices,
    const std::vector<Eigen::Vector2d>& observed_points_2D,
    const Eigen::Matrix3d& K) {

    for (size_t pose_idx = 0; pose_idx < all_projection_matrices.size(); ++pose_idx) {
        const auto& P = all_projection_matrices[pose_idx];
        std::cout << "\nProjection matrix P for pose " << pose_idx + 1 << ":\n" << P << "\n\n";

        auto [R, t] = extractCameraPose(P);
        std::cout << "Rotation matrix R:\n" << R << "\n";
        std::cout << "Translation vector t:\n" << t << "\n\n";

        for (size_t point_idx = 0; point_idx < points_3D_h.size(); ++point_idx) {
            Eigen::Vector4d point_3D_h = points_3D_h[point_idx];
            Eigen::Vector3d projected_point_h = P * point_3D_h;
            Eigen::Vector2d projected_point = projected_point_h.hnormalized();

            std::cout << "Original 2D point: " << observed_points_2D[point_idx].transpose()
                << ", Reprojected 2D point: " << projected_point.transpose() << std::endl;
        }
    }
}