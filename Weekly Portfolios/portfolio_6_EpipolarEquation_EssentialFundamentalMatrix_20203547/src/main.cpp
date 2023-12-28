// Include other dependencies
// Forward declarations - moved all into common.h
#include "common.h"

// Main Loop
int main() {
    try {
        // Suppress OpenCV informational logs warnings 
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

        // Load images
        cv::Mat img1 = cv::imread("../../../../resources/0001.jpg"); // replace with actual path
        cv::Mat img2 = cv::imread("../../../../resources/0002.jpg"); // replace with actual path
        if (img1.empty() || img2.empty()) {
            std::cerr << "Error loading images" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << " Exercise 2..." << std::endl;

        // Load Data
        auto [pointsLeft, pointsRight, intrinsicMatrix] = loadData();
        std::cout << " Data loaded successfully." << std::endl;

        // Check data loaded
        checkAndPrintData(pointsLeft, pointsRight, intrinsicMatrix);

        // Get Fundamanetal Matrix
        cv::Mat F = findFundamentalMatrix(pointsLeft, pointsRight);
        std::cout << " Fundamental matrix F: \n" << F << std::endl;

        // Compute the Essential Matrix and decompose it
        cv::Mat R1, R2, T;
        cv::Mat E = computeEssentialAndDecompose(F, intrinsicMatrix, R1, R2, T);

        // Perform cheirality check and select the correct R and T
        auto [bestR, bestT] = selectRT(pointsLeft, pointsRight, R1, R2, T, intrinsicMatrix);

        // Validate the determinant of R2
        checkRotationMatrix(R2);
        
        // Create empty vectors to store noisy points
        std::vector<cv::Point2f> emptyNoisyPointsLeft, emptyNoisyPointsRight;
        bool epipolLinesYesOrNo = true; // boolean to set a default to visualize epipolar lines

        // Visualize initial points + re-projected points + epipolar lines without noise
        bool vizEpipolarLines = false; // Change to true to see epipolar lines
        vizCorrespond(img1, img2, pointsLeft, pointsRight, emptyNoisyPointsLeft, emptyNoisyPointsRight, bestR, bestT, intrinsicMatrix, F, false); // change this to "true" to see epipolar lines
        cv::waitKey(5000); // Show exercise 2 point a) results for 5 seconds

        // Set Gaussian Noise Paramaters
        double maxNoiseLevel = 5.0; 
        double noiseStep = 0.1;

        // Call the function and get the results
        NoiseAnalysisResults analysisResults = analyzeNoiseImpact(img1, img2, pointsLeft, pointsRight, intrinsicMatrix, maxNoiseLevel, noiseStep, epipolLinesYesOrNo);

        // Open the generated HTML file in the default browser
        std::string command;
        #if defined(_WIN32) || defined(_WIN64)
        command = "plot.html"; // Windows command
        #endif
        system(command.c_str());
    }
    catch (const std::exception& e) {
        std::cerr << " An error occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// Utility function to open a file and return ifstream
std::ifstream openFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(" Error opening file: " + filename);
    }
    return file;
}

// Load Data function
std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>, cv::Mat> loadData() {
    // Define file paths for the matching points and the intrinsic matrix
    std::string matchesFile1 = "../../../../resources/matches0001.txt";
    std::string matchesFile2 = "../../../../resources/matches0002.txt";
    std::string intrinsicFile = "../../../../resources/intrinsic.txt";

    // Inline function to load matching points
    auto loadMatchingPoints = [](const std::string& filename) {
        std::ifstream file = openFile(filename);
        std::string lineX, lineY;
        std::getline(file, lineX);
        std::getline(file, lineY);

        std::istringstream issX(lineX), issY(lineY);
        std::vector<cv::Point2f> points;
        float x, y;

        // Parse and store points from the file
        while (issX >> x && issY >> y) {
            points.emplace_back(x, y);
        }

        file.close();
        return points;
        };
    // Load matching points for the left and right images
    std::vector<cv::Point2f> pointsLeft = loadMatchingPoints(matchesFile1);
    std::vector<cv::Point2f> pointsRight = loadMatchingPoints(matchesFile2);

    // Load intrinsic matrix from file
    std::ifstream intrinsic = openFile(intrinsicFile);
    cv::Mat intrinsicMatrix(3, 3, CV_32F);
    for (int row = 0; row < 3; ++row) { // Read matrix values from the file
        for (int col = 0; col < 3; ++col) {
            intrinsic >> intrinsicMatrix.at<float>(row, col);
        }
    }
    // Return the loaded data as a tuple
    return std::make_tuple(pointsLeft, pointsRight, intrinsicMatrix);
}

// Function to check and print data loaded from files
void checkAndPrintData(const std::vector<cv::Point2f>& pointsLeft, const std::vector<cv::Point2f>& pointsRight, const cv::Mat& intrinsicMatrix) {
    std::cout << " Data file has " << pointsLeft.size() << " points from matches0001.txt" << std::endl;
    if (!pointsLeft.empty()) {
        std::cout << " The first point is: " << pointsLeft.front() << std::endl;
    }

    std::cout << " Data file has " << pointsRight.size() << " points from matches0002.txt" << std::endl;
    if (!pointsRight.empty()) {
        std::cout << " The first point is: " << pointsRight.front() << std::endl;
    }
    // Print the intrinsic matrix that was loaded
    std::cout << " Intrinsic matrix loaded: \n" << intrinsicMatrix << std::endl;
}

// Function to normalize points
std::tuple<std::vector<cv::Point2f>, cv::Mat> normalizePoints(const std::vector<cv::Point2f>& points) {
    cv::Point2f centroid(0, 0);

    // Calculate the centroid of the points
    for (const auto& p : points) {
        centroid += p;
    }
    centroid *= (1.0 / points.size());
    std::cout << "\n Centroid: " << centroid << std::endl;

    // Calculate the scale factor to normalize the point distances
    float scale = 0;
    for (const auto& p : points) {
        cv::Point2f shifted = p - centroid;
        scale += sqrt(shifted.x * shifted.x + shifted.y * shifted.y);
    }
    scale /= points.size();
    scale = sqrt(2.0) / scale;
    std::cout << " Scale: " << scale << std::endl;

    // Create the transformation matrix
    cv::Mat T = (cv::Mat_<float>(3, 3) << scale, 0, -scale * centroid.x,
        0, scale, -scale * centroid.y,
        0, 0, 1);
    std::cout << " Transformation Matrix T: \n" << T << std::endl;

    // Apply the transformation to each point
    std::vector<cv::Point2f> normalizedPoints;
    for (const auto& p : points) {
        cv::Point2f shifted = p - centroid;
        normalizedPoints.emplace_back(shifted.x * scale, shifted.y * scale);
    }
    // Return the normalized points and the transformation matrix
    return { normalizedPoints, T };
}

// Function to find the Fundamental Matrix using normalized 8-point algorithm
cv::Mat findFundamentalMatrix(const std::vector<cv::Point2f>& pointsLeft, const std::vector<cv::Point2f>& pointsRight) {
     // Create a synthetic F matrix for DEBUG
    /*  cv::Mat F = (cv::Mat_<float>(3, 3) << 0.5, 0.3, 0.2,
        0.1, 0.5, 0.3,
        0.4, 0.2, 0.7); */

    // 1. Normalize the points 
    auto [normalizedPointsLeft, T1] = normalizePoints(pointsLeft);
    auto [normalizedPointsRight, T2] = normalizePoints(pointsRight);

    // 2. Construct the matrix A
    cv::Mat A(normalizedPointsLeft.size(), 9, CV_32F);
    for (size_t i = 0; i < normalizedPointsLeft.size(); ++i) {
        auto& ptLeft = normalizedPointsLeft[i];
        auto& ptRight = normalizedPointsRight[i];

        auto fillRow = [&](int row, float x1, float y1, float x2, float y2) {
            A.at<float>(row, 0) = x2 * x1;
            A.at<float>(row, 1) = x2 * y1;
            A.at<float>(row, 2) = x2;
            A.at<float>(row, 3) = y2 * x1;
            A.at<float>(row, 4) = y2 * y1;
            A.at<float>(row, 5) = y2;
            A.at<float>(row, 6) = x1;
            A.at<float>(row, 7) = y1;
            A.at<float>(row, 8) = 1.0f;
            };
        fillRow(i, ptLeft.x, ptLeft.y, ptRight.x, ptRight.y);
    }

    // Ensure A is of type CV_32F or CV_64F
    A.convertTo(A, CV_32F); 
    std::cout << "\n Matrix A created with size: " << A.rows << "x" << A.cols << " and type: " << A.type() << std::endl;
    //std::cout << "Matrix A: \n" << A << std::endl; // DEBUG ONLY

    // 3. Compute the Fundamental Matrix using SVD on A
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    std::cout << " SVD on matrix A. Shapes - W: " << w.size() << ", U: " << u.size() << ", VT: " << vt.size() << std::endl;

    // 4. Extract the fundamental matrix from the last row of vt
    cv::Mat F = vt.row(8).reshape(0, 3);    // Last row of V^T
    cv::Mat F_initial = F.clone();          // Used to verify difference b/w F_initial & F_constrained

    // 5. Enforce Rank-2 Constraint on F
    cv::Mat w_F, u_F, vt_F;
    cv::SVD::compute(F, w_F, u_F, vt_F);
    std::cout << " SVD on matrix F. Shapes - W: " << w_F.size() << ", U: " << u_F.size() << ", VT: " << vt_F.size() << std::endl;
    
    // Set the smallest singular value to zero
    std::cout << "\n Singular values before rank-2 constraint: " << w_F.t() << std::endl;
    w_F.at<float>(2) = 0; 
    std::cout << " Singular values after rank-2 constraint: " << w_F.t() << std::endl;
    
    // Reconstruct F with the modified singular values
    cv::Mat wDiag_F = cv::Mat::diag(w_F);
    F = u_F * wDiag_F * vt_F; 

    // Reconstruct F and print intermediate products
    cv::Mat intermediate_product = u_F * wDiag_F;
    //std::cout << "Intermediate Product (u_F * wDiag_F): \n" << intermediate_product << std::endl;

    F = intermediate_product * vt_F; // Reconstruct F
    std::cout << "\n Rank-2 constrained Fundamental matrix F:: \n" << F << std::endl;

    cv::Mat F_constrained = F;

    // Debug before Denormilize
    /* std::cout << "T1 size: " << T1.rows << "x" << T1.cols << std::endl;
       std::cout << "T2 size: " << T2.rows << "x" << T2.cols << std::endl;
       std::cout << "F size before denormalization: " << F.rows << "x" << F.cols << std::endl;  */

    // Re-check reconstruction
    std::cout << "\n F_constrained: \n" << F_constrained << std::endl;
    std::cout << "\n F_initial: \n" << F_initial << std::endl;
    cv::Mat F_difference = F_constrained - F_initial;
    std::cout << "\n Difference between initial F and rank-2 constrained F: \n" << F_difference << std::endl;

    // 6. Denormalize F matrix
    F = T2.t() * F * T1;
    std::cout << "\n Denormalized Fundamental matrix F: \n" << F << "\n" << std::endl;

    return F;
}

// Function to compute E, R, and T
cv::Mat computeEssentialAndDecompose(const cv::Mat& F, const cv::Mat& K, cv::Mat& R1, cv::Mat& R2, cv::Mat& T) {
    // Compute the Essential Matrix
    cv::Mat E = K.t() * F * K;
    std::cout << "\n Essential Matrix E:\n" << E << std::endl;

    // Decompose the Essential Matrix to obtain U, W, Vt
    cv::Mat w, u, vt;
    cv::SVD::compute(E, w, u, vt);
    std::cout << "\n Decomposed Essential Matrix.\n U:\n" << u << "\n W:\n" << w << "\n Vt:\n" << vt << std::endl;

    // Ensure the correct determinant of R (+1)
    if (cv::determinant(u) < 0) u = -u;
    if (cv::determinant(vt) < 0) vt = -vt;

    // Create W for R (90-degree rotation matrix)
    cv::Mat W = (cv::Mat_<float>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
    std::cout << "\n W Matrix for Rotation:\n" << W << std::endl;

    // Compute two possible rotations, R1 and R2
    R1 = u * W * vt;
    R2 = u * W.t() * vt;

    // Ensure the determinant of R is +1
    if (cv::determinant(R1) < 0) R1 = -R1;
    if (cv::determinant(R2) < 0) R2 = -R2;

    std::cout << "\n First Rotation Matrix R1:\n" << R1 << std::endl;
    std::cout << "\n Second Rotation Matrix R2:\n" << R2 << std::endl;

    // Extract the translation vector (up to a scale)
    T = u.col(2);
    std::cout << "\n Translation Vector T (up to a scale):\n" << T << std::endl;

    return E;
}

// Function to select the correct R and T from possible combinations
std::pair<cv::Mat, cv::Mat>  selectRT(const std::vector<cv::Point2f>& pointsLeft,
    const std::vector<cv::Point2f>& pointsRight,
    const cv::Mat& R1,
    const cv::Mat& R2,
    const cv::Mat& T,
    const cv::Mat& K) {

    // Define camera matrix for the left camera using the intrinsic matrix K
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_32F);
    std::cout << "\n Left Camera Matrix P1:\n" << P1 << "\n\n";

    // Prepare to store the best rotation and translation
    cv::Mat bestR;
    cv::Mat bestT;
    int bestScore = 0;

    // Check both possible rotations with both possible translations (+T and -T)
    std::vector<cv::Mat> rotations = { R1, R2 };
    for (const auto& R : rotations) {
        for (int sign : {1, -1}) { // Check for T and -T
            // Construct camera matrix for the second camera
            cv::Mat P2 = K * (cv::Mat_<float>(3, 4) <<
                R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2), sign * T.at<float>(0),
                R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2), sign * T.at<float>(1),
                R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2), sign * T.at<float>(2));

            std::cout << " Testing with Rotation:\n" << R << "\n and Translation:\n" << sign * T << "\n\n";
            std::cout << " Right Camera Matrix P2:\n" << P2 << "\n\n";

            // Triangulate points for each correspondence
            cv::Mat points3D(4, pointsLeft.size(), CV_32F);
            for (size_t i = 0; i < pointsLeft.size(); i++) {
                cv::Mat point3D = triangulatePoint(pointsLeft[i], pointsRight[i], P1, P2);
                point3D.copyTo(points3D.col(i));
            }

            // Count the number of points with positive depth for both cameras
            int score = countPositiveDepthPoints(points3D, P1) + countPositiveDepthPoints(points3D, P2);
            std::cout << " Number of points with positive depth: " << score << "\n\n";

            // Select the best rotation and translation based on the score
            if (score > bestScore) {
                bestScore = score;
                bestR = R.clone();
                bestT = sign * T;
            }
        }
    }

    // Output the best rotation matrix and translation vector
    std::cout << " Best Rotation Matrix:\n" << bestR << "\n";
    std::cout << " Best Translation Vector:\n" << bestT << "\n";
    std::cout << " Best Score (Number of points with positive depth): " << bestScore << "\n";

    return { bestR, bestT };
}

// Function to triangulate a point from two views
cv::Mat triangulatePoint(const cv::Point2f& pt1, const cv::Point2f& pt2,
    const cv::Mat& P1, const cv::Mat& P2) {
    // Construct matrices A for triangulation
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);

    // SVD decomposition to solve for X (the 3D point in homogeneous coordinates)
    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // The solution is the last column of vt, normalized (homogeneous coordinates)
    cv::Mat X = vt.row(3).t();
    X = X / X.at<float>(3);

    return X;
}

// Implement a function to count the number of points with positive depth
int countPositiveDepthPoints(const cv::Mat& points3D, const cv::Mat& P) {
    int count = 0;
    for (int i = 0; i < points3D.cols; i++) {
        // Convert to non-homogeneous coordinates
        cv::Mat point3D = points3D.col(i);
        point3D = point3D / point3D.at<float>(3);

        // Transform point to camera coordinates
        cv::Mat pointCam = P * point3D;

        // Check if the point has positive depth
        if (pointCam.at<float>(2) > 0) {
            count++;
        }
    }
    return count;
}

// Function to check the determinant of the rotation matrix
void checkRotationMatrix(const cv::Mat& R) {
    double det = cv::determinant(R);
    std::cout << " Determinant of the Rotation Matrix: " << det << std::endl;
    if (std::abs(det - 1.0) < 1e-6) {
        std::cout << " This rotation matrix is valid." << std::endl;
    }
    else {
        std::cout << " The rotation matrix is NOT valid." << std::endl;
    }
}

// Visualizing the results
void vizCorrespond(const cv::Mat& img1, const cv::Mat& img2,
        const std::vector<cv::Point2f>& pointsLeft,
        const std::vector<cv::Point2f>& pointsRight,
        const std::vector<cv::Point2f>& noisyPointsLeft,
        const std::vector<cv::Point2f>& noisyPointsRight,
        const cv::Mat& R, const cv::Mat& T, const cv::Mat& K,
        const cv::Mat& F, bool vizEpipolarLines) {
    // Ensure the main images are not empty
    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: One or both main images are empty." << std::endl;
        return;
    } // Clone images
    cv::Mat img1Copy = img1.clone();
    cv::Mat img2Copy = img2.clone();

    // Check if R and T matrices are valid for constructing P2
    cv::Mat P2;
    if (!R.empty() && !T.empty() && R.rows == 3 && R.cols == 3 && T.rows == 3 && T.cols == 1) {
        P2 = K * (cv::Mat_<float>(3, 4) <<
            R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2), T.at<float>(0),
            R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2), T.at<float>(1),
            R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2), T.at<float>(2));
    }
    else {
        std::cerr << "Error: Rotation or Translation matrix is empty or has invalid dimensions." << std::endl;
        return;
    }

    // Triangulate points to 3D and project them back
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> projectedPointsRight;
    if (!pointsLeft.empty() && pointsLeft.size() == pointsRight.size()) {
        for (size_t i = 0; i < pointsLeft.size(); i++) {
            cv::Mat point3D = triangulatePoint(pointsLeft[i], pointsRight[i], K, P2);
            points3D.push_back(cv::Point3f(point3D.at<float>(0), point3D.at<float>(1), point3D.at<float>(2)));
        }
        // Manually conver to Rodrigue format - automation creates a nasty BUG!
        cv::Mat rvec;
        cv::Rodrigues(R, rvec); // Convert rotation matrix to rotation vector
        cv::projectPoints(points3D, rvec, T, K, cv::Mat(), projectedPointsRight);
    }
    // Set the position, font, scale, and color for the title text
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.6;             // Slightly smaller text
    int thickness = 2; cv::Scalar fontColor(0, 255, 255);
    int textYPos = img1Copy.rows - 30;  // Y position of the text

    if (vizEpipolarLines) {
        // Draw the points and lines between corresponding points
        for (size_t i = 0; i < pointsLeft.size(); ++i) {
            cv::circle(img1Copy, pointsLeft[i], 5, cv::Scalar(0, 255, 0), -1);
            cv::circle(img2Copy, pointsRight[i], 5, cv::Scalar(0, 255, 0), -1);
            cv::circle(img2Copy, projectedPointsRight[i], 5, cv::Scalar(0, 0, 255), -1);
            cv::line(img2Copy, pointsRight[i], projectedPointsRight[i], cv::Scalar(255, 0, 0));
        }
        // Vector to store epipolar lines corresponding to the points in each image
        std::vector<cv::Vec3f> linesLeft, linesRight;
        cv::computeCorrespondEpilines(pointsRight, 2, F, linesLeft); // For points in the right image
        cv::computeCorrespondEpilines(pointsLeft, 1, F, linesRight); // For points in the left image

        // Draw the epipolar lines on the left image
        for (const auto& line : linesLeft) {
            cv::Point pt1(0, -line[2] / line[1]);
            cv::Point pt2(img1Copy.cols, -(line[2] + line[0] * img1Copy.cols) / line[1]);
            cv::line(img1Copy, pt1, pt2, cv::Scalar(255, 0, 0), 1); // Red lines on the left image
        }

        // Draw the epipolar lines on the right image
        for (const auto& line : linesRight) {
            cv::Point pt1(0, -line[2] / line[1]);
            cv::Point pt2(img2Copy.cols, -(line[2] + line[0] * img2Copy.cols) / line[1]);
            cv::line(img2Copy, pt1, pt2, cv::Scalar(255, 0, 0), 1); // Red lines on the right image
        }

    }
    else { // Making the function adaptable to all visualization cases, fixed values, gaussian and epipolar
        if (!noisyPointsLeft.empty() && !noisyPointsRight.empty()) {
            // Draw the original points in green
            for (size_t i = 0; i < pointsLeft.size(); ++i) {
                cv::circle(img1Copy, pointsLeft[i], 5, cv::Scalar(0, 255, 0), -1);  // Green for original left points
                cv::circle(img2Copy, pointsRight[i], 5, cv::Scalar(0, 255, 0), -1); // Green for original right points
            }

            // Draw the noisy points in red
            for (size_t i = 0; i < noisyPointsLeft.size(); ++i) {
                cv::circle(img1Copy, noisyPointsLeft[i], 5, cv::Scalar(0, 0, 255), -1);     // Red for noisy left points
                cv::circle(img2Copy, noisyPointsRight[i], 5, cv::Scalar(0, 0, 255), -1);    // Red for noisy right points
            }

            // Draw lines from original to noisy points for comparison
            for (size_t i = 0; i < pointsLeft.size(); ++i) {
                cv::line(img1Copy, pointsLeft[i], noisyPointsLeft[i], cv::Scalar(255, 0, 0));   // Line from original to noisy on left image
                cv::line(img2Copy, pointsRight[i], noisyPointsRight[i], cv::Scalar(255, 0, 0)); // Line from original to noisy on right image
            }
        }
        else {
            if (!pointsLeft.empty() && !pointsRight.empty() && pointsLeft.size() == pointsRight.size() && pointsLeft.size() == projectedPointsRight.size()) {
                // Draw the original points in green and lines between corresponding points
                for (size_t i = 0; i < pointsLeft.size(); ++i) {
                    cv::circle(img1Copy, pointsLeft[i], 5, cv::Scalar(0, 255, 0), -1);
                    cv::circle(img2Copy, pointsRight[i], 5, cv::Scalar(0, 255, 0), -1);
                    cv::circle(img2Copy, projectedPointsRight[i], 5, cv::Scalar(0, 0, 255), -1);
                    cv::line(img2Copy, pointsRight[i], projectedPointsRight[i], cv::Scalar(255, 0, 0));
                }
            }
            else {
                std::cout << "Warning: One or more of the point vectors are empty or mismatched in size." << std::endl;
            }
        }   
    }

    // Define titles for the images
    std::string titleLeft = "Original Points - in Green &|| Projected / Noisy - in Red";
    std::string titleRight = "Points: Original - in Green &|| Projected / Noisy - in Red";

    // If visualizing epipolar lines, modify the title accordingly
    if (vizEpipolarLines) {
        titleLeft += " with Epipolar Lines";
        titleRight += " with Epipolar Lines";
    }

    // Add titles to the images
    cv::putText(img1Copy, titleLeft, cv::Point(10, textYPos), fontFace, fontScale, fontColor, thickness);
    cv::putText(img2Copy, titleRight, cv::Point(10, textYPos), fontFace, fontScale, fontColor, thickness);

    // Resize the images for display
    cv::Size newSize(img1Copy.cols / 2, img1Copy.rows / 2); // Adjust the divisor as needed
    cv::Mat resizedImg1, resizedImg2;
    cv::resize(img1Copy, resizedImg1, newSize);
    cv::resize(img2Copy, resizedImg2, newSize);

    // Display the resized images with or without epipolar lines based on the flag
    cv::imshow(vizEpipolarLines ? "Camera Left with Epipolar Lines" : "Camera Left", resizedImg1);
    cv::imshow(vizEpipolarLines ? "Camera Right with Epipolar Lines" : "Camera Right", resizedImg2);
    cv::waitKey(1);
}

// Exercise 2: Point b) - Add Gaussian Noise <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// Function to add Gaussian noise to a vector of 2D points
std::vector<cv::Point2f> addGaussianNoise(const std::vector<cv::Point2f>& points, double sigma) {
    // Check if sigma is valid
    if (sigma <= 0) {
        std::cerr << " Invalid sigma value: " << sigma << ". Sigma must be positive." << std::endl;
        throw std::invalid_argument("Sigma must be positive");
    }
    // Print the sigma value for debugging
    std::cout << "\n\n Sigma value for Gaussian noise: " << sigma << std::endl;

    std::vector<cv::Point2f> noisyPoints;
    // Use random_device to seed the generator for different results across runs
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<double> dist(0.0, sigma);

    // Loop over each point in the points vector + adding random noise
    for (const auto& point : points) {
        cv::Point2f noisyPoint(point.x + dist(generator), point.y + dist(generator));
        // Print the original and noisy point for debugging
        std::cout << " Original point: (" << point.x << ", " << point.y << ") ";
        std::cout << " Noisy point: (" << noisyPoint.x << ", " << noisyPoint.y << ")" << std::endl;
        noisyPoints.push_back(noisyPoint); // Add the noisy point to the noisyPoints vector
    }
    return noisyPoints;
}

// Calculate the ROtation deviation 
double computeRotationDeviation(const cv::Mat& baselineR, const cv::Mat& noisyR) {
    // Check if matrices are valid rotation matrices (3x3 and type CV_32F or CV_64F)
    if (baselineR.size() != cv::Size(3, 3) || noisyR.size() != cv::Size(3, 3) ||
        (baselineR.type() != CV_32F && baselineR.type() != CV_64F) ||
        (noisyR.type() != CV_32F && noisyR.type() != CV_64F)) {
        std::cerr << " Invalid matrix size or type. Matrices must be 3x3 and of type CV_32F or CV_64F." << std::endl;
        return -1; // or throw an exception
    }

    // Check determinants to ensure they are proper rotation matrices
    if (std::abs(cv::determinant(baselineR) - 1.0) > 1e-6 ||
        std::abs(cv::determinant(noisyR) - 1.0) > 1e-6) {
        std::cerr << " Invalid rotation matrix. Determinant must be 1." << std::endl;
        return -1; // or throw an exception
    }

    // Debugging: Print input matrices
    std::cout << " \n Baseline rotation matrix:\n" << baselineR << std::endl;
    std::cout << " \n Noisy rotation matrix:\n" << noisyR << std::endl;

    // Compute the relative rotation matrix
    cv::Mat relativeRotation = baselineR.inv() * noisyR;

    // Debugging: Print relative rotation matrix
    std::cout << " \n Relative rotation matrix:\n" << relativeRotation << std::endl;

    // Ensure the matrix is of type CV_64F
    if (relativeRotation.type() != CV_64F) {
        relativeRotation.convertTo(relativeRotation, CV_64F);
    }

    // Debugging: Print the type of relativeRotation
    std::cout << " Type of relativeRotation matrix: " << relativeRotation.type() << std::endl;

    // Convert to axis-angle representation
    cv::Vec3d rotationVector;
    cv::Rodrigues(relativeRotation, rotationVector); // Converts to rotation vector

    // Debugging: Print rotation vector
    std::cout << " Rotation vector (axis-angle):\n" << rotationVector << std::endl;

    // Debugging: Print rotation vector
    std::cout << " Rotation vector (axis-angle):\n" << rotationVector << std::endl;

    // The magnitude of the rotation vector is the rotation angle in radians
    double angle = cv::norm(rotationVector);

    // Convert angle to degrees for easier interpretation
    angle = angle * 180.0 / CV_PI;

    // Debugging: Print the angle of rotation
    std::cout << " Rotation deviation angle (degrees): " << angle << std::endl;

    return angle;
}

// Calculate the Translation Deviation
double computeTranslationDeviation(const cv::Mat& baselineT, const cv::Mat& noisyT) {
    return cv::norm(baselineT - noisyT);
}

// Analyze the impact of Gaussian noise on Stereo Vision
NoiseAnalysisResults analyzeNoiseImpact(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const std::vector<cv::Point2f>& pointsLeft,
    const std::vector<cv::Point2f>& pointsRight,
    const cv::Mat& intrinsicMatrix,
    double maxNoiseLevel,
    double noiseStep,
    bool vizEpipolarLines) {
    // Open an HTML file for output
    std::ofstream htmlFile("plot.html");

    // Write the initial part of the HTML and JavaScript code
    htmlFile << R"(
<!DOCTYPE html>
<html>
<head>
    <title>Noise Impact Plot</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
<div id="plot" style="width:900px;height:600px;"></div>
<script>
var x = [];
var yR = [];
var yT = [];
var yF = [];
var yReprojError = [];
)";
    // Declaration of T, R1, and R2
    cv::Mat T, R1, R2;

    // Baseline computation
    cv::Mat baselineF = findFundamentalMatrix(pointsLeft, pointsRight);
    cv::Mat baselineE = computeEssentialAndDecompose(baselineF, intrinsicMatrix, R1, R2, T);
    auto [baselineR, baselineT] = selectRT(pointsLeft, pointsRight, R1, R2, T, intrinsicMatrix);

    // Ensure noiseStep is positive to avoid an infinite loop
    if (noiseStep <= 0) {
        std::cerr << " Noise step must be positive to avoid an infinite loop." << std::endl;
        throw std::invalid_argument("noiseStep must be positive");
    }

    NoiseAnalysisResults results;   // Instantiate the Data structure
    std::vector<PlotData> plotData; // Vector to store plot data

    // Loop to incremente the level of Gaussian noise 
    for (double noiseLevel = noiseStep; noiseLevel <= maxNoiseLevel; noiseLevel += noiseStep) {
        auto noisyPointsLeft = addGaussianNoise(pointsLeft, noiseLevel);
        auto noisyPointsRight = addGaussianNoise(pointsRight, noiseLevel);

        std::cout << "Noisy Points Left: " << noisyPointsLeft.size() << " points" << std::endl;
        std::cout << "Noisy Points Right: " << noisyPointsRight.size() << " points" << std::endl;

        // Recompute F with the noisy points
        cv::Mat noisyF = findFundamentalMatrix(noisyPointsLeft, noisyPointsRight);

        cv::Mat noisyE = computeEssentialAndDecompose(noisyF, intrinsicMatrix, R1, R2, T);
        auto [noisyR, noisyT] = selectRT(noisyPointsLeft, noisyPointsRight, R1, R2, T, intrinsicMatrix);

        // Compare noisyR and noisyT with baselineR and baselineT
        double deviationR = computeRotationDeviation(baselineR, noisyR);
        double deviationT = computeTranslationDeviation(baselineT, noisyT);
        double deviationF = cv::norm(noisyF - baselineF, cv::NORM_L2);

        // Calculate the reprojection error
        double reprojectionError = calculateReprojectionError(noisyF, noisyPointsLeft, noisyPointsRight);

        std::cout << "Reprojection Error: " << reprojectionError << std::endl;
        std::cout << "Deviation in R: " << deviationR << std::endl;
        std::cout << "Deviation in T: " << deviationT << std::endl;
        std::cout << "Deviation in F (Frobenius norm): " << deviationF << std::endl;
        std::cout << "Noise Level: " << noiseLevel << std::endl;

        // Store the results for plotting
        plotData.push_back({ noiseLevel, deviationR, deviationT, deviationF, reprojectionError });
        
        std::cout << " \nGaussian Noise Impact Results:\n Noise Level: " << noiseLevel << ", \n Deviation in R: " << deviationR << " degrees, \n Deviation in T: " << deviationT << std::endl;
        
        // Visualize the correspondences with Gaussian noise
        vizCorrespond(img1, img2, pointsLeft, pointsRight, noisyPointsLeft, noisyPointsRight, noisyR, noisyT, intrinsicMatrix, noisyF, false);
        
        // Store results instead of directly visualizing
        results.noisyPointsLeft = noisyPointsLeft;
        results.noisyPointsRight = noisyPointsRight;
        results.noisyR = noisyR;
        results.noisyT = noisyT;
        results.noisyF = noisyF;
    }

    // Write the data points into the HTML file
    for (const auto& data : plotData) {
        htmlFile << "x.push(" << data.noiseLevel << "); ";
        htmlFile << "yR.push(" << data.deviationR << "); ";
        htmlFile << "yT.push(" << data.deviationT << ");\n";
        htmlFile << "yF.push(" << data.deviationF << ");\n";
        htmlFile << "yReprojError.push(" << data.reprojectionError << ");\n";
    }

    // Finalize the JavaScript code and close the HTML
    htmlFile << R"(
var layout = {
  title: 'Impact of Gaussian Noise on Algorithm Performance',
  xaxis: {
    title: 'Gaussian Noise Level (Standard Deviation)',
    showgrid: true,
    zeroline: false
  },
  yaxis: {
    title: 'Error Metrics',
    showline: false
  }
};

var traceR = {
    x: x,
    y: yR,
    mode: 'lines+markers',
    type: 'scatter',
    name: 'Deviation in R',
    line: {shape: 'spline', smoothing: 1.3},
    marker: {size: 5}
};
var traceT = {
    x: x,
    y: yT,
    mode: 'lines+markers',
    type: 'scatter',
    name: 'Deviation in T',
    line: {shape: 'spline', smoothing: 1.3},
    marker: {size: 5}
};
var traceF = {
    x: x,
    y: yF,
    mode: 'lines+markers',
    type: 'scatter',
    name: 'Deviation in F',
    line: {shape: 'spline', smoothing: 1.3},
    marker: {size: 5}
};
var traceReprojError = {
    x: x,
    y: yReprojError,
    mode: 'lines+markers',
    type: 'scatter',
    name: 'Reprojection Error',
    line: {shape: 'spline', smoothing: 1.3},
    marker: {size: 5}
};
var data = [traceR, traceT, traceF, traceReprojError];

Plotly.newPlot('plot', data, layout);
</script>
</body>
</html>
)";
    // Close the HTML file
    htmlFile.close();

    return results;
}

// Calculate Projection Error for Plot
double calculateReprojectionError(const cv::Mat& F,
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2) {
    double totalError = 0.0;
    const int pointCount = static_cast<int>(points1.size());
    cv::Mat F_t = F.t(); // Precompute the transpose of F

    for (int i = 0; i < pointCount; ++i) {
        // Convert points to homogeneous coordinates as cv::Mat
        cv::Mat point1_h = (cv::Mat_<float>(3, 1) << points1[i].x, points1[i].y, 1.0f);
        cv::Mat point2_h = (cv::Mat_<float>(3, 1) << points2[i].x, points2[i].y, 1.0f);

        // Compute the epiline for each point in the other image
        cv::Mat line1 = F_t * point2_h; // Epipolar line in the first image
        cv::Mat line2 = F * point1_h;   // Epipolar line in the second image

        // Compute the distances from the points to the epilines
        float distance1 = static_cast<float>(fabs(point1_h.dot(line1)) / cv::norm(line1.rowRange(0, 2)));
        float distance2 = static_cast<float>(fabs(point2_h.dot(line2)) / cv::norm(line2.rowRange(0, 2)));

        // Add the distances to the total error
        totalError += distance1 + distance2;
    }

    // Return the average reprojection error
    return totalError / (pointCount * 2); // Divided by 2 to get the average per point pair
}