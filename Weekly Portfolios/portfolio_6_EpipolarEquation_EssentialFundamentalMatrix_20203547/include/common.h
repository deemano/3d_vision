// Include other dependencies
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <cstdlib> 
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Core>
#include <vector>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <string> 
#include <random>


// Hold Data for Gaussian Analysis
struct NoiseAnalysisResults {
    std::vector<cv::Point2f> noisyPointsLeft;
    std::vector<cv::Point2f> noisyPointsRight;
    cv::Mat noisyR;
    cv::Mat noisyT;
    cv::Mat noisyF;
};

// Define a struct to hold the results for ploting
struct PlotData {
    double noiseLevel;
    double deviationR;
    double deviationT;
    double deviationF;
    double reprojectionError;
    double fmQuality;           // Fundamental Matrix Quality for current noise level
    double reconError;          // 3D Reconstruction Error for current noise level
    double detR;                // Determinant of Rotation Matrix for current noise level
    double condNumber;          // Condition Number of Matrix for current noise level
    double normalizedDeviationR; // Normalized Error for R for current noise level
};

// Utility function to open a file and return ifstream
std::ifstream openFile(const std::string& filename);

// Load data
std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point2f>, cv::Mat> loadData();

// Feedback of loaded data
void checkAndPrintData(const std::vector<cv::Point2f>& pointsLeft, const std::vector<cv::Point2f>& pointsRight, const cv::Mat& intrinsicMatrix);

// Function to normalize points
std::tuple<std::vector<cv::Point2f>, cv::Mat> normalizePoints(const std::vector<cv::Point2f>& points);

// Calculate Fundamental Matrix
cv::Mat findFundamentalMatrix(const std::vector<cv::Point2f>& pointsLeft, const std::vector<cv::Point2f>& pointsRight);

// function to compute E, R, and T
cv::Mat computeEssentialAndDecompose(const cv::Mat& F, const cv::Mat& K, cv::Mat& R1, cv::Mat& R2, cv::Mat& T);

// Select the correct Rotation and Translation
std::pair<cv::Mat, cv::Mat>  selectRT(const std::vector<cv::Point2f>& pointsLeft,
    const std::vector<cv::Point2f>& pointsRight,
    const cv::Mat& R1,
    const cv::Mat& R2,
    const cv::Mat& T,
    const cv::Mat& K);

// Gathers data and exectes ploting
void vizCorrespond(const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<cv::Point2f>& originalPointsLeft,
    const std::vector<cv::Point2f>& originalPointsRight,
    const std::vector<cv::Point2f>& noisyPointsLeft,
    const std::vector<cv::Point2f>& noisyPointsRight,
    const cv::Mat& R, const cv::Mat& T, const cv::Mat& K,
    const cv::Mat& F, bool vizEpipolarLines);

///  Function to triangulate a point from two views
cv::Mat triangulatePoint(const cv::Point2f& pt1, const cv::Point2f& pt2,
    const cv::Mat& P1, const cv::Mat& P2);

// Gets the positive depth points
int countPositiveDepthPoints(const cv::Mat& points3D, const cv::Mat& P);

// Function to check the determinant of the rotation matrix
void checkRotationMatrix(const cv::Mat& R);

// Function to analyze the Gaussian noise impact
NoiseAnalysisResults analyzeNoiseImpact(
    const cv::Mat& img1,
    const cv::Mat& img2,
    const std::vector<cv::Point2f>& pointsLeft,
    const std::vector<cv::Point2f>& pointsRight,
    const cv::Mat& intrinsicMatrix,
    double maxNoiseLevel,
    double noiseStep,
    bool vizEpipolarLines);

// Calculates the Projection error
double calculateReprojectionError(const cv::Mat& F,
    const std::vector<cv::Point2f>& points1,
    const std::vector<cv::Point2f>& points2);