#include "common.h"
#include "rectifyCheck.hpp"

// check rectification with feature matching
void checkRectification(const cv::Mat& leftImage, const cv::Mat& rightImage) {
    // ORB detector
    cv::Ptr<cv::ORB> detector = cv::ORB::create();

    // Keypoints and descriptors
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;

    // Detect and compute features
    detector->detectAndCompute(leftImage, cv::noArray(), keypointsLeft, descriptorsLeft);
    detector->detectAndCompute(rightImage, cv::noArray(), keypointsRight, descriptorsRight);

    // Matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsLeft, descriptorsRight, matches);

    // Use RANSAC to filter matches
    std::vector<cv::DMatch> goodMatches;
    std::vector<cv::Point2f> leftPoints;
    std::vector<cv::Point2f> rightPoints;

    for (const auto& match : matches) {
        leftPoints.push_back(keypointsLeft[match.queryIdx].pt);
        rightPoints.push_back(keypointsRight[match.trainIdx].pt);
    }

    if (leftPoints.size() >= 4 && rightPoints.size() >= 4) {
        std::vector<uchar> inliersMask(leftPoints.size());
        cv::findFundamentalMat(leftPoints, rightPoints, cv::FM_RANSAC, 3, 0.99, inliersMask);

        for (size_t i = 0; i < inliersMask.size(); i++) {
            if (inliersMask[i]) {
                goodMatches.push_back(matches[i]);
            }
        }
    }

    // Draw matches
    cv::Mat imgMatches;
    cv::drawMatches(leftImage, keypointsLeft, rightImage, keypointsRight, goodMatches, imgMatches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Display the matches
    double resizeFactor = 0.5;  // Resize imgMatches 
    cv::Mat resizedImgMatches;
    cv::resize(imgMatches, resizedImgMatches, cv::Size(), resizeFactor, resizeFactor);
    cv::imshow("Good Matches after RANSAC", resizedImgMatches);
    cv::waitKey(0); // Wait for a key press to close the displayed window

    // Check y-coordinates for good matches
    int alignedMatchesCount = 0;
    int notAllignedMatches = 0;
    for (const auto& match : goodMatches) {
        float dy = keypointsLeft[match.queryIdx].pt.y - keypointsRight[match.trainIdx].pt.y;
        if (std::abs(dy) <= 1.0) {  // Allow a small tolerance
            alignedMatchesCount++;
        }
    }
    // Print the number of aligned matches
    std::cout << "Number of aligned matches on epipolar lines: " << alignedMatchesCount << " out of " << goodMatches.size() << std::endl;

    float matchingRate = (static_cast<float>(alignedMatchesCount) / goodMatches.size()) * 100.0f;

    if (matchingRate > 60.0f) {
        std::cout << "Images pair seem rectified, with a " << matchingRate << "% matching rate" << std::endl;
    }
    else if (matchingRate > 45.0f) {
        std::cout << "Not enough matching points on their epipolar lines - Images pair look un-rectified." << std::endl;
    }
    else {
        std::cout << "Not enough good matching points, please refine parameters" << std::endl;
    }
}
 