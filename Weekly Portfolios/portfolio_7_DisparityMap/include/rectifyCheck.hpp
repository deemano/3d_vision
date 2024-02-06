// rectifyCheck.hpp
#ifndef RECTIFY_CHECK_HPP
#define RECTIFY_CHECK_HPP

#include <opencv2/core/mat.hpp> // Include the necessary OpenCV headers

// To check if images are rectified
void checkRectification(const cv::Mat& leftImage, const cv::Mat& rightImage);

#endif // RECTIFY_CHECK_HPP
