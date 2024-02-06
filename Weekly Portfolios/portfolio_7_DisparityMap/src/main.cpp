#include "common.h"
#include "rectifyCheck.hpp"

// Function prototypes - Forward declarations
void processStereoPair(const cv::Range& range,
    const std::vector<std::string>& leftImages,
    const std::vector<std::string>& rightImages,
    const std::filesystem::path& results_directory);

int main() {
    // Suppress OpenCV informational logs warnings 
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    // Set current working directories as variable
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path base_directory = cwd.parent_path().parent_path().parent_path().parent_path();
    std::filesystem::path left_directory = base_directory / "resources" / "left";
    std::filesystem::path right_directory = base_directory / "resources" / "right";
    std::filesystem::path results_directory = base_directory / "results";

    // Create the results directory if it does not exist
    if (!std::filesystem::exists(results_directory)) {
        std::filesystem::create_directories(results_directory);
    }
    // define image name formats
    std::string leftImageFilename = "000000.png";
    std::string rightImageFilename = "000000.png";

    // Combine the directory path with the filenames
    std::filesystem::path leftImagePath = left_directory / leftImageFilename;
    std::filesystem::path rightImagePath = right_directory / rightImageFilename;

    // Output Feedback
    //std::cout << " Loading images from: " << leftImagePath << " and " << rightImagePath << std::endl;

    
    // Load 1 set of images to check for epipolar line matching
    cv::Mat leftImage = cv::imread(leftImagePath.string(), cv::IMREAD_GRAYSCALE);
    cv::Mat rightImage = cv::imread(rightImagePath.string(), cv::IMREAD_GRAYSCALE);

    if (leftImage.empty() || rightImage.empty()) {
        std::cerr << " Error loading images." << std::endl;
        return -1;
    }

    // Check if images are rectified - used 1 time on an image pair
    // checkRectification(leftImage, rightImage);

    // Load image paths into vectors
    std::vector<std::string> leftImages;   // Fill with left image paths
    std::vector<std::string> rightImages;  // Fill with right image paths

    // Load image paths into vectors
    char filename[512]; // Buffer size for parallel threding
    for (int i = 0; i < 100; ++i) { // Assuming there are 100 images
        snprintf(filename, sizeof(filename), "%06d.png", i);
        leftImages.push_back((left_directory / filename).string());
        rightImages.push_back((right_directory / filename).string());
    }

    std::cout << " Loaded " << leftImages.size() << " left images and " << rightImages.size() << " right images." << std::endl;

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << " Now starting parallel processing for " << leftImages.size() << " image pairs." << std::endl;

    // Parallel processing
    cv::parallel_for_(cv::Range(0, leftImages.size()), [&](const cv::Range& range) {
        processStereoPair(range, leftImages, rightImages, results_directory);
        });

    std::cout << "\n Parallel processing completed." << std::endl;

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate and print the elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << " Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}

// Function to process a range of stereo pairs
void processStereoPair(const cv::Range& range,
    const std::vector<std::string>& leftImages,
    const std::vector<std::string>& rightImages,
    const std::filesystem::path& results_directory) {

    // Parameters
    int minDisparity = 0;
    int numDisparities = 64;  // Must be divisible by 16
    int blockSize = 15;       // Typical values are 3, 5, 7, etc.

    // Initialize StereoBM
    auto localStereoBM = cv::StereoBM::create(numDisparities, blockSize);
    localStereoBM->setPreFilterType(cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE);
    localStereoBM->setPreFilterSize(5);
    localStereoBM->setPreFilterCap(61);
    localStereoBM->setTextureThreshold(10);
    localStereoBM->setUniquenessRatio(15); // this is nice 1
    localStereoBM->setSpeckleWindowSize(100);
    localStereoBM->setSpeckleRange(32);
    localStereoBM->setDisp12MaxDiff(1);

    for (int i = range.start; i < range.end; i++) {
        cv::Mat leftImage = cv::imread(leftImages[i], cv::IMREAD_GRAYSCALE);
        cv::Mat rightImage = cv::imread(rightImages[i], cv::IMREAD_GRAYSCALE);
        // Catch image load errors
        if (leftImage.empty() || rightImage.empty()) {
            std::cerr << "Error loading image pair: " << leftImages[i] << ", " << rightImages[i] << std::endl;
            continue;
        }
        // Extract the number part of the filename for feedback
        std::string imageNumber = leftImages[i].substr(leftImages[i].length() - 10, 6);
        std::cout << " " << " Processing image pair number: " << imageNumber << "\n";

        // Process Disparity
        cv::Mat disparity;
        localStereoBM->compute(leftImage, rightImage, disparity);
        if (disparity.empty()) { // Error catcher
            std::cerr << "Error computing disparity for image pair: " << leftImages[i] << ", " << rightImages[i] << std::endl;
            continue;
        }

        // Normalize the disparity map to 8-bit for visualization
        cv::Mat dispNorm;
        cv::normalize(disparity, dispNorm, 0, 255, cv::NORM_MINMAX, CV_8U);

        // Invert the normalized disparity map
        cv::Mat invertedDisp;
        cv::subtract(cv::Scalar::all(255), dispNorm, invertedDisp);
        
        // Save the normalized disparity map
        std::string savePath = (results_directory / ("CV_32F_disp/disparity_" + imageNumber + ".png")).string();
        //std::string savePath = (results_directory / ("CV_32F_disp/disparity_" + imageNumber + ".exr")).string();
        cv::imwrite(savePath, dispNorm);
    }
}

