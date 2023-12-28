#include <iostream>
#include <filesystem>
#include <string>
#include "other.hpp"

// Wrapping the script as namespace
namespace other {
    // Get the working folder before checking the image files below
    void printWorkingDirectory() {
        std::filesystem::path currentPath = std::filesystem::current_path();
        std::filesystem::path relativePath = currentPath / "../../../../";
        std::filesystem::path resolvedPath = std::filesystem::weakly_canonical(relativePath);
        std::cout << "Resolved Relative Path: " << resolvedPath << std::endl;
    }
    // Checking the existance of the image files using the path structure from above
    std::string getImagePath(int imageIndex) {
        std::string imagePath = "../../../../resources/images/camera_calib" + std::to_string(imageIndex) + ".jpg";
        // Feedback output
        if (std::filesystem::exists(imagePath)) {
            std::cout << "Image exists: " << imagePath << std::endl;
            return imagePath;
        }
        else {
            std::cerr << "Image file does not exist: " << imagePath << std::endl;
            return "";
        }
    }
}
