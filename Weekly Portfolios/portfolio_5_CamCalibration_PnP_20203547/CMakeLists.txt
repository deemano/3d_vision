cmake_minimum_required(VERSION 3.21) # Version 3.21 is recommended for VS2022
project(augmentedReality)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Adjust the paths for Eigen and OpenCV to reflect the 'libs' folder
#set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen")
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen/eigen-3.4.0")
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/opencv/build")

# Include directories for Eigen and OpenCV
include_directories(${EIGEN_DIR})
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

# Set up the executable, assuming your source files are in the 'src' folder
# file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp") #4 automation
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Print the Working Directory in CMake
message(STATUS "Setting the working directory to: ${CMAKE_SOURCE_DIR}")
add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/other.cpp
    src/w5_ex2.cpp
    src/camera_calibration.cpp
    # ... and so on for other .cpp files
 "src/w5_ex2.cpp")

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Set runtime library directories for Debug and Release
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)

# Set additional properties for Visual Studio
if(MSVC)
  set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
  
  # Set the working directory to the project root directory
  set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

# Post-build command to copy DLL files to the output directory
if(WIN32)
    # Update the path to the actual location of OpenCV DLLs
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/libs/opencv/build/x64/vc16/bin"
                $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()
