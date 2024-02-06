cmake_minimum_required(VERSION 3.21)  # Version 3.21 is recommended for VS2022
project(Portfolio_7)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the paths for the included libraries
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/eigen/eigen-3.4.0")
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/opencv/build") 
set(GLEW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/glew")
set(GLFW_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw/include")
set(GLM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs")
set(ASSIMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/assimp")
set(ASSIMP_INCLUDE_DIRS "${ASSIMP_DIR}/include")
set(ASSIMP_LIBRARY_DIR "${ASSIMP_DIR}/lib")
set(GLFW_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw/lib-vc2022")
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/temps)

# Post-build command to copy OpenCV DLL files to the output directory
set(OPENCV_DLL_DEBUG_PATH "${OpenCV_DIR}/x64/vc16/bin/opencv_world480d.dll")
set(OPENCV_DLL_RELEASE_PATH "${OpenCV_DIR}/x64/vc16/bin/opencv_world480.dll")

# Specify the Windows SDK version 
set(CMAKE_SYSTEM_VERSION "10.0.22621.0")

# Set CMAKE_PREFIX_PATH to help CMake find the Qt6 installation
#set(CMAKE_PREFIX_PATH "C:/Qt/6.6.1/msvc2019_64")

# Find the required libraries
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
find_library(GLEW_LIBRARY NAMES glew32 PATHS "${GLEW_PATH}/lib/Release/x64" NO_DEFAULT_PATH)
find_library(GLFW_LIBRARY NAMES glfw3 PATHS "${GLFW_LIB_PATH}" NO_DEFAULT_PATH)
find_library(ASSIMP_LIBRARY NAMES assimp-vc143-mt PATHS "${ASSIMP_LIBRARY_DIR}/x64" NO_DEFAULT_PATH)

include_directories(
    ${EIGEN_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${GLFW_PATH}
    ${GLEW_PATH}/include
    ${ASSIMP_INCLUDE_DIRS}
    ${GLM_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    #${CMAKE_CURRENT_SOURCE_DIR}/gui
)

# Find and link the Threads package
find_package(Threads REQUIRED)

# Set up the executable with all your .cpp files
add_executable(${PROJECT_NAME}
    src/main.cpp

    # ... other header files
 "src/rectifyCheck.cpp")

# Link all libraries together
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${GLEW_LIBRARY}
    ${GLFW_LIBRARY}
    ${ASSIMP_LIBRARY}
    opengl32
    Threads::Threads
)

# Set runtime library directories for Debug and Release
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)

# Set additional properties for Visual Studio
if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
    set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

# Handle errors if libraries are not found
# Post-build command to copy DLL files to the output directory
if(WIN32)
    # Correct path for GLEW DLL
    set(GLEW_DLL "${CMAKE_CURRENT_SOURCE_DIR}/libs/glew/bin/Release/x64/glew32.dll")
    if(EXISTS "${GLEW_DLL}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLEW_DLL}"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    else()
        message(WARNING "GLEW DLL not found: ${GLEW_DLL}")
    endif()

    # Correct path for GLFW DLL
    set(GLFW_DLL "${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw/lib-vc2022/glfw3.dll")
    if(EXISTS "${GLFW_DLL}")
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GLFW_DLL}" 
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    else()
        message(WARNING "GLFW DLL not found: ${GLFW_DLL}")
    endif()
endif()

# Post-build command to copy Assimp DLL files to the output directory
set(ASSIMP_DLL "${ASSIMP_DIR}/bin/x64/assimp-vc143-mt.dll") 
if(EXISTS "${ASSIMP_DLL}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ASSIMP_DLL}"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
else()
    message(WARNING "Assimp DLL not found: ${ASSIMP_DLL}")
endif()

# Copy the appropriate DLLs for Debug configuration
if(EXISTS "${OPENCV_DLL_DEBUG_PATH}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPENCV_DLL_DEBUG_PATH}"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
else()
    message(WARNING "OpenCV debug DLL not found: ${OPENCV_DLL_DEBUG_PATH}")
endif()

# Copy the appropriate DLLs for Release configuration
if(EXISTS "${OPENCV_DLL_RELEASE_PATH}")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPENCV_DLL_RELEASE_PATH}"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
else()
    message(WARNING "OpenCV release DLL not found: ${OPENCV_DLL_RELEASE_PATH}")
endif()