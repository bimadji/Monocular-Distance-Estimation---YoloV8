cmake_minimum_required(VERSION 3.5)
project(inferenceTest VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.2")
find_package(CUDA 12 REQUIRED)

set(CMAKE_CUDA_STANDARD 12)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.2)

# OpenCV
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

set(PROJECT_SOURCES
    main.cpp
    inference.h
    inference.cpp
    # threshold.cpp
)

add_executable(inferenceTest ${PROJECT_SOURCES})
target_link_libraries(inferenceTest ${OpenCV_LIBS})
