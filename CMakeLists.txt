cmake_minimum_required(VERSION 3.4...3.28 FATAL_ERROR)

# Project Name and Version
set(PROJECT_NAME LaneDepartureSystem)
project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES C CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the architecture
set(CMAKE_GENERATOR_PLATFORM x64)

# Path to OpenCV
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/external/opencv/build)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add the executable target first
add_executable(${PROJECT_NAME} 
    ${CMAKE_SOURCE_DIR}/main.cpp 
    ${CMAKE_SOURCE_DIR}/src/LaneDetector.cpp
    ${CMAKE_SOURCE_DIR}/inc/LaneDetector.hpp
)

# Then specify include directories and link libraries
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Debugging Messages
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
