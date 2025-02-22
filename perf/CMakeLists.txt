cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER clang CACHE STRING "Clang compiler") 
set(CMAKE_CXX_COMPILER clang++ CACHE STRING "Clang++ compiler")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile commands")

set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD 23 CACHE STRING "C++ standard to conform to")
set(CMAKE_C_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)

# Set the path to the TorchConfig.cmake file
set(Torch_DIR /Volumes/code/env/pymetal/lib/python3.9/site-packages/torch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

project("perf")

set(CMAKE_VERBOSE_MAKEFILE ON)

file(GLOB_RECURSE PERF_C *.c)
file(GLOB_RECURSE PERF_CPP *.cpp)
add_executable(perf ${PERF_C} ${PERF_CPP})

# Link against the PyTorch libraries
target_link_libraries(perf "${TORCH_LIBRARIES}")

# Optionally suppress developer warnings
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "Suppress developer warnings")