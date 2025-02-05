cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang CACHE STRING "Clang compiler") 
set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++ CACHE STRING "Clang++ compiler")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Generate compile commands")

set(CMAKE_CXX_STANDARD 23 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD 23 CACHE STRING "C++ standard to conform to")
set(CMAKE_C_STANDARD_REQUIRED True)

project("simd")

# set(CMAKE_VERBOSE_MAKEFILE ON)
file(GLOB_RECURSE SIMD_C SIMD/*.c)
file(GLOB_RECURSE SIMD_CPP SIMD/*.cpp)
add_executable(simd main.cpp ${SIMD_C} ${SIMD_CPP})
# target_link_options(simd PRIVATE -fmodules-ts -x c++-system-header iostream)
# target_compile_options(simd PRIVATE -fmodules-ts -x c++-system-header iostream)
