cmake_minimum_required(VERSION 3.24)
project(SSD_SAD_improvement)

set(CMAKE_CXX_STANDARD 11)

# Enable AVX2 instructions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")

add_executable(SSD_SAD_improvement main.cpp)
