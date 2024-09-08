cmake_minimum_required(VERSION 3.5.0)
project(cstuff VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)  # Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(cstuff forest.cpp)

