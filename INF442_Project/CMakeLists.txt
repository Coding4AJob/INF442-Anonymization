cmake_minimum_required(VERSION 3.16)
project(INF442_Project)

set(CMAKE_CXX_STANDARD 11)
include_directories("eigen3")
add_executable(INF442_Project main.cpp svm.cpp svm.h kernel.cpp kernel.h)