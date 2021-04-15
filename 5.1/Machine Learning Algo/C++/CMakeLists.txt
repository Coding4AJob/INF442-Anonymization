cmake_minimum_required(VERSION 3.16)
project(LogisticRegression)

set(CMAKE_CXX_STANDARD 11)
include_directories("eigen-3.3.9")
add_executable(LoadData LoadData.cpp)
add_executable(LogisticRegression LogisticRegression.cpp)