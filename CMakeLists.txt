cmake_minimum_required(VERSION 3.10.0)
project(BasicNN VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Speed up matrix multiplication
# Add O3 and fopenmp flags to compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -g")
# Find the OpenMP for cpu parallelism
find_package(OpenMP REQUIRED)

# Fetch cpptraces
include(FetchContent)
FetchContent_Declare(
  cpptrace
  GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
  GIT_TAG        v1.0.4
)
FetchContent_MakeAvailable(cpptrace)

# Fetch pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0   # stable, modern
)

FetchContent_MakeAvailable(pybind11)

# Set python path
set(Python3_ROOT_DIR "${PROJECT_SOURCE_DIR}/.venv")
set(Python3_FIND_VIRTUALENV FIRST)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Enable testing
enable_testing()

# Add the necessary folders
add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(test)