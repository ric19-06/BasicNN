# Collect all .cpp files in src as the library implementation
add_library(basicnn
    data_loader.cpp
    functional.cpp
    layer.cpp
    neural_network.cpp
    omp_init.cpp
    optimization.cpp
    training.cpp
    utils.cpp
)

# Optionally specify the public include directory
target_include_directories(basicnn PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Link core to OpenMP and cpptrace
target_link_libraries(basicnn 
    PRIVATE OpenMP::OpenMP_CXX
    PUBLIC cpptrace::cpptrace
    PUBLIC pybind11::embed
)