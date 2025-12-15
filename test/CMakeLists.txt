find_package(GTest REQUIRED)

function(add_gtest name)

    add_executable(${name} ${ARGN})

    target_link_libraries(${name}
        PRIVATE
            basicnn
            GTest::gtest
            GTest::gtest_main
            pthread
    )

    include(GoogleTest)
    gtest_discover_tests(${name}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        PROPERTIES TIMEOUT 10
    )
    
endfunction()

# Add the gtests
add_gtest(loader_tests loader_tests.cpp)
add_gtest(dataset_tests dataset_tests.cpp)
add_gtest(matrix_tests matrix_tests.cpp)
add_gtest(linalg_tests linalg_tests.cpp)
add_gtest(utils_tests utils_tests.cpp)
add_gtest(functional_tests functional_tests.cpp)
add_gtest(layer_tests layer_tests.cpp)
add_gtest(neural_network_tests neural_network_tests.cpp)
add_gtest(optimization_tests optimization_tests.cpp)