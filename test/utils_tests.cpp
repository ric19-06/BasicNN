#include "basicnn/linalg/matrix.hpp"
#include "basicnn/utils.hpp"
#include "basicnn/linalg/linalg.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace basicnn::utils;
using namespace basicnn::linalg;

TEST(UtilsTest, VectorOnes){

    std::vector<double> v = ones<double>(6);

    for(size_t i=0; i<v.size(); ++i){

        EXPECT_EQ(v[i], 1.0);
    }
};

TEST(UtilsTest, VectorZeros){

    std::vector<double> v = zeros<double>(6);

    for(size_t i=0; i<v.size(); ++i){

        EXPECT_EQ(v[i], 0.0);
    }
}

TEST(UtilsTest, MatrixOnes){

    Matrix<double> M = ones<double>(2, 3);

    for(size_t i=0; i<M.rows(); ++i){
        for(size_t j=0; j<M.cols(); ++j){

            EXPECT_EQ(M(i,j), 1.0);
        }
    }
}

TEST(UtilsTest, MatrixZeros){

    Matrix<double> M = zeros<double>(2, 3);

    for(size_t i=0; i<M.rows(); ++i){
        for(size_t j=0; j<M.cols(); ++j){

            EXPECT_EQ(M(i,j), 0.0);
        }
    }
}

TEST(UtilsTest, MatrixIdentity){

    Matrix<double> M = identity<double>(2);
    Matrix<double> I({1,0,0,1}, 2, 2);

    EXPECT_EQ(M, I);
}