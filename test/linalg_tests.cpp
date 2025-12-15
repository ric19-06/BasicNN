#include "basicnn/linalg/linalg.hpp"
#include "basicnn/linalg/matrix.hpp"
#include "basicnn/utils.hpp"
#include <gtest/gtest.h>

using namespace basicnn::linalg;

TEST(LinAlgTest, VectorScalar){

    std::vector<double> a = {1,2,3};
    std::vector<double> b = {2,4,6};

    EXPECT_EQ(2.0 * a, b);
    EXPECT_EQ(a * 2.0, b);
    EXPECT_EQ(b / 2.0, a);
}

TEST(LinAlgTest, VectorSum) {
    std::vector<double> a = {1,2,3};
    std::vector<double> b = {4,5,6};

    EXPECT_EQ(a + b, (std::vector<double>{5,7,9}));
    EXPECT_EQ(b - a, (std::vector<double>{3,3,3}));
}

TEST(LinAlgTest, VectorHadamard) {
    std::vector<double> a = {1,2,3};
    std::vector<double> b = {4,5,6};

    EXPECT_EQ(a * b, (std::vector<double>{4,10,18}));
}

TEST(LinAlgTest, VectorDivision){

    std::vector<double> a = {4,5,6};
    std::vector<double> b = {1,2,3};

    EXPECT_EQ(a / b, (std::vector<double>{4,2.5,2}));
}

TEST(LinAlgTest, DotProduct) {
    std::vector<double> a = {1,2,3};
    std::vector<double> b = {4,5,6};

    EXPECT_EQ(dot(a, b), 32.0);
}

TEST(LinAlgTest, VectorPow){

    std::vector<double> v = {2, 2, 2};
    std::vector<double> u = {std::pow(2,2), std::pow(2,2), std::pow(2,2)};

    EXPECT_EQ(pow(v,2.0), u);
}

TEST(LinAlgTest, VectorSqrt){

    std::vector<double> v = {2, 2, 2};
    std::vector<double> u = {std::sqrt(2), std::sqrt(2), std::sqrt(2)};

    EXPECT_EQ(sqrt(v), u);
}

TEST(LinAlgTest, MatrixScalar) {
    Matrix<double> A({1,2,3,4}, 2, 2);
    Matrix<double> R({2,4,6,8}, 2, 2);

    EXPECT_EQ(2.0 * A, R);
    EXPECT_EQ(A * 2.0, R);
    EXPECT_EQ(R / 2.0, A);
}

TEST(LinAlgTest, VectorMatrix) {
    Matrix<double> A({1,2,3,4}, 2, 2);
    std::vector<double> v = {1,1};

    EXPECT_EQ(A * v, (std::vector<double>{3,7}));
    EXPECT_EQ(v * A, (std::vector<double>{4,6}));
}

TEST(LinAlgTest, MatrixSum) {
    Matrix<double> A({1,2,3,4}, 2, 2);
    Matrix<double> B({2,0,1,2}, 2, 2);
    Matrix<double> R1({3,2,4,6}, 2, 2);
    Matrix<double> R2({-1,2,2,2}, 2, 2);

    EXPECT_EQ(A + B, R1);
    EXPECT_EQ(A - B, R2);
}

TEST(LinAlgTest, MatrixMul) {
    Matrix<double> A({1,2,3,4}, 2, 2);
    Matrix<double> B({2,0,1,2}, 2, 2);
    Matrix<double> R({4,4,10,8}, 2, 2);

    EXPECT_EQ(A * B, R);
}

TEST(LinAlgTest, MatrixDivision) {
    Matrix<double> A({2,0,1,2}, 2, 2);
    Matrix<double> B({1,2,2,2}, 2, 2);
    Matrix<double> R({2,0,0.5,1}, 2, 2);

    EXPECT_EQ(A / B, R);
}

TEST(LinAlgTest, MatrixHadamard) {
    Matrix<double> A({1,2,3,4}, 2, 2);
    Matrix<double> B({2,0,1,2}, 2, 2);
    Matrix<double> R({2,0,3,8}, 2, 2);
    
    EXPECT_EQ(hadamard(A, B), R);
}

TEST(LinAlgTest, MatrixPow){

    Matrix<double> M({2, 2, 2, 2}, 2, 2);
    Matrix<double> Q({std::pow(2,2), std::pow(2,2), std::pow(2,2), std::pow(2,2)}, 2, 2);

    EXPECT_EQ(pow(M, 2.0), Q);
}

TEST(LinAlgTest, MatrixSqrt){

    Matrix<double> M({2, 2, 2, 2}, 2, 2);
    Matrix<double> Q({std::sqrt(2), std::sqrt(2), std::sqrt(2), std::sqrt(2)}, 2, 2);

    EXPECT_EQ(sqrt(M), Q);
}

TEST(LinAlgTest, MatrixLog){

    Matrix<double> M({2, 3, 2, 2}, 2, 2);
    Matrix<double> Q({std::log(2), std::log(3), std::log(2), std::log(2)}, 2, 2);

    EXPECT_EQ(log(M), Q);
}