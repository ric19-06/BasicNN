#include "basicnn/linalg/matrix.hpp"
#include "basicnn/linalg/linalg.hpp"
#include <vector>
#include <gtest/gtest.h>

using namespace basicnn::linalg;

TEST(MatrixTest, Init){

    std::vector<std::vector<double>> data = {{1, 1}, {1, 1}, {1, 1}};
    Matrix<double> M(data);

    EXPECT_EQ(M.rows(), 3);
    EXPECT_EQ(M.cols(), 2);
}

TEST(MatrixTest, EmptyInit){

    Matrix<double> M;

    EXPECT_EQ(M.rows(), 0);
    EXPECT_EQ(M.cols(), 0);
}

TEST(MatrixTest, Clear){

    std::vector<double> data = {1, 2, 3,
                                4, 5, 6};
    Matrix<double> M(data, 2, 3);

    EXPECT_FALSE(M.empty());
    M.clear();
    EXPECT_TRUE(M.empty());
}

TEST(MatrixTest, PushBack){

    std::vector<double> data1 = {1, 2, 3};
    std::vector<double> data2 = {4, 5, 6};
    std::vector<double> dataFull = {1, 2, 3,
                                    4, 5, 6};
    Matrix<double> M;
    Matrix<double> Q;
    M.reserve(2, 3);
    Q.reserve(2, 3);

    M.push_back(data1);
    M.push_back(data2);
    Q.push_back(dataFull, 2, 3);
    M(1,1) = 2.0;
    Q(1,1) = 2.0;

    EXPECT_EQ(M, Q);
}

TEST(MatrixTest, RoundBrackets){

    std::vector<double> data = {1, 2, 3,
                                4, 5, 6};
    Matrix<double> M(data, 2, 3);

    EXPECT_EQ(M(4), 5.0);
    M(1,1) = 2.0;
    EXPECT_EQ(M(4), 2.0);
}

TEST(MatrixTest, VectorOutput){

    std::vector<double> data = {1, 2, 3};
    Matrix<double> A(data, 1, 3);
    Matrix<double> B(data, 3, 1);

    EXPECT_EQ(A.vector(), B.vector());
}

TEST(MatrixTest, Transpose){

    std::vector<double> data1 = {1, 2, 3,
                                4, 5, 6};
    std::vector<double> data2 = {1, 4,
                                 2, 5,
                                 3, 6};
    Matrix<double> M(data1, 2, 3);
    Matrix<double> Q(data2, 3, 2);

    EXPECT_EQ(M.transpose(), Q);
}