#include "basicnn/nn/optimization.hpp"
#include "basicnn/nn/layer.hpp"
#include "basicnn/utils.hpp"
#include "basicnn/linalg/linalg.hpp"
#include "basicnn/linalg/matrix.hpp"
#include <vector>
#include <memory>
#include <gtest/gtest.h>

using namespace basicnn::nn;
using namespace basicnn::nn::optimization;
using namespace basicnn::linalg;
using namespace basicnn::utils;

TEST(OptimizationTest, SGDUpdate){

    float lr = 0.001f;
    Matrix<float> W1 = ones<float>(4,4);
    Matrix<float> W2 = W1*2.0f;
    std::vector<float> b1 = ones<float>(4);
    std::vector<float> b2 = b1*2.0f;
    std::vector<LayerWeights> netW = {{W1,b1},{W2,b2}};
    Matrix<float> W1_copy = W1;
    Matrix<float> W2_copy = W2;
    std::vector<float> b1_copy = b1;
    std::vector<float> b2_copy = b2;

    Matrix<float> gW1 = ones<float>(4,4);
    Matrix<float> gW2 = gW1*2.0f;
    std::vector<float> gb1 = ones<float>(4);
    std::vector<float> gb2 = gb1*2.0f;
    std::vector<LayerGradients> netg = {{gW1,gb1},{gW2,gb2}};

    std::unique_ptr<Optimizer> optimizer = std::make_unique<SGD>(netW, netg, lr);

    EXPECT_EQ(netW[0].W, W1);
    EXPECT_EQ(netW[0].b, b1);
    EXPECT_EQ(netW[1].W, W2);
    EXPECT_EQ(netW[1].b, b2);
    
    optimizer->update();

    Matrix<float> W2_new = W2_copy - lr * gW2;
    std::vector<float> b2_new = b2_copy - lr * gb2;

    EXPECT_EQ(netW[1].W, W2_new);
    EXPECT_EQ(netW[1].b, b2_new);
}

TEST(OptimizationTest, ADAMUpdate){

    float lr = 0.001f;
    Matrix<float> W1 = ones<float>(4,4);
    Matrix<float> W2 = W1*2.0f;
    std::vector<float> b1 = ones<float>(4);
    std::vector<float> b2 = b1*2.0f;
    std::vector<LayerWeights> netW = {{W1,b1},{W2,b2}};
    Matrix<float> W1_copy = W1;
    Matrix<float> W2_copy = W2;
    std::vector<float> b1_copy = b1;
    std::vector<float> b2_copy = b2;

    Matrix<float> gW1 = ones<float>(4,4);
    Matrix<float> gW2 = gW1*2.0f;
    std::vector<float> gb1 = ones<float>(4);
    std::vector<float> gb2 = gb1*2.0f;
    std::vector<LayerGradients> netg = {{gW1,gb1},{gW2,gb2}};

    std::unique_ptr<Optimizer> optimizer = std::make_unique<ADAM>(netW, netg, lr);

    EXPECT_EQ(netW[0].W, W1);
    EXPECT_EQ(netW[0].b, b1);
    EXPECT_EQ(netW[1].W, W2);
    EXPECT_EQ(netW[1].b, b2);
    
    optimizer->update();

    Matrix<float> W2_new = W2_copy - lr * gW2 / (gW2 + 1e-8f * ones<float>(W2.rows(), W2.cols()));
    std::vector<float> b2_new = b2_copy - lr * gb2 / (gb2 + 1e-8f * ones<float>(b2.size()));

    EXPECT_EQ(netW[1].W, W2_new);
    EXPECT_EQ(netW[1].b, b2_new);
}

TEST(OptimizationTest, EarlyStop){

    Matrix<float> W = ones<float>(3,3);
    std::vector<float> b = ones<float>(3);

    std::vector<LayerWeights> weights = {{W,b}, {W,b}, {W,b}};

    EarlyStop earlystop(2);

    earlystop.epoch(weights, 0.2f);
    EXPECT_FALSE(earlystop.stop());

    auto [W1, b1] = weights[0];
    W1(0,0) = 4.0f;
    earlystop.epoch(weights, 0.19f);
    EXPECT_FALSE(earlystop.stop());

    earlystop.epoch(weights, 0.21f);
    EXPECT_FALSE(earlystop.stop());

    earlystop.epoch(weights, 0.24f);
    EXPECT_FALSE(earlystop.stop());

    earlystop.epoch(weights, 0.26f);
    EXPECT_TRUE(earlystop.stop());

    auto [W2, b2] = earlystop.bestWeights();
    EXPECT_EQ(W2[0](0,0), 4.0f);
}