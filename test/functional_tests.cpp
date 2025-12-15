#include <gtest/gtest.h>
#include "basicnn/nn/functional.hpp"
#include "basicnn/linalg/linalg.hpp"
#include "basicnn/linalg/matrix.hpp"

using namespace basicnn::nn::functional;
using namespace basicnn::linalg;

TEST(FunctionalTest, Identity){

    EXPECT_EQ(identity(0.5f), 0.5f);
}

TEST(FunctionalTest, Sigmoid){

    EXPECT_GT(sigmoid(0.5f), 0.6f);
}

TEST(FunctionalTest, Relu){

    EXPECT_EQ(relu(-0.5f), 0.0f);
    EXPECT_EQ(relu(0.5f), 0.5f);
}

TEST(FunctionalTest, Softmax){

    Matrix<float> M({1,1,
                      4,3}, 2, 2);
    
    Matrix<float> S = softmax(M);

    EXPECT_LT(S(0,0), 0.05);
    EXPECT_GT(S(1,0), 0.95);
    EXPECT_LT(S(0,1),0.12);
    EXPECT_GT(S(1,1), 0.88);
}

TEST(FunctionalTest, IdentityGrad){

    EXPECT_EQ(identity_grad(0.5f), 1.0f);
}

TEST(FunctionalTest, SigmoidGrad){

    EXPECT_GT(sigmoid_grad(0.5f),0.2f);
}

TEST(FunctionalTest, ReluGrad){

    EXPECT_EQ(relu_grad(-0.5f), 0.0f);
    EXPECT_EQ(relu_grad(0.5f), 1.0f);
}

TEST(FunctionalTest, Golorot) {

    std::mt19937 rng(123);
    size_t fan_in = 8, fan_out = 16;
    
    float limit = std::sqrt(6.0 / (fan_in + fan_out));

    for(size_t i = 0; i<1000; ++i) {
        float x = glorot(rng, fan_in, fan_out);

        EXPECT_GE(x, -limit);
        EXPECT_LE(x,  limit);
    }
}

TEST(FunctionalTest, He){

    std::mt19937 rng(123);
    size_t fan_in = 8;

    float sigma = 2.0 / fan_in;

    float avg = 0;
    for(size_t i=0; i<50000; ++i){

        float x = he(rng, fan_in, 0);

        EXPECT_FALSE(std::isnan(x));
        avg += x;
    }

    avg = avg / 50000;
    EXPECT_NEAR(avg, 0.0, 0.05);
}

TEST(FunctionalTest, Loss){

    Matrix<float> Yhat({0.8,0.2,0.9,
                        0.2,0.8,0.1}, 2, 3);
    Matrix<float> Ytrue1({1,0,0,
                          0,1,1}, 2, 3);
    Matrix<float> Ytrue2({1,0,1,
                          0,1,0}, 2, 3);

    EXPECT_GT(crossEntropy(Yhat,Ytrue1),crossEntropy(Yhat,Ytrue2));
}