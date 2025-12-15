#include "basicnn/nn/layer.hpp"
#include "basicnn/utils.hpp"
#include "basicnn/linalg/linalg.hpp"
#include "basicnn/linalg/matrix.hpp"
#include <vector>
#include <gtest/gtest.h>
#include <cmath>

using namespace basicnn::nn;
using namespace basicnn::linalg;
using namespace basicnn::utils;

TEST(LayerTest, Init){

    Layer layer(NeuronType::HIDDEN, 5, ActivationType::SIGMOID);

    EXPECT_EQ(layer.getType(), NeuronType::HIDDEN);
    EXPECT_EQ(layer.getSize(), 5);
    EXPECT_EQ(layer.getActivationType(), ActivationType::SIGMOID);
}

TEST(LayerTest, GetWeights){

    Layer layer(NeuronType::HIDDEN, 4, ActivationType::SIGMOID);

    std::vector<float> data = {1, 5,
                                2, 6,
                                3, 7,
                                4, 8};
    Matrix<float> Z(data, 4, 2);
    std::mt19937 rng(42);
    layer.weightInit(rng, 4, 4);
    Matrix<float> Z_out1 = layer.forward(Z);

    auto [W, b] = layer.getWeights();
    W(0,0) = -W(0,0);
    W(1,1) = -W(1,1);

    Matrix<float> Z_out2 = layer.forward(Z);

    EXPECT_NE(Z_out1(0,0), Z_out2(0,0));
    EXPECT_NE(Z_out1(1,1), Z_out2(1,1));
    EXPECT_EQ(Z_out1(2,0), Z_out2(2,0));
    EXPECT_EQ(Z_out1(3,1), Z_out2(3,1));
}

TEST(LayerTest, SetParameters){

    Layer layer(NeuronType::HIDDEN, 5, ActivationType::SIGMOID);

    EXPECT_EQ(layer.getType(), NeuronType::HIDDEN);
    EXPECT_EQ(layer.getSize(), 5);
    EXPECT_EQ(layer.getActivationType(), ActivationType::SIGMOID);

    layer.setParameters(NeuronType::OUTPUT, 10, ActivationType::SOFTMAX);

    EXPECT_EQ(layer.getType(), NeuronType::OUTPUT);
    EXPECT_EQ(layer.getSize(), 10);
    EXPECT_EQ(layer.getActivationType(), ActivationType::SOFTMAX);
}

TEST(LayerTest, SetWeights){

    Layer layer(NeuronType::HIDDEN, 2, ActivationType::SIGMOID);

    std::mt19937 rng(42);
    layer.weightInit(rng, 2, 2);
    auto [W, b] = layer.getWeights();
    Matrix<float> W_copy = W;

    Matrix<float> W_new({1,2,3,4}, 2, 2);
    layer.setWeights(W_new, b);

    EXPECT_NE(W, W_copy);
}

TEST(LayerTest, Backward){

    Layer layer(NeuronType::HIDDEN, 4, ActivationType::SIGMOID);

    std::vector<float> data = {1};
    Matrix<float> Z_kminus1(data, 1, 1);

    // [ 4 x 1 ] all sigma(1)
    Matrix<float> Z = layer.forward(Z_kminus1);

    // [ 4 x 1 ] all sigma(1) - 1
    Matrix<float> D_kplus1 = Z - ones<float>(4,1);

    // [ 4 x 4 ] identity matrix
    Matrix<float> W_kplus1 = identity<float>(4);

    layer.backward(W_kplus1, D_kplus1);

    // [ 4 x 1 ] all -e/(e+1)^3
    Matrix<float> D = layer.getDelta();

    float e = std::exp(1);
    float value = - e / std::pow((e + 1), 3);
    Matrix<float> M = value * ones<float>(4,1);

    EXPECT_EQ(D, M);
}

TEST(LayerTest, ComputeGradients){

    Layer layer(NeuronType::HIDDEN, 4, ActivationType::SIGMOID);

    std::vector<float> data = {1};
    Matrix<float> Z_kminus1(data, 1, 1);

    // [ 4 x 1 ] all sigma(1)
    Matrix<float> Z = layer.forward(Z_kminus1);

    // [ 4 x 1 ] all sigma(1) - 1
    Matrix<float> D_kplus1 = Z - ones<float>(4,1);

    // [ 4 x 4 ] identity matrix
    Matrix<float> W_kplus1 = identity<float>(4);

    layer.backward(W_kplus1, D_kplus1);

    // [ 4 x 1 ] all -e/(e+1)^3
    Matrix<float> D = layer.getDelta();

    layer.computeGradients(Z_kminus1);

    auto gradients = layer.getGradients();

    float e = std::exp(1);
    float value = - e / std::pow((e + 1), 3);

    // [ 4 x 1 ] all -e/(e+1)^3
    Matrix<float> gW = value * ones<float>(4,1);

    // [ 4 x 1 ] all -e/(e+1)^3
    std::vector<float> gb = value * ones<float>(4);

    EXPECT_EQ(gradients.gW, gW);
    EXPECT_EQ(gradients.gb, gb);
}