#include "basicnn/nn/neural_network.hpp"
#include "basicnn/nn/layer.hpp"
#include "basicnn/nn/functional.hpp"
#include "basicnn/utils.hpp"
#include "basicnn/linalg/linalg.hpp"
#include "basicnn/linalg/matrix.hpp"
#include <vector>
#include <gtest/gtest.h>

using namespace basicnn::nn;
using namespace basicnn::linalg;
using namespace basicnn::utils;
using namespace basicnn::nn::functional;

TEST(NetworkTest, Init){

    std::vector<size_t> sizes = {4,4,4};
    std::vector<ActivationType> activations = {ActivationType::SIGMOID, ActivationType::SOFTMAX};
    size_t seed = 42;

    NeuralNetwork net(sizes, activations, 0.0f, seed);

    EXPECT_EQ(net.getSize(), 3);
    EXPECT_EQ(net.getSizes(), sizes);
    EXPECT_EQ(net.getActivations(), activations);
    EXPECT_FALSE(net.initialized());
}

TEST(NetworkTest, GetWeights){

    std::vector<size_t> sizes = {4,4,4};
    std::vector<ActivationType> activations = {ActivationType::SIGMOID, ActivationType::SOFTMAX};
    size_t seed = 42;

    NeuralNetwork net(sizes, activations, 0.0f, seed);
    Matrix<float> W = ones<float>(4,1);
    std::vector<float> b = zeros<float>(4);

    auto weights = net.getWeights();
    for(size_t i=0; i<weights.size(); ++i){

        auto [W_, b_] = weights[i];
        EXPECT_EQ(W_, W);
        EXPECT_EQ(b_, b);
    }
}

TEST(NetworkTest, SetWeights){

    std::vector<size_t> sizes = {4,4,4};
    std::vector<ActivationType> activations = {ActivationType::SIGMOID, ActivationType::SOFTMAX};
    size_t seed = 42;

    NeuralNetwork net(sizes, activations, 0.0f, seed);
    Matrix<float> W = ones<float>(4,4) * 2.0f;
    std::vector<float> b = ones<float>(4);

    std::vector<Matrix<float>> W_new = {W, W};
    std::vector<std::vector<float>> b_new = {b, b};

    auto layerW = net.getWeights();

    std::vector<Matrix<float>> W_old;
    std::vector<std::vector<float>> b_old;

    for(size_t i=0; i<layerW.size(); ++i){

        auto [W_, b_] = layerW[i];
        W_old.push_back(W_);
        b_old.push_back(b_);
    }

    net.weightInit();

    std::cout << std::endl;

    for(size_t i=0; i<layerW.size(); ++i){

        auto [W_, b_] = layerW[i];
        std::cout << W_ << "\n" << W_old[i] << std::endl;
        std::cout << b_ << "\n" << b_old[i] << std::endl;
        EXPECT_NE(W_, W_old[i]);
        EXPECT_EQ(b_, b_old[i]);
    }

    std::cout << std::endl;

    net.setWeights(W_new, b_new);
    for(size_t i=0; i<layerW.size(); ++i){

        auto [W_, b_] = layerW[i];
        EXPECT_EQ(W_, W_new[i]);
        EXPECT_EQ(b_, b_new[i]);
    }
}

TEST(NetworkTest, Forward){

    std::vector<size_t> sizes = {1,4};
    std::vector<ActivationType> activations = {ActivationType::SIGMOID};
    size_t seed = 42;

    NeuralNetwork net(sizes, activations, 0.0f, seed);

    std::vector<float> z_in = {1};
    Matrix<float> Z_in(z_in, 1, 1);

    // [ 4 x 1 ] all sigma(1)
    std::vector<float> z_out = net.forward(z_in);
    Matrix<float> Z_out = net.forward(Z_in);

    EXPECT_EQ(z_out, ones<float>(4) * sigmoid(1.0));
    EXPECT_EQ(Z_out, ones<float>(4,1) * sigmoid(1.0));
}

TEST(NetworkTest, ComputeGradients){

    std::vector<size_t> sizes = {1,4};
    std::vector<ActivationType> activations = {ActivationType::SIGMOID};
    size_t seed = 42;

    NeuralNetwork net(sizes, activations, 0.0f, seed);

    std::vector<float> z_in = {1};
    Matrix<float> Z_in(z_in, 1, 1);

    // [ 4 x 1 ] all sigma(1)
    Matrix<float> Z_out = net.forward(Z_in);

    // [ 4 x 1 ] all sigma(1) - 1
    Matrix<float> G = Z_out - ones<float>(4,1);

    net.backward(G);
    net.computeGradients();

    float value = sigmoid(1.0) - 1;

    // [ 4 x 1 ] all sigma(1) - 1
    Matrix<float> gW = value * ones<float>(4,1);

    // [ 4 x 1 ] all sigma(1) - 1
    std::vector<float> gb = value * ones<float>(4);

    std::vector<LayerGradients> gradients = net.getGradients();

    EXPECT_EQ(gradients[0].gW, gW);
    EXPECT_EQ(gradients[0].gb, gb);

    net.backward(G * 2.0f);
    net.computeGradients();

    EXPECT_NE(gradients[0].gW, gW);
    EXPECT_NE(gradients[0].gb, gb);
}