#pragma once

#include "basicnn/linalg/matrix.hpp"
#include <random>

namespace basicnn::nn::functional{

    // Activation functions
    float identity(const float u);
    basicnn::linalg::Matrix<float> identity(const basicnn::linalg::Matrix<float> & U);

    float sigmoid(const float u);
    basicnn::linalg::Matrix<float> sigmoid(const basicnn::linalg::Matrix<float> & U);

    float relu(const float u);
    basicnn::linalg::Matrix<float> relu(const basicnn::linalg::Matrix<float> & U);

    std::vector<float> softmax(const std::vector<float> & u);
    basicnn::linalg::Matrix<float> softmax(const basicnn::linalg::Matrix<float> & U);


    // Activation function derivatives
    float identity_grad(const float u);
    basicnn::linalg::Matrix<float> identity_grad(const basicnn::linalg::Matrix<float> & U);
    
    float sigmoid_grad(const float u);
    basicnn::linalg::Matrix<float> sigmoid_grad(const basicnn::linalg::Matrix<float> & U);

    float relu_grad(const float u);
    basicnn::linalg::Matrix<float> relu_grad(const basicnn::linalg::Matrix<float> & U);
    

    // Initialization functions
    float glorot(std::mt19937 & rng, const size_t fan_in, const size_t fan_out);
    float he(std::mt19937 & rng, const size_t fan_in, const size_t);


    // Loss functions
    float crossEntropy(basicnn::linalg::Matrix<float> Yhat, basicnn::linalg::Matrix<float> Ytrue);
    basicnn::linalg::Matrix<float> crossEntropy_grad(basicnn::linalg::Matrix<float> Yhat, basicnn::linalg::Matrix<float> Ytrue);
}