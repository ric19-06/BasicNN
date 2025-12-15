#pragma once

#include "basicnn/data/dataset.hpp"
#include "basicnn/nn/neural_network.hpp"
#include "basicnn/nn/optimization.hpp"
#include "basicnn/nn/functional.hpp"
#include <vector>

using ftype = float;

namespace basicnn::nn::optimization{

    std::pair<std::vector<float>, std::vector<float>> train_model(
        basicnn::nn::NeuralNetwork & model,
        basicnn::data::Dataset<std::vector<ftype>, size_t> & trainset,
        basicnn::nn::optimization::Optimizer & optimizer,
        basicnn::nn::optimization::LRScheduler & lrschedule,
        float (*train_loss)(basicnn::linalg::Matrix<float> ,basicnn::linalg::Matrix<float>),
        basicnn::linalg::Matrix<float> (*train_loss_grad)(basicnn::linalg::Matrix<float> ,basicnn::linalg::Matrix<float>),
        basicnn::data::Dataset<std::vector<ftype>, size_t> & valset,
        float (*val_loss)(basicnn::linalg::Matrix<float> ,basicnn::linalg::Matrix<float>),
        size_t epochs,
        size_t N_classes,
        basicnn::nn::optimization::EarlyStop & earlystop
    );

    std::pair<std::vector<size_t>, std::vector<size_t>> get_predictions(
        basicnn::nn::NeuralNetwork & model,
        basicnn::data::Dataset<std::vector<ftype>, size_t> & testset,
        size_t N_classes
    );
}