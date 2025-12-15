#include "basicnn/data/dataset.hpp"
#include "basicnn/nn/training.hpp"
#include "basicnn/nn/neural_network.hpp"
#include "basicnn/nn/optimization.hpp"
#include "basicnn/nn/functional.hpp"
#include "basicnn/utils.hpp"
#include <iostream>
#include <chrono>
#include <ctime>
#include <algorithm>

namespace basicnn::nn::optimization{

    std::pair<std::vector<float>, std::vector<float>> train_model(
        basicnn::nn::NeuralNetwork & model,
        basicnn::data::Dataset<std::vector<ftype>, size_t> & trainset,
        basicnn::nn::optimization::Optimizer & optimizer,
        basicnn::nn::optimization::LRScheduler & lrschedule,
        float (*train_loss_func)(basicnn::linalg::Matrix<float> ,basicnn::linalg::Matrix<float>),
        basicnn::linalg::Matrix<float> (*train_loss_grad)(basicnn::linalg::Matrix<float> ,basicnn::linalg::Matrix<float>),
        basicnn::data::Dataset<std::vector<ftype>, size_t> & valset,
        float (*val_loss_func)(basicnn::linalg::Matrix<float> ,basicnn::linalg::Matrix<float>),
        size_t epochs,
        size_t N_classes,
        basicnn::nn::optimization::EarlyStop & earlystop
    ) {

        using namespace basicnn::linalg;
        using namespace basicnn::utils;

        size_t train_size = trainset.size();
        size_t val_size = valset.size();

        std::vector<float> train_loss;
        std::vector<float> val_loss;
        train_loss.reserve(epochs);
        val_loss.reserve(epochs);

        std::cout << "Starting training ..." << std::endl;

        auto start = std::chrono::system_clock::now();

        for(size_t epoch=0; epoch<epochs; ++epoch){

            std::cout << "Epoch " << epoch + 1 << ", ";

            // Training
            model.setStatus(Status::TRAINING);
            trainset.shuffle();
            std::vector<float> train_loss_batch;
            train_loss_batch.reserve(train_size);

            for(size_t batch_id=0; batch_id<train_size; ++batch_id){

                // Forward
                std::pair<Matrix<float>, std::vector<size_t>> batch = trainset.batch(batch_id);
                Matrix<float> X = batch.first;
                std::vector<size_t> ytrue = batch.second;
                Matrix<float> Ytrue = onehot<float>(ytrue, N_classes);
                Matrix<float> Yhat = model.forward(X.transpose());
                train_loss_batch.push_back(train_loss_func(Yhat, Ytrue));

                // Backward
                Matrix<float> G = train_loss_grad(Yhat, Ytrue);
                model.backward(G);
                model.computeGradients();
                optimizer.update();

            }

            train_loss.push_back(average(train_loss_batch));

            // Validation
            model.setStatus(Status::VALIDATING);
            std::vector<float> val_loss_batch;
            val_loss_batch.reserve(val_size);

            for(size_t batch_id=0; batch_id<val_size; ++batch_id){

                std::pair<Matrix<float>, std::vector<size_t>> batch = valset.batch(batch_id);
                Matrix<float> X = batch.first;
                std::vector<size_t> ytrue = batch.second;
                Matrix<float> Ytrue = onehot<float>(ytrue, N_classes);
                Matrix<float> Yhat = model.forward(X.transpose());
                val_loss_batch.push_back(val_loss_func(Yhat, Ytrue));

            }

            val_loss.push_back(average(val_loss_batch));

            std::cout << "training loss: " << train_loss[epoch] << ", ";
            std::cout << "validation loss: " << val_loss[epoch] << std::endl;

            // LR scheduler
            lrschedule.step();

            // Early stop
            std::vector<LayerWeights> weights = model.getWeights();
            earlystop.epoch(weights, val_loss[epoch]);
            if(earlystop.stop()) {

                std::cout << "No improvement after " << earlystop.getPatience()
                    << " epochs, training stopped" << std::endl;

                break;
            }

        }

        std::cout << "Restoring weights of epoch " << earlystop.bestEpoch() + 1 << std::endl;
        auto [W, b] = earlystop.bestWeights();
        model.setWeights(W, b);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "Training performed in: ";
        double seconds = elapsed_seconds.count();
        size_t minutes;
        if (seconds >= 60) {
            std::cout << floor(seconds / 60) << "m " << std::fmod(seconds,60) << "s" << std::endl;
        } else {
            std::cout << seconds << "s" << std::endl;
        }

        return {train_loss, val_loss};
    }

    std::pair<std::vector<size_t>, std::vector<size_t>> get_predictions(
        basicnn::nn::NeuralNetwork & model,
        basicnn::data::Dataset<std::vector<ftype>, size_t> & testset,
        size_t N_classes
    ) {
        
        using namespace basicnn::utils;
        using namespace basicnn::linalg;

        model.setStatus(Status::VALIDATING);
        size_t test_size = testset.size();

        std::vector<size_t> ytrue;
        std::vector<size_t> yhat;

        ytrue.reserve(test_size);
        yhat.reserve(test_size);

        for(size_t i=0; i<test_size; ++i){

            std::pair<Matrix<float>, std::vector<size_t>> batch = testset.batch(i);
            Matrix<float> X = batch.first;
            Matrix<float> Prob = model.forward(X.transpose());

            size_t ytrue1 = batch.second[0];
            std::vector<float> prob = Prob.data();
            size_t yhat1 = std::distance(prob.begin(),std::max_element(prob.begin(), prob.end()));

            ytrue.push_back(ytrue1);
            yhat.push_back(yhat1);
        }

        return {ytrue, yhat};
    }
}