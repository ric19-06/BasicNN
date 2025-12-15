#pragma once

#include "basicnn/linalg/matrix.hpp"
#include "layer.hpp"
#include <vector>
#include <memory>

namespace basicnn::nn{

    enum class Status{

        TRAINING,
        VALIDATING
    };

    class NeuralNetwork{

        public:

            NeuralNetwork(std::vector<size_t> sizes, std::vector<ActivationType> activations, float droprate);
            NeuralNetwork(std::vector<size_t> sizes, std::vector<ActivationType> activations) : 
                NeuralNetwork(sizes, activations, 0.0f) {}
            NeuralNetwork(std::vector<size_t> sizes, std::vector<ActivationType> activations, float droprate, size_t seed);

            NeuralNetwork(const NeuralNetwork&) = delete;
            NeuralNetwork& operator=(const NeuralNetwork&) = delete;
            NeuralNetwork(NeuralNetwork&&) noexcept = default;
            NeuralNetwork& operator=(NeuralNetwork&&) noexcept = default;
            ~NeuralNetwork() = default;

            size_t getSize() { return size_; }
            std::vector<size_t> getSizes() { return sizes_; }
            std::vector<ActivationType> getActivations() { return activations_; }
            std::vector<LayerWeights> getWeights();
            std::vector<LayerGradients> getGradients();

            void setStatus(Status status) { status_ = status; }
            void setSeed(size_t seed);
            void setWeights(const std::vector<basicnn::linalg::Matrix<float>> & W,
                            const std::vector<std::vector<float>> & b);

            bool training() const { return status_ == Status::TRAINING; }
            bool initialized() const;
            void weightInit();
            std::vector<float> forward(const std::vector<float> & z);
            basicnn::linalg::Matrix<float> forward(const basicnn::linalg::Matrix<float> & Z);
            void backward(const basicnn::linalg::Matrix<float> & G);
            void computeGradients();

        private:
        
            size_t size_;
            std::vector<size_t> sizes_;
            std::vector<ActivationType> activations_;
            float droprate_;
            Status status_;
            std::vector<std::unique_ptr<Layer>> layers_;
            std::mt19937 rng_;

            void makeLayers();
    };

}