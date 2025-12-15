#pragma once

#include "basicnn/linalg/matrix.hpp"
#include <vector>
#include <random>

namespace basicnn::nn{

    enum class NeuronType{

        INPUT,
        HIDDEN,
        OUTPUT
    };

    enum class ActivationType{

        SIGMOID,
        RELU,
        SOFTMAX
    };

    struct LayerWeights{

        basicnn::linalg::Matrix<float> & W;
        std::vector<float> & b;
    };

    struct LayerGradients{

        basicnn::linalg::Matrix<float> & gW;
        std::vector<float> & gb;
    };

    class Layer{

        public:

            Layer(NeuronType type, size_t size, ActivationType activationType);
            Layer(NeuronType type, size_t size) : 
                Layer(type, size, ActivationType::SIGMOID) {}
            Layer() { size_ = 0; }

            Layer(const Layer&) = delete;
            Layer& operator=(const Layer&) = delete;
            Layer(Layer&&) noexcept = default;
            Layer& operator=(Layer&&) noexcept = default;
            ~Layer() = default;

            NeuronType getType() const { return type_; }
            size_t getSize() const { return size_; }
            ActivationType getActivationType() const { return activationType_; }
            LayerWeights getWeights() { return {W_, B_.vector()}; }
            LayerGradients getGradients() { return {gW_,gB_.vector()}; }
            basicnn::linalg::Matrix<float> getActivation() const { return Z_; }
            basicnn::linalg::Matrix<float> getDelta() const { return D_; }

            void setNeuronType(const NeuronType type) { type_ = type; }
            void setSize(const size_t size) { size_ = size; }
            void setActivationType(const ActivationType activationType);
            void setParameters(const NeuronType type,
                               const size_t size,
                               const ActivationType activationType);
            void setWeights(const basicnn::linalg::Matrix<float> & W, const std::vector<float> & b);
            
            bool initialized() const;
            void weightInit(std::mt19937 & rng, size_t fan_in, size_t fan_out);
            std::vector<float> forward(const std::vector<float> & z_kminus1);
            basicnn::linalg::Matrix<float> forward(const basicnn::linalg::Matrix<float> & Z_kminus1);
            void backward(const basicnn::linalg::Matrix<float> & W_kplus1,
                          const basicnn::linalg::Matrix<float> & D_kplus1);
            void computeGradients(const basicnn::linalg::Matrix<float> & Z_kminus1);

        private:

            NeuronType type_;
            size_t size_;
            ActivationType activationType_;
            basicnn::linalg::Matrix<float> (*activation_)(const basicnn::linalg::Matrix<float> &); // Function pointer
            basicnn::linalg::Matrix<float> (*activationGrad_)(const basicnn::linalg::Matrix<float> &); // Function pointer

            // Weights
            basicnn::linalg::Matrix<float> W_;
            basicnn::linalg::Matrix<float> B_;

            // Activations
            basicnn::linalg::Matrix<float> Z_;
            basicnn::linalg::Matrix<float> U_;

            // Backward delta
            basicnn::linalg::Matrix<float> D_;

            // Gradients
            basicnn::linalg::Matrix<float> gW_;
            basicnn::linalg::Matrix<float> gB_;
    };
}