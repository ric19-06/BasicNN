#include "basicnn/linalg/matrix.hpp"
#include "basicnn/nn/layer.hpp"
#include "basicnn/nn/neural_network.hpp"
#include "basicnn/utils.hpp"
#include <vector>

#define INVALID_ARGUMENT(msg) cpptrace::invalid_argument::exception_with_message(msg);
#define DOMAIN_ERROR(msg) cpptrace::domain_error::exception_with_message(msg);

namespace basicnn::nn{

    void NeuralNetwork::makeLayers(){

        // Add the input layer
        layers_.push_back(
            std::make_unique<Layer>(NeuronType::INPUT, sizes_[0])
        );

        // Add the hidden layers
        for(size_t i=1; i<size_-1; ++i){

            layers_.push_back(
                std::make_unique<Layer>(NeuronType::HIDDEN, sizes_[i], activations_[i-1])
            );
        }

        // Add the output layer
        layers_.push_back(
            std::make_unique<Layer>(NeuronType::OUTPUT, sizes_[size_ - 1], activations_[size_ - 2])
        );
    }

    NeuralNetwork::NeuralNetwork(std::vector<size_t> sizes, std::vector<ActivationType> activations, float droprate) :
        sizes_(sizes), activations_(activations), droprate_(droprate){

            if (sizes_.size()-1  != activations_.size())
                throw INVALID_ARGUMENT("Layer sizes vector should be the same size as the activation vector minus 1.");

            size_ = sizes_.size();
            layers_.reserve(size_);
            setStatus(Status::TRAINING);

            makeLayers();
    }

    NeuralNetwork::NeuralNetwork(std::vector<size_t> sizes, std::vector<ActivationType> activations, float droprate, size_t seed) :
        sizes_(sizes), activations_(activations), droprate_(droprate){

            if (sizes_.size()-1 != activations_.size())
                throw INVALID_ARGUMENT("Layer sizes vector should be the same size as the activation vector minus 1.");

            size_ = sizes_.size();
            layers_.reserve(size_);
            setStatus(Status::TRAINING);
            setSeed(seed);

            makeLayers();
    }

    std::vector<LayerWeights> NeuralNetwork::getWeights(){

        std::vector<LayerWeights> weights;
        weights.reserve(size_);

        for(size_t i=1; i<size_; ++i){

            weights.push_back(layers_[i]->getWeights());
        }

        return weights;
    }

    std::vector<LayerGradients> NeuralNetwork::getGradients(){

        std::vector<LayerGradients> gradients;
        gradients.reserve(size_);

        for(size_t i=1; i<size_; ++i){

            gradients.push_back(layers_[i]->getGradients());
        }

        return gradients;
    }

    void NeuralNetwork::setSeed(size_t seed){

        rng_ = std::mt19937(seed);
    }

    void NeuralNetwork::setWeights(const std::vector<basicnn::linalg::Matrix<float>> & W,
                                      const std::vector<std::vector<float>> & b){
        
        if (W.size() != b.size())
            throw INVALID_ARGUMENT("Input vectors should have the same size.");

        if (W.size() != size_-1)
            throw INVALID_ARGUMENT("Input vectors should match the network size.");
        
        for(size_t i=0; i<size_-1; ++i){

            layers_[i+1]->setWeights(W[i], b[i]);
        }
    }

    bool NeuralNetwork::initialized() const {

        bool init = true;
        for(size_t i=0; i<size_; ++i){

            if (!layers_[i]->initialized())
                init = false;
        }

        return init;
    }

    void NeuralNetwork::weightInit(){

        std::vector<size_t> sizes = sizes_;
        sizes.push_back(0);
        for(size_t i=1; i<size_; ++i){

            size_t fan_in = sizes[i-1];
            size_t fan_out = sizes[i+1];
            layers_[i]->weightInit(rng_, fan_in, fan_out);
        }
    }

    basicnn::linalg::Matrix<float> Dropout(const basicnn::linalg::Matrix<float> & Z, float droprate, std::mt19937 rng){

        using namespace basicnn::utils;
        using namespace basicnn::linalg;

        std::vector<size_t> ids = range(Z.rows());
        size_t N_zeros = droprate * Z.rows();

        Matrix<float> Z_out = Z;
        for(size_t j=0; j<Z.cols(); ++j){

            std::shuffle(ids.begin(), ids.end(), rng);
            for(size_t i=0; i<N_zeros; ++i){

                Z_out(ids[i],j) = 0.0f;
            }
        }

        return Z_out;
    }

    std::vector<float> NeuralNetwork::forward(const std::vector<float> & z){

        using namespace basicnn::linalg;

        std::vector<float> z_out = z;
        for(size_t i=0; i<size_; ++i){

            if (training() && layers_[i]->getType() == NeuronType::HIDDEN)
                z_out = Dropout(Matrix<float>(z_out, z_out.size(), 1), droprate_, rng_).vector();

            z_out = layers_[i]->forward(z_out);
        }

        return z_out;
    }

    basicnn::linalg::Matrix<float> NeuralNetwork::forward(const basicnn::linalg::Matrix<float> & Z){

        basicnn::linalg::Matrix<float> Z_out = Z;
        for(size_t i=0; i<size_; ++i){

            if (training() && layers_[i]->getType() == NeuronType::HIDDEN)
                Z_out = Dropout(Z_out, droprate_, rng_);

            Z_out = layers_[i]->forward(Z_out);
        }

        return Z_out;
    }

    void NeuralNetwork::backward(const basicnn::linalg::Matrix<float> & G){

        using namespace basicnn::linalg;

        layers_[size_-1]->backward(Matrix<float>(),G);
        Matrix<float> D_kplus1 = layers_[size_-1]->getDelta();
        auto [W, b] = layers_[size_-1]->getWeights();
        Matrix<float> W_kplus1 = W;

        for(size_t i=size_-2; i>0; --i){
            
            layers_[i]->backward(W_kplus1, D_kplus1);
            D_kplus1 = layers_[i]->getDelta();
            auto [W, b] = layers_[i]->getWeights();
            W_kplus1 = W;
        }
    }

    void NeuralNetwork::computeGradients(){

        for(size_t i=1; i<size_; ++i){

            layers_[i]->computeGradients(layers_[i-1]->getActivation());
        }
    }
}