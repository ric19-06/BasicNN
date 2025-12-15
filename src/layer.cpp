#include "basicnn/nn/layer.hpp"
#include "basicnn/nn/functional.hpp"
#include "basicnn/linalg/matrix.hpp"
#include "basicnn/linalg/linalg.hpp"
#include "basicnn/utils.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <cpptrace/cpptrace.hpp>

#define INVALID_ARGUMENT(msg) cpptrace::invalid_argument::exception_with_message(msg);
#define DOMAIN_ERROR(msg) cpptrace::domain_error::exception_with_message(msg);

namespace basicnn::nn{

    Layer::Layer(NeuronType type, size_t size, ActivationType activationType) : 
        type_(type), size_(size) {

            using namespace basicnn::linalg;
            using namespace basicnn::utils;

            if (type_ != NeuronType::INPUT){
                setActivationType(activationType);
                Matrix<float> W = ones<float>(size_, 1);
                std::vector<float> b = zeros<float>(size_);
                setWeights(W, b);
            }
    }

    void Layer::setActivationType(const ActivationType activationType){

        using namespace basicnn::nn::functional;

        if (!(type_ == NeuronType::INPUT ||
              type_ == NeuronType::HIDDEN ||
              type_ == NeuronType::OUTPUT)){

            throw DOMAIN_ERROR("NeuronType is not defined.");
        }

        if(type_ == NeuronType::INPUT)
            throw INVALID_ARGUMENT("Cannot set an activation function to an input neuron.");

        activationType_ = activationType;

        switch (activationType_) {

            case ActivationType::SIGMOID:
                activation_ = & sigmoid;
                activationGrad_ = & sigmoid_grad;
                break;
            
            case ActivationType::RELU:
                activation_ = & relu;
                activationGrad_ = & relu_grad;
                break;
            
            case ActivationType::SOFTMAX:
                activation_ = & softmax;
                // activationGrad_ is not used for the output layer
                break;
        }
    }

    void Layer::setParameters(const NeuronType type,
                        const size_t size,
                        const ActivationType activationType){
        
        setNeuronType(type);
        setSize(size);
        setActivationType(activationType);
    }

    void Layer::setWeights(const basicnn::linalg::Matrix<float> & W, const std::vector<float> & b){

        if (type_ == NeuronType::INPUT)
            throw DOMAIN_ERROR("Cannot set weights to an input layer.");

        if (W.rows() != b.size())
            throw INVALID_ARGUMENT("W column size and b size must match.");

        W_ = W;
        B_ = basicnn::linalg::Matrix<float> (b, b.size(), 1);
        size_ = B_.rows();
    }

    bool Layer::initialized() const {

        using namespace basicnn::linalg;
        using namespace basicnn::utils;

        if (type_ == NeuronType::INPUT)
            return true;

        return !(W_ == ones<float>(size_, 1) && B_ == zeros<float>(size_, 1));
    }

    void Layer::weightInit(std::mt19937 & rng, size_t fan_in, size_t fan_out){

        using namespace basicnn::nn::functional;

        if (type_ == NeuronType::INPUT)
            throw DOMAIN_ERROR("Cannot initialize weights for an input layer.");

        float (*initFunction)(std::mt19937 &, size_t, size_t);
        switch (activationType_) {

            case ActivationType::SIGMOID:
                initFunction = & glorot;
                break;
            
            case ActivationType::RELU:
                initFunction = & he;
                break;

            case ActivationType::SOFTMAX:
                initFunction = & glorot;
                break;
        }
    
        W_.clear();
        W_.reserve(size_, fan_in);

        for(size_t i=0; i<size_; ++i){

            std::vector<float> row;
            for(size_t j=0; j<fan_in; ++j){

                row.push_back(initFunction(rng, fan_in, fan_out));
            }
            W_.push_back(row);
        }
    }

    std::vector<float> Layer::forward(const std::vector<float> & z_kminus1){

        using namespace basicnn::linalg;

        if (type_ == NeuronType::INPUT) {
            Z_ = Matrix<float>(z_kminus1, z_kminus1.size(), 1);
            return z_kminus1;
        }

        if (W_.cols() != z_kminus1.size())
            throw INVALID_ARGUMENT("Input does not match the declared size.");

        if (!initialized())
            std::cerr << "Warning: forward method called on a non-initialized neuron" << std::endl;

        // [ n(k-1) x 1 ]
        Matrix<float> Z_kminus1(z_kminus1, z_kminus1.size(), 1);

        // [ nk x 1 ] = [ nk x n(k-1) ] * [ n(k-1) x 1 ] + [ nk x 1 ]
        U_ = W_ * Z_kminus1 + B_;

        Z_ = activation_(U_);

        return Z_.vector();
    }

    basicnn::linalg::Matrix<float> Layer::forward(const basicnn::linalg::Matrix<float> & Z_kminus1){

        using namespace basicnn::linalg;
        using namespace basicnn::utils;

        if (type_ == NeuronType::INPUT) {
            Z_ = Z_kminus1;
            return Z_kminus1;
        }

        if (W_.cols() != Z_kminus1.rows())
            throw INVALID_ARGUMENT("Input does not match the declared size.");

        if (!initialized())
            std::cerr << "Warning: forward method called on a non-initialized neuron" << std::endl;

        // [ nk x N ] = [ nk x 1 ] * [ 1 x N ]
        Matrix<float> B = B_ * ones<float>(1, Z_kminus1.cols());
        
        // [ nk x N ] = [ nk x n(k-1) ] * [ n(k-1) x N ] + [ nk x N ]
        U_ = W_ * Z_kminus1 + B;

        Z_ = activation_(U_);

        return Z_;
    }

    void Layer::backward(const basicnn::linalg::Matrix<float> & W_kplus1, const basicnn::linalg::Matrix<float> & D_kplus1){

        using namespace basicnn::linalg;

        if (type_ != NeuronType::OUTPUT && W_kplus1.rows() != D_kplus1.rows())
            throw INVALID_ARGUMENT("Input does not match the declared size.");

        if (U_.empty())
            throw DOMAIN_ERROR("Backward method called before forward.");

        // If the neuron is an output the input delta is the same and W_kplus1 is empty
        if (type_ == NeuronType::OUTPUT){

            // [ C x N ]
            D_ = D_kplus1;

        // Otherwise apply the backpropagation formula
        } else {

            // [ nk x N ] = [ nk x n(k+1) ] * [ n(k+1) x N ] hadamard [ nk x N ]
            D_ = hadamard(W_kplus1.transpose() * D_kplus1, activationGrad_(U_));
        }
    }

    void Layer::computeGradients(const basicnn::linalg::Matrix<float> & Z_kminus1){

        if (type_ == NeuronType::INPUT)
            throw DOMAIN_ERROR("Cannot compute gradients for an input layer.");

        if (D_.empty())
            throw DOMAIN_ERROR("Cannot compute gradients without a backward pass.");

        using namespace basicnn::linalg;
        using namespace basicnn::utils;
        
        float N = D_.cols();

        // [ nk x n(k-1) ] = [ 1 x 1 ] * [ nk x N ] * [ N x n(k-1) ]
        gW_ = (1 / N) * D_ * Z_kminus1.transpose();

        // [ nk x 1 ] = [ 1 x 1 ] * [ nk x N ] * [ N x 1 ]
        gB_ = (1 / N) * D_ * ones<float>(N,1);
    }
}