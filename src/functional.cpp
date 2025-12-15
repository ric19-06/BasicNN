#include "basicnn/nn/functional.hpp"
#include "basicnn/linalg/matrix.hpp"
#include "basicnn/linalg/linalg.hpp"
#include "basicnn/utils.hpp"
#include <random>
#include <vector>
#include <cmath>

namespace basicnn::nn::functional{

    template <typename Function>
    basicnn::linalg::Matrix<float> applyFunc(const basicnn::linalg::Matrix<float> & U, Function f){

        std::vector<float> data;
        for(size_t i=0; i<U.size(); ++i){

            data.push_back( f( U(i) ) );
        }

        return basicnn::linalg::Matrix<float> (data, U.rows(), U.cols());
    }


    // Activation functions
    float identity(const float u){ return u; }
    basicnn::linalg::Matrix<float> identity(const basicnn::linalg::Matrix<float> & U){

        float (*func)(float) = & identity;
        return applyFunc(U, func);
    };

    float sigmoid(const float u){ return 1.0f / (1.0f + std::exp(-u)); }
    basicnn::linalg::Matrix<float> sigmoid(const basicnn::linalg::Matrix<float> & U){

        float (*func)(float) = & sigmoid;
        return applyFunc(U, func);
    }

    float relu(const float u){ return std::max(0.0f, u); }
    basicnn::linalg::Matrix<float> relu(const basicnn::linalg::Matrix<float> & U){

        float (*func)(float) = & relu;
        return applyFunc(U, func);
    }

    std::vector<float> softmax(const std::vector<float> & u){

        using namespace basicnn::linalg;
        
        float max = *std::max_element(u.begin(), u.end());
        float exp_sum = 0.0f;

        std::vector<float> exps;
        exps.reserve(u.size());
        for(size_t i=0; i<u.size(); ++i){

            exps.push_back(std::exp(u[i] - max));
            exp_sum += exps[i];
        }

        return exps / exp_sum;
    }

    basicnn::linalg::Matrix<float> softmax(const basicnn::linalg::Matrix<float> & U){

        using namespace basicnn::linalg;

        // [ N x C ]
        Matrix<float> Z_T;
        Z_T.reserve(U.rows(), U.cols());

        // [ N x C ]
        Matrix<float> U_T = U.transpose();

        for(size_t i=0; i<U_T.rows(); ++i){

            std::vector<float> row;
            for(size_t j=0; j<U_T.cols(); ++j){

                row.push_back(U_T(i,j));
            }
            Z_T.push_back(softmax(row));
        }

        // [ C x N ]
        return Z_T.transpose();
    }


    // Activation function derivatives
    float identity_grad(const float u){ return 1.0f; }
    basicnn::linalg::Matrix<float> identity_grad(const basicnn::linalg::Matrix<float> & U){

        float (*func)(float) = & identity_grad;
        return applyFunc(U, func);
    }

    float sigmoid_grad(const float u){ return (std::exp(-u) / std::pow(1.0f + std::exp(-u), 2.0f)); }
    basicnn::linalg::Matrix<float> sigmoid_grad(const basicnn::linalg::Matrix<float> & U){

        float (*func)(float) = & sigmoid_grad;
        return applyFunc(U, func);
    }

    float relu_grad(const float u){ return (u > 0.0f); }
    basicnn::linalg::Matrix<float> relu_grad(const basicnn::linalg::Matrix<float> & U){

        float (*func)(float) = & relu_grad;
        return applyFunc(U, func);
    }


    // Initialization functions
    float glorot(std::mt19937 & rng, const size_t fan_in, const size_t fan_out){

        float b = std::sqrt(6.0f / (float(fan_out) + float(fan_in)));
        std::uniform_real_distribution<float> dist(-b, b);

        return dist(rng);
    }

    float he(std::mt19937 & rng, const size_t fan_in, const size_t){

        float sigma = std::sqrt(2.0f / float(fan_in));
        std::normal_distribution<float> dist(0.0f, sigma);

        return dist(rng);
    }


    // Loss functions
    float crossEntropy(basicnn::linalg::Matrix<float> Yhat, basicnn::linalg::Matrix<float> Ytrue){

        using namespace basicnn::linalg;
        using namespace basicnn::utils;

        if (Yhat.rows() != Ytrue.rows() || Yhat.cols() != Ytrue.cols())
            throw INVALID_ARGUMENT("Input paramteres should have the same shape.");

        float N = Yhat.cols();
        float eps = 1e-8;
        Matrix<float> Eps = eps * ones<float>(Yhat.rows(), Yhat.cols());
        Matrix<float> CE = hadamard(Ytrue, log(Yhat + Eps));
        float loss = -sum(CE) / N;

        return loss;
    }

    basicnn::linalg::Matrix<float> crossEntropy_grad(basicnn::linalg::Matrix<float> Yhat, basicnn::linalg::Matrix<float> Ytrue){

        using namespace basicnn::linalg;

        if (Yhat.rows() != Ytrue.rows() || Yhat.cols() != Ytrue.cols())
            throw INVALID_ARGUMENT("Input paramteres should have the same shape.");

        Matrix<float> G = Yhat - Ytrue;

        return G;
    }
}