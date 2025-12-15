#include "basicnn/linalg/matrix.hpp"
#include "basicnn/linalg/linalg.hpp"
#include "basicnn/nn/optimization.hpp"
#include "basicnn/utils.hpp"
#include <vector>
#include <iostream>

#define INVALID_ARGUMENT(msg) cpptrace::invalid_argument::exception_with_message(msg);

namespace basicnn::nn::optimization{

    // Optimizers
    SGD::SGD(std::vector<LayerWeights> & netW, std::vector<LayerGradients> & netg, float & lr) : 
        netW_(netW), netg_(netg), lr_(lr){

            if (netW.size() != netg.size())
                throw INVALID_ARGUMENT("Network weight and gradient vector sizes should match.");
            
            size_ = netW.size();
    }

    void SGD::update(basicnn::linalg::Matrix<float> & W, std::vector<float> & b,
                        basicnn::linalg::Matrix<float> & gW, std::vector<float> & gb){

        using namespace basicnn::linalg;

        W = W - lr_ * gW;
        b = b - lr_ * gb;
    }

    void SGD::update(){

        for(size_t i=0; i<size_; ++i){

            auto [W, b] = netW_[i];
            auto [gW, gb] = netg_[i];
            update(W, b, gW, gb);
        }
    }

    ADAM::ADAM(std::vector<LayerWeights> & netW, std::vector<LayerGradients> & netg,
        float & lr, float beta1, float beta2, float eps, float weight_decay) : 
            netW_(netW), netg_(netg), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) {

            using namespace basicnn::utils;

            if (netW.size() != netg.size())
                throw INVALID_ARGUMENT("Network weight and gradient vector sizes should match.");

            size_ = netW.size();
            pW_.reserve(size_);
            sW_.reserve(size_);
            pb_.reserve(size_);
            sb_.reserve(size_);

            for(size_t i=0; i<size_; ++i){

                auto [W, b] = netW_[i];
                pW_.push_back(zeros<float>(W.rows(), W.cols()));
                sW_.push_back(zeros<float>(W.rows(), W.cols()));
                pb_.push_back(zeros<float>(b.size()));
                sb_.push_back(zeros<float>(b.size()));
            }

            t_ = 0.0f;
    }

    void ADAM::update(basicnn::linalg::Matrix<float> & W, std::vector<float> & b,
                         basicnn::linalg::Matrix<float> & gW, std::vector<float> & gb,
                         basicnn::linalg::Matrix<float> & pW, basicnn::linalg::Matrix<float> & sW,
                         std::vector<float> & pb, std::vector<float> & sb){
        
        using namespace basicnn::linalg;
        using namespace basicnn::utils;

        // Apply weight decay
        if(weight_decay_ > 0.0f) {
            gW = gW + weight_decay_ * W;
        }

        // Update momentums
        pW = beta1_ * pW + (1 - beta1_) * gW;
        pb = beta1_ * pb + (1 - beta1_) * gb;
        sW = beta2_ * sW + (1 - beta2_) * pow(gW, 2.0f);
        sb = beta2_ * sb + (1 - beta2_) * pow(gb, 2.0f);

        // Bias corrections
        Matrix<float> pW_hat = pW / (1 - std::pow(beta1_, t_));
        std::vector<float> pb_hat = pb / (1 - std::pow(beta1_, t_));
        Matrix<float> sW_hat = sW / (1 - std::pow(beta2_, t_));
        std::vector<float> sb_hat = sb / (1 - std::pow(beta2_, t_));

        // Update weights
        W = W - lr_ * pW_hat / (sqrt(sW_hat) + eps_ * ones<float>(W.rows(), W.cols()));
        b = b - lr_ * pb_hat / (sqrt(sb_hat) + eps_ * ones<float>(b.size()));
    }

    void ADAM::update(){

        // Update step
        t_++;

        for(size_t i=0; i<size_; ++i){

            auto [W, b] = netW_[i];
            auto [gW, gb] = netg_[i];
            update(W, b, gW, gb, pW_[i], sW_[i], pb_[i], sb_[i]);
        }
    }


    // LRSchedulers
    StepLR::StepLR(float & lr, size_t step_size, float gamma) : 
        lr_(lr), step_size_(step_size), gamma_(gamma) { t_=0; }

    void StepLR::step(){

        t_++;
        if (t_ % step_size_ == 0) {

            lr_ = lr_ * gamma_;
        }
    }

    // Earlystop
    size_t EarlyStop::bestEpoch(){

        size_t best_id = 0;
        for(size_t i=0; i<loss_.size(); ++i){

            if (loss_[i] < loss_[best_id])
                best_id = i;
        }

        return best_id;
    }

    std::pair<std::vector<basicnn::linalg::Matrix<float>>, std::vector<std::vector<float>>> EarlyStop::bestWeights(){

        size_t best_id = bestEpoch();
        return {W_[best_id], b_[best_id]};
    }

    void EarlyStop::epoch(std::vector<LayerWeights> & netW, float loss){

        using namespace basicnn::linalg;

        std::vector<Matrix<float>> W_copy;
        std::vector<std::vector<float>> b_copy;
        W_copy.reserve(netW.size());
        b_copy.reserve(netW.size());
        for(size_t i=0; i<netW.size(); ++i){

            auto [W_net, b_net] = netW[i];
            Matrix<float> W = W_net;
            std::vector<float> b = b_net;
            W_copy.push_back(W);
            b_copy.push_back(b);
        }

        W_.push_back(W_copy);
        b_.push_back(b_copy);
        loss_.push_back(loss);
    }

    bool EarlyStop::stop() const {

        size_t cnt = 0;
        for(size_t i=1; i<loss_.size(); ++i){

            float loss_diff = loss_[i-1] - loss_[i];
            bool negative = loss_diff < 0;
            bool lowerMinDelta = abs(loss_diff) < min_delta_;

            if (negative || lowerMinDelta){
                cnt++;
            } else {
                cnt = 0;
            }
        }

        return cnt > patience_;
    }
}