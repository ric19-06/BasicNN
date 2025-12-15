#pragma once

#include "basicnn/linalg/matrix.hpp"
#include "basicnn/linalg/linalg.hpp"
#include "basicnn/nn/layer.hpp"
#include <vector>
#include <iostream>

namespace basicnn::nn::optimization{

    // Optimizers
    class Optimizer{

        public:

            virtual ~Optimizer() = default;
            
            virtual void update() = 0;
    };

    class SGD : public Optimizer{

        public:

            SGD(std::vector<LayerWeights> & netW, std::vector<LayerGradients> & netg, float & lr);
            ~SGD() override = default;

            void update() override;

        private:

            size_t size_;
            std::vector<LayerWeights> netW_;
            std::vector<LayerGradients> netg_;

            float & lr_;

            void update(basicnn::linalg::Matrix<float> & W, std::vector<float> & b,
                        basicnn::linalg::Matrix<float> & gW, std::vector<float> & gb);
    };

    class ADAM : public Optimizer{

        public:

            ADAM(std::vector<LayerWeights> & netW, std::vector<LayerGradients> & netg, 
                float & lr, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=1e-4);
            ~ADAM() override = default;

            void update() override;

        private:

            size_t size_;
            std::vector<LayerWeights> netW_;
            std::vector<LayerGradients> netg_;

            float & lr_;
            float beta1_, beta2_, eps_, weight_decay_;

            std::vector<basicnn::linalg::Matrix<float>> pW_, sW_; // 1st and 2nd momentum for W
            std::vector<std::vector<float>> pb_, sb_; // 1st and 2nd momentum for b
            float t_; // Update counter

            void update(basicnn::linalg::Matrix<float> & W, std::vector<float> & b,
                        basicnn::linalg::Matrix<float> & gW, std::vector<float> & gb,
                        basicnn::linalg::Matrix<float> & pW, basicnn::linalg::Matrix<float> & sW,
                        std::vector<float> & pb, std::vector<float> & sb);
    };


    // LRSchedulers
    class LRScheduler{

        public:

            virtual ~LRScheduler() = default;

            virtual void step() = 0;
    };

    class StepLR : public LRScheduler{

        public:

            StepLR(float & lr, size_t step_size=20, float gamma=0.5);
            ~StepLR() override = default;

            void step() override;

        private:

            size_t t_;
            float & lr_;
            size_t step_size_;
            float gamma_;
    };


    // Earlystop
    class EarlyStop{

        public:

            EarlyStop(size_t patience, float min_delta = 1e-4) :
                patience_(patience), min_delta_(min_delta) {}
            ~EarlyStop() = default;

            size_t getPatience() { return patience_; }
            size_t bestEpoch();
            std::pair<std::vector<basicnn::linalg::Matrix<float>>, std::vector<std::vector<float>>> bestWeights();
            void epoch(std::vector<LayerWeights> & netW, float loss);
            bool stop() const;

        private:

            size_t patience_;
            float min_delta_;
            std::vector<std::vector<basicnn::linalg::Matrix<float>>> W_;
            std::vector<std::vector<std::vector<float>>> b_;
            std::vector<float> loss_;
    };
}