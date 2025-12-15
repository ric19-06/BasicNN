#pragma once

#include "data_loader.hpp"
#include <vector>
#include <concepts>
#include <memory>
#include <random>

namespace basicnn::data{

    template <typename Data, typename Label>
    class Dataset{

        public:

            Dataset(std::shared_ptr<DataLoader<Data, Label>> loader, size_t batchsize, size_t seed,
                std::vector<size_t> exclude_ids={});
            ~Dataset() = default;

            size_t size() { return size_; }
            void shuffle();
            std::pair<std::vector<Data>, std::vector<Label>> batch(size_t batch_id);

        private:

            std::shared_ptr<DataLoader<Data, Label>> loader_;
            size_t batchsize_;
            size_t size_;
            std::vector<size_t> indices_;
            std::mt19937 rng_;
    };
}

#include "dataset.tpp"