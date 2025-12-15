#pragma once

#include "dataset.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <cpptrace/cpptrace.hpp>
#include <iostream>

#define INVALID_ARGUMENT(msg) cpptrace::invalid_argument::exception_with_message(msg);

namespace basicnn::data{

    template <typename Data, typename Label>
    Dataset<Data, Label>::Dataset(std::shared_ptr<DataLoader<Data, Label>> loader,
        size_t batchsize, size_t seed, std::vector<size_t> exclude_ids) : 

        loader_(loader), batchsize_(batchsize){
            
            rng_ = std::mt19937(seed);
            std::vector<size_t> loader_ids = loader->indices();

            // Exclude ids and populate indices_
            indices_.reserve(loader_ids.size() - exclude_ids.size());
            for(size_t i=0; i<loader_ids.size(); ++i){

                bool match = false;
                std::vector<size_t>::iterator it;
                for(it = exclude_ids.begin(); it != exclude_ids.end(); ++it){

                    if (loader_ids[i] == *it) {

                        match = true;
                        exclude_ids.erase(it);
                        break;
                    }
                }

                if (!match)
                    indices_.push_back(loader_ids[i]);
            }

            // Dataset size
            size_ = floor(indices_.size() / batchsize);
            if (indices_.size() % batchsize_ > 0)
                size_++;
    }

    template <typename Data, typename Label>
    void Dataset<Data, Label>::shuffle(){

        std::shuffle(indices_.begin(), indices_.end(), rng_);
    }

    template <typename Data, typename Label>
    std::pair<std::vector<Data>, std::vector<Label>> Dataset<Data, Label>::batch(size_t batch_id){

        if (batch_id >= size_)
            throw INVALID_ARGUMENT("Argument batch_id should be smaller than the dataset size.");

        std::vector<Data> data;
        std::vector<Label> label;
        data.reserve(batchsize_);
        label.reserve(batchsize_);

        size_t start = batchsize_*batch_id;
        size_t stop = std::min(start + batchsize_, indices_.size());

        for(size_t i=start; i<stop; ++i){

            data.push_back(loader_->data(indices_[i]));
            label.push_back(loader_->label(indices_[i]));
        }

        return {data, label};
    }
}