#include "basicnn/linalg/matrix.hpp"
#include "basicnn/utils.hpp"
#include "basicnn/data/data_loader.hpp"
#include <vector>
#include <random>
#include <iostream>

namespace basicnn::utils{

    std::pair<std::vector<size_t>, std::vector<size_t>> split(
        std::shared_ptr<basicnn::data::DataLoader<std::vector<float>, size_t>> dataloader,
        size_t N_classes, double p, size_t seed){

        std::mt19937 rng(seed);
        std::vector<size_t> indices = dataloader->indices();
        size_t N = indices.size();

        // Reserve
        std::vector<std::vector<size_t>> indices_by_class;
        indices_by_class.reserve(N_classes);
        for(size_t i=0; i<N_classes; ++i){

            std::vector<size_t> class_indices;
            class_indices.reserve(N / N_classes);
            indices_by_class.push_back(class_indices);
        }

        for(size_t i=0; i<N; ++i){

            size_t label = dataloader->label(i);
            indices_by_class[label].push_back(i);
        }

        std::vector<size_t> train_id;
        std::vector<size_t> val_id;
        train_id.reserve(floor(p * N));
        val_id.reserve(N - floor(p * N));

        for(size_t i=0; i<N_classes; ++i){
            
            std::vector<size_t> & class_indices = indices_by_class[i];
            size_t N_indices = class_indices.size();
            size_t N_train = static_cast<size_t>(std::round(N_indices * p));

            std::shuffle(class_indices.begin(), class_indices.end(), rng);

            train_id.insert(train_id.end(), class_indices.begin(),
                        class_indices.begin() + N_train);

            val_id.insert(val_id.end(), class_indices.begin() + N_train,
                        class_indices.end());
        }

        std::shuffle(train_id.begin(), train_id.end(), rng);
        std::shuffle(val_id.begin(), val_id.end(), rng);

        return {train_id, val_id};
    }

    std::vector<size_t> range(size_t n){

        std::vector<size_t> range;
        range.reserve(n);
        for(size_t i=0; i<n; ++i){

            range.push_back(i);
        }

        return range;
    }
}