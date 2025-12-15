#pragma once

#include "basicnn/linalg/matrix.hpp"
#include "basicnn/data/data_loader.hpp"
#include <ostream>
#include <vector>
#include <memory>

namespace basicnn::utils{

    // Vectors
    template <std::floating_point T>
    std::vector<T> zeros(size_t size);

    template <std::floating_point T>
    std::vector<T> ones(size_t size);

    template <std::floating_point T>
    T average(std::vector<T> v);

    std::vector<size_t> range(size_t n);


    // Matrices
    template <std::floating_point T>
    basicnn::linalg::Matrix<T> zeros(size_t rows, size_t cols);

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> ones(size_t rows, size_t cols);

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> identity(size_t size);

    template <std::floating_point T>
    T sum(basicnn::linalg::Matrix<T> M);

    template <std::floating_point T>
    T average(basicnn::linalg::Matrix<T> M);


    // Dataset
    std::pair<std::vector<size_t>, std::vector<size_t>> split(
        std::shared_ptr<basicnn::data::DataLoader<std::vector<float>, size_t>> dataloader, 
        size_t N_classes, double p, size_t seed);

    template <std::floating_point T>
    std::vector<T> onehot(size_t label, size_t N_classes);

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> onehot(std::vector<size_t> labels, size_t N_classes);


    // ostream overloads
    template <std::floating_point T>
    std::ostream & operator<<(std::ostream & os, const std::vector<T> & v);

    template <std::floating_point T>
    std::ostream & operator<<(std::ostream & os, const basicnn::linalg::Matrix<T> & m);
}

#include "utils.tpp"