#pragma once

#include "basicnn/linalg/matrix.hpp"
#include <ostream>
#include <vector>
#include <cpptrace/cpptrace.hpp>
#include <random>

#define INVALID_ARGUMENT(msg) cpptrace::invalid_argument::exception_with_message(msg);

namespace basicnn::utils{

    // Vectors
    template <std::floating_point T>
    std::vector<T> zeros(size_t size){

        std::vector<T> v(size, T(0));

        return v;
    }

    template <std::floating_point T>
    std::vector<T> ones(size_t size){

        std::vector<T> v(size, T(1));

        return v;
    }

    template <std::floating_point T>
    T average(std::vector<T> v){

        T sum = T(0);
        for(size_t i=0; i<v.size(); ++i){

            sum += v[i];
        }
        T avg = sum / v.size();

        return avg;
    }


    // Matrices
    template <std::floating_point T>
    basicnn::linalg::Matrix<T> zeros(size_t rows, size_t cols){

        using namespace basicnn::linalg;

        Matrix<T> M(std::vector<T> (rows * cols, T(0)), rows, cols);

        return M;
    }

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> ones(size_t rows, size_t cols){
        
        using namespace basicnn::linalg;

        Matrix<T> M(std::vector<T> (rows * cols, T(1)), rows, cols);

        return M;
    }

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> identity(size_t size){

        using namespace basicnn::linalg;

        Matrix<T> M = zeros<T>(size,size);
        for(size_t i=0; i<size; ++i){
            for(size_t j=0; j<size; ++j){

                if(i==j)
                    M(i,j) = T(1);
            }
        }

        return M;
    }

    template <std::floating_point T>
    T sum(basicnn::linalg::Matrix<T> M){

        T sum = 0;
        for(size_t i=0; i<M.rows(); ++i){
            for(size_t j=0; j<M.cols(); ++j){

                sum += M(i,j);
            }
        }

        return sum;
    }

    template <std::floating_point T>
    T average(basicnn::linalg::Matrix<T> M){

        T sum = 0;
        for(size_t i=0; i<M.rows(); ++i){
            for(size_t j=0; j<M.cols(); ++j){

                sum += M(i,j);
            }
        }
        T avg = sum / M.size();

        return avg;
    }


    // Dataset
    template <std::floating_point T>
    std::vector<T> onehot(size_t label, size_t N_classes){

        std::vector<T> label_encoded = zeros<T>(N_classes);
        label_encoded[label] = T(1);

        return label_encoded;
    }

    template <std::floating_point T>
    basicnn::linalg::Matrix<T> onehot(std::vector<size_t> labels, size_t N_classes){

        basicnn::linalg::Matrix<T> labels_encoded;
        labels_encoded.reserve(labels.size(), N_classes);

        for(size_t i=0; i<labels.size(); ++i){

            labels_encoded.push_back(onehot<T>(labels[i], N_classes));
        }

        return labels_encoded.transpose();
    }


    // ostream overloads
    template <std::floating_point T>
    std::ostream & operator<<(std::ostream & os, const std::vector<T> & v){

        if (v.empty())
            throw INVALID_ARGUMENT("Vector is empty.");

        os << "{";
        for(size_t i=0; i<v.size()-1; ++i){

            os << v[i] << ", ";
        }
        os << v[v.size()-1] << "}";

        return os;
    }

    template <std::floating_point T>
    std::ostream & operator<<(std::ostream & os, const basicnn::linalg::Matrix<T> & m){

        if (m.empty())
            throw INVALID_ARGUMENT("Matrix is empty.");

        for(size_t i=0; i<m.rows(); ++i){
            os << "| ";
            for(size_t j=0; j<m.cols(); ++j){

                os << m(i,j) << " ";
            }
            if (i < m.rows() - 1)
                os << "|\n";
            else
                os << "|";
        }

        return os;
    }
}